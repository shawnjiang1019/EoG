#!/bin/bash
# =============================================================================
# EoG Stage-1 (SFT) -- FULL fine-tune on Narval, 1 node x 4 A100-40GB.
# Builds its venv on node-local NVMe (no Lustre pain), builds the SFT parquet if
# missing, then trains. Submit with:  sbatch narval/sbatch_sft_fullft.sh
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-sftft-2wiki
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --time=08:00:00                # venv build + model stage + parquet + ~1.5-3 h train
#SBATCH --output=%x-%j.out

set -euo pipefail

# REPO can live anywhere (project space recommended): `export REPO=$PWD` before
# sbatch from your clone. Big artifacts stay on $SCRATCH, never under the repo.
REPO=${REPO:-$SCRATCH/EoG}
MODEL_DIR=$SCRATCH/models/Qwen2.5-7B-Instruct
DATA=$SCRATCH/eog/data/2wiki_sft_train.parquet
SAVE=$SCRATCH/eog/ckpt/2wiki_sft_fullft
NGPUS=4

# 1) Build + activate the training venv on $SLURM_TMPDIR (NVMe); sets PYTHONPATH
source "$REPO/narval/setup_env.sh"

# 2) Offline (no internet on compute nodes) + temp on local disk
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$SLURM_TMPDIR

# 2b) Stage the base model to node-local NVMe. Loading 15GB from $SCRATCH (Lustre)
#     with 4 ranks reading concurrently is pathologically slow; a one-time
#     sequential copy + local reads fixes it.
LOCAL_MODEL=$SLURM_TMPDIR/base_model
echo "[stage] copying $MODEL_DIR -> $LOCAL_MODEL"
cp -r "$MODEL_DIR" "$LOCAL_MODEL"
MODEL_DIR=$LOCAL_MODEL

# 3) Build the SFT parquet once (uses local tokenizer; no internet needed)
if [ ! -f "$DATA" ]; then
  echo "[sbatch] building SFT parquet -> $DATA"
  mkdir -p "$(dirname "$DATA")"
  python "$REPO/data/kg_qa_sft_process.py" \
    --input  "$REPO/sft_data/2wikimultihop_train_sft.jsonl" \
    --output "$DATA" --model_path "$MODEL_DIR" --max_tokens 12000
fi

# 4) Train (full fine-tune, FSDP2 FULL_SHARD, offload OFF -- short seqs fit 40GB)
mkdir -p "$SAVE"; cd "$REPO"
torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.max_length=8192 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    optim.lr=2e-5 \
    model.partial_pretrain="$MODEL_DIR" \
    model.lora_rank=0 \
    model.strategy=fsdp2 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    trainer.default_local_dir="$SAVE" \
    trainer.project_name="eog_sft" \
    trainer.experiment_name="2wiki_qwen7b_fullft" \
    trainer.logger=console \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=1 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true

echo "Full-FT SFT done -> $SAVE"

# NOTES
# * OOM fallback: model.fsdp_config.cpu_offload=True offload_params=True  and/or data.max_length=4096
# * If flash_attn isn't usable: use_remove_padding=false
# * Verify the env first on an interactive node (catches missing wheels / flash_attn):
#     salloc --account=def-YOURPI --gres=gpu:a100:1 --cpus-per-task=6 --mem=32G --time=0:30:00
#     export REPO=$SCRATCH/EoG && source $REPO/narval/setup_env.sh
#     python -c "import torch,flash_attn,verl,transformers; print('env OK', torch.__version__)"
