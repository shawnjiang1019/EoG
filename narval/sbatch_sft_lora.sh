#!/bin/bash
# =============================================================================
# EoG Stage-1 (SFT) -- LoRA fine-tune on Narval, 1 node x 4 A100-40GB.
# Same pipeline as the full-FT script, but trains LoRA adapters (rank 32):
# much lower memory (no OOM), faster, and the output is an ADAPTER, not a full
# model (merge it before testing -- see narval/merge_lora.py).
#   export REPO=$PWD ; sbatch narval/sbatch_sft_lora.sh
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-sftft-lora
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO=${REPO:-$SCRATCH/EoG}
MODEL_DIR=${MODEL_DIR:-$SCRATCH/models/Qwen2.5-7B-Instruct}
DATA=${DATA:-$SCRATCH/eog/data/2wiki_sft_train.parquet}
SAVE=${SAVE:-$SCRATCH/eog/ckpt/2wiki_sft_lora}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
NGPUS=4

# 1) Build + activate the training venv on $SLURM_TMPDIR (NVMe); sets PYTHONPATH
source "$REPO/narval/setup_env.sh"

# 2) Offline + temp on local disk
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$SLURM_TMPDIR

# 3) Stage the base model to node-local NVMe (avoids slow concurrent Lustre reads)
LOCAL_MODEL=$SLURM_TMPDIR/base_model
echo "[stage] copying model to $LOCAL_MODEL"
cp -r "$MODEL_DIR" "$LOCAL_MODEL"
MODEL_DIR=$LOCAL_MODEL

# 4) Build the SFT parquet once if missing (local tokenizer; no internet)
if [ ! -f "$DATA" ]; then
  echo "[sbatch] building SFT parquet -> $DATA"
  mkdir -p "$(dirname "$DATA")"
  python "$REPO/data/kg_qa_sft_process.py" \
    --input  "$REPO/sft_data/2wikimultihop_train_sft.jsonl" \
    --output "$DATA" --model_path "$MODEL_DIR" --max_tokens 12000
fi

# 5) Train (LoRA adapters; base frozen -> fits 40GB easily, no offload needed)
mkdir -p "$SAVE"; cd "$REPO"
torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.max_length=8192 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    optim.lr=2e-5 \
    model.partial_pretrain="$MODEL_DIR" \
    model.lora_rank=$LORA_RANK \
    model.lora_alpha=$LORA_ALPHA \
    model.target_modules=all-linear \
    model.strategy=fsdp2 \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    trainer.default_local_dir="$SAVE" \
    trainer.project_name="eog_sft" \
    trainer.experiment_name="2wiki_qwen7b_lora" \
    trainer.logger=console \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=1

echo "LoRA SFT done -> $SAVE"

# NOTE: output is a LoRA adapter, not a full model. Merge it into the base before
# testing/GRPO:  python narval/merge_lora.py --base <model> --adapter $SAVE/... --out $SAVE/hf
