#!/bin/bash
# =============================================================================
# EoG Stage-1 (SFT) on Narval. Submit with:  sbatch narval/sbatch_sft.sh
# One full Narval GPU node = 4x A100-SXM4-40GB.
# Defaults: Qwen2.5-7B-Instruct + LoRA (low-risk fit on 40GB). See notes at bottom.
# =============================================================================
#SBATCH --account=def-YOURPI          # <-- EDIT: your allocation
#SBATCH --job-name=eog-sft-2wiki
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# ---- paths (must match prep_sft.sh) ----
REPO=$SCRATCH/EoG
MODEL_DIR=$SCRATCH/models/Qwen2.5-7B-Instruct
DATA=$SCRATCH/EoG/data/2wiki_sft_train.parquet
SAVE=$SCRATCH/EoG/ckpt/2wiki_sft
NGPUS=4
# ----------------------------------------

module load StdEnv/2023 gcc arrow python/3.11 cuda/12.2
source "$SCRATCH/EoG/venv/bin/activate"

# Compute nodes have no internet -> force offline + keep temp off shared /tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$SCRATCH/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$SLURM_TMPDIR

mkdir -p "$SAVE"
cd "$REPO"

torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.max_length=12288 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    optim.lr=2e-5 \
    model.partial_pretrain="$MODEL_DIR" \
    model.lora_rank=32 \
    model.lora_alpha=32 \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir="$SAVE" \
    trainer.project_name="eog_sft" \
    trainer.experiment_name="2wiki_qwen7b_lora" \
    trainer.logger=console \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true

echo "SFT done. Checkpoints under: $SAVE"

# -----------------------------------------------------------------------------
# NOTES
# * LoRA is the safe first run. To match the paper (full fine-tune) set:
#       model.lora_rank=0  ulysses_sequence_parallel_size=2
#   and expect to need CPU offload (heavier; may still be tight on 40GB).
# * If flash_attn isn't installed, set:  use_remove_padding=false
# * Paper Appendix D uses lr=1e-5; this repo's script ships 2e-5 (kept here).
# * OOM? lower data.max_length (e.g. 8192) and/or data.train_batch_size.
# -----------------------------------------------------------------------------
