#!/bin/bash
# =============================================================================
# EoG Stage-1 (SFT) -- FULL fine-tune (paper-faithful) on Narval.
# Option A: single node, 4x A100-40GB, FSDP2 FULL_SHARD, CPU offload OFF.
# Submit with:  sbatch narval/sbatch_sft_fullft.sh
#
# Why offload is OFF by default: the 2Wiki SFT data is short (~2.3k tokens avg,
# p90 ~3.3k), so activations are tiny and the ~28 GB/GPU of FULL_SHARD optimizer
# state fits 40 GB without offload. If it OOMs, flip the two offload lines below.
# =============================================================================
#SBATCH --account=def-YOURPI          # <-- EDIT: your allocation
#SBATCH --job-name=eog-sftft-2wiki
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --time=04:00:00                # est ~1.5-3 h; resume_mode=auto if killed
#SBATCH --output=%x-%j.out

set -euo pipefail

# ---- paths (must match prep_sft.sh) ----
REPO=$SCRATCH/EoG
MODEL_DIR=$SCRATCH/models/Qwen2.5-7B-Instruct
DATA=$SCRATCH/EoG/data/2wiki_sft_train.parquet
SAVE=$SCRATCH/EoG/ckpt/2wiki_sft_fullft
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

echo "Full-FT SFT done. Checkpoints under: $SAVE"

# -----------------------------------------------------------------------------
# NOTES
# * OOM fallback (enable CPU offload, ~2-3x slower):
#       model.fsdp_config.cpu_offload=True  model.fsdp_config.offload_params=True
#   ...and/or lower data.max_length=4096 (covers p90 for 2Wiki).
# * Checkpoints include optimizer state (large). max_ckpt_to_keep=1 keeps only the
#   latest to protect $SCRATCH quota; the final checkpoint is always written.
# * The saved checkpoint is a sharded FSDP checkpoint. To feed Stage-2 (GRPO) or
#   run inference, convert to HF format with verl's model_merger, e.g.:
#       python -m verl.model_merger merge --backend fsdp \
#           --local_dir "$SAVE/global_step_<N>" --target_dir "$SAVE/hf"
# * Paper Appendix D uses lr=1e-5; this repo's script ships 2e-5 (kept here).
# -----------------------------------------------------------------------------
