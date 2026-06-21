#!/bin/bash
# =============================================================================
# Merge the SFT checkpoint -> HF, then generate on a few samples to confirm the
# model emits <think>...</think><answer>...</answer>. Disconnect-proof (batch).
#   export REPO=$PWD ; sbatch narval/sbatch_test_sft.sh
# Override paths if needed:  --export=ALL,SAVE=...,DATA=...
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-test-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=%x-%j.out

set -uo pipefail
REPO=${REPO:-$SCRATCH/EoG}
SAVE=${SAVE:-$SCRATCH/eog/ckpt/2wiki_sft_fullft}
DATA=${DATA:-$SCRATCH/eog/data/2wiki_sft_train.parquet}
HF=$SAVE/hf

source "$REPO/narval/setup_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf TMPDIR=$SLURM_TMPDIR

# 1) Merge sharded FSDP checkpoint -> HF (skip if already done)
if [ ! -f "$HF/config.json" ]; then
  HFCFG=$(find "$SAVE" -name huggingface -type d 2>/dev/null | sort -V | tail -1)
  if [ -z "$HFCFG" ]; then
    echo "ERROR: no huggingface/ config dir under $SAVE. Checkpoint tree:"
    ls -R "$SAVE" | head -60
    exit 1
  fi
  CKPT=$(dirname "$HFCFG")
  echo "[merge] $CKPT -> $HF"
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$HF"
fi

# 2) Quick generation sanity test
echo "[test] generating from $HF"
python "$REPO/narval/test_sft.py" --model "$HF" --data "$DATA" --n 3
