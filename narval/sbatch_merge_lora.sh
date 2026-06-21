#!/bin/bash
# =============================================================================
# Merge a LoRA SFT adapter into the base -> standalone HF model. CPU-only,
# disconnect-proof. Run after the LoRA job finishes:
#   export REPO=$PWD ; sbatch narval/sbatch_merge_lora.sh
# Override BASE/SAVE/OUT via env if your paths differ.
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-merge-lora
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out

set -uo pipefail
REPO=${REPO:-$SCRATCH/EoG}
BASE=${BASE:-$SCRATCH/models/Qwen2.5-7B-Instruct}
SAVE=${SAVE:-$SCRATCH/eog/ckpt/2wiki_sft_lora}     # LoRA job's save dir (holds the adapter)
OUT=${OUT:-$SAVE/hf}                               # merged model lands here

source "$REPO/narval/setup_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf TMPDIR=$SLURM_TMPDIR

python "$REPO/narval/merge_lora.py" --base "$BASE" --adapter "$SAVE" --out "$OUT"
echo "done -> $OUT"
