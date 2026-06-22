#!/bin/bash
# =============================================================================
# Batch-eval a merged SFT checkpoint: Hit@1 / F1 / format-rate over N samples.
# Run once per model (set MODEL + a distinct -J):
#   export REPO=$PWD
#   MODEL=$SCRATCH/eog/ckpt/2wiki_sft_v2/hf   sbatch -J eog-eval-full narval/sbatch_eval_sft.sh
#   MODEL=$SCRATCH/eog/ckpt/2wiki_sft_lora/hf sbatch -J eog-eval-lora narval/sbatch_eval_sft.sh
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-eval-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=%x-%j.out

set -uo pipefail
REPO=${REPO:-$SCRATCH/EoG}
MODEL=${MODEL:?set MODEL=<merged hf dir>, e.g. \$SCRATCH/eog/ckpt/2wiki_sft_v2/hf}
DATA=${DATA:-$SCRATCH/eog/data/2wiki_sft_train.parquet}
N=${N:-300}

source "$REPO/narval/setup_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf TMPDIR=$SLURM_TMPDIR

python "$REPO/narval/eval_sft.py" --model "$MODEL" --data "$DATA" --n "$N"
