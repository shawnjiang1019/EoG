#!/bin/bash
# =============================================================================
# Disconnect-proof env check: builds the training venv on NVMe and imports the
# key packages on a GPU. Runs server-side (sbatch), so SSH drops don't matter.
#   sbatch narval/sbatch_verify_env.sh
#   # then read the log when it finishes:
#   cat eog-verify-env-<jobid>.out
# A clean "ENV OK ..." line at the end means you're ready to submit the SFT job.
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-verify-env
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out

set -uo pipefail
export REPO=$SCRATCH/EoG

source "$REPO/narval/setup_env.sh"

python - <<'PY'
import torch, flash_attn, transformers, datasets, peft, tensordict, verl
print("ENV OK | torch", torch.__version__, "| flash_attn", flash_attn.__version__,
      "| transformers", transformers.__version__)
PY
