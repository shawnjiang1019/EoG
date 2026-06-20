#!/bin/bash
# =============================================================================
# Build the training venv on node-local NVMe ($SLURM_TMPDIR) -- fast, and avoids
# the Lustre small-file / inode pain on $SCRATCH.
#
# SOURCE this inside a Slurm allocation (sbatch body or `salloc`):
#     export REPO=$SCRATCH/EoG
#     source $REPO/narval/setup_env.sh
# It activates the venv and puts the repo on PYTHONPATH (verl has no setup.py).
# No `set -e` here on purpose, so a hiccup won't kill your interactive shell.
# =============================================================================
: "${SLURM_TMPDIR:?source this INSIDE a Slurm job/salloc -- SLURM_TMPDIR is unset}"
REPO=${REPO:-$SCRATCH/EoG}

module load StdEnv/2023 gcc arrow python/3.11 cuda/12.2

VENV=$SLURM_TMPDIR/venv
echo "[setup_env] building venv at $VENV"
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
# torch is pinned first in the requirements file -> single clean resolution
pip install --no-index -r "$REPO/narval/requirements_narval.txt"

# verl is imported from the repo (no pip package); make it importable everywhere
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python - <<'PY'
import torch
print("[setup_env] torch", torch.__version__)
PY
echo "[setup_env] venv ready; PYTHONPATH includes the repo"
