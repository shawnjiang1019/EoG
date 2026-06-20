#!/bin/bash
# =============================================================================
# LOGIN node only (needs internet). RUN INSIDE tmux so a disconnect can't kill it:
#     tmux new -s dl
#     bash narval/prep_download.sh
#     # detach: Ctrl-b then d   |   reattach: tmux attach -t dl
#
# Downloads the base model into $SCRATCH using a tiny dedicated venv (just
# huggingface_hub) so it doesn't depend on the heavy training env. Resumable.
# This REPLACES the venv-build + model-download parts of the old prep_sft.sh;
# the venv is now built per-job on NVMe by setup_env.sh, and the SFT parquet is
# built inside the job.
# =============================================================================
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$SCRATCH/models/Qwen2.5-7B-Instruct}

module load StdEnv/2023 gcc python/3.11

DLVENV=$SCRATCH/EoG/dlvenv
if [ ! -d "$DLVENV" ]; then
  virtualenv --no-download "$DLVENV"
fi
source "$DLVENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index huggingface_hub hf_xet
export HF_XET_HIGH_PERFORMANCE=1     # fast Xet transfer (replaces deprecated HF_HUB_ENABLE_HF_TRANSFER)

mkdir -p "$MODEL_DIR"
echo "Downloading Qwen2.5-7B-Instruct -> $MODEL_DIR (resumable; re-run to continue)"
# NOTE: huggingface_hub >=1.x removed `huggingface-cli`; the CLI is now `hf`.
hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$MODEL_DIR"

echo "Done. Size check:"; du -sh "$MODEL_DIR"
