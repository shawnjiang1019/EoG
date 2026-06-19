#!/bin/bash
# =============================================================================
# Run this on a Narval LOGIN node (it needs internet). One-time prep for SFT.
#   - builds a Python venv in $SCRATCH (compute nodes have NO internet)
#   - downloads the base model
#   - converts the shipped 2Wiki SFT jsonl -> training parquet
# =============================================================================
set -euo pipefail

# ---- EDIT THESE ----
REPO=$SCRATCH/EoG                                  # where you cloned this repo
MODEL_DIR=$SCRATCH/models/Qwen2.5-7B-Instruct
DATA_OUT=$SCRATCH/EoG/data/2wiki_sft_train.parquet
MAX_TOKENS=12000                                  # keep <= data.max_length in the sbatch
# --------------------

module load StdEnv/2023 gcc arrow python/3.11 cuda/12.2

# 1) venv in $SCRATCH (build here on the login node; reuse it in the job)
if [ ! -d "$SCRATCH/EoG/venv" ]; then
  virtualenv --no-download "$SCRATCH/EoG/venv"
fi
source "$SCRATCH/EoG/venv/bin/activate"
pip install --no-index --upgrade pip

# 2) Deps. SFT does NOT need vLLM. Prefer the Alliance wheelhouse (--no-index).
#    First check what's available:  avail_wheels torch transformers flash_attn datasets peft accelerate
pip install --no-index torch transformers datasets accelerate peft tensordict
# flash-attn powers use_remove_padding; if the wheel is missing, skip it and set
# use_remove_padding=false in the sbatch instead.
pip install --no-index flash_attn || echo "WARN: flash_attn not in wheelhouse -> set use_remove_padding=false"
# Install the verl package from this repo (no-deps to avoid fighting version pins):
pip install --no-deps -e "$REPO"

# 3) Download base model (login node has internet)
mkdir -p "$MODEL_DIR"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir "$MODEL_DIR"

# 4) Build the SFT parquet from the shipped 2Wiki SFT jsonl
mkdir -p "$(dirname "$DATA_OUT")"
python "$REPO/data/kg_qa_sft_process.py" \
  --input  "$REPO/sft_data/2wikimultihop_train_sft.jsonl" \
  --output "$DATA_OUT" \
  --model_path "$MODEL_DIR" \
  --max_tokens "$MAX_TOKENS"

echo "Prep complete."
echo "  venv:   $SCRATCH/EoG/venv"
echo "  model:  $MODEL_DIR"
echo "  data:   $DATA_OUT"
