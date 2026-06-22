#!/bin/bash
# =============================================================================
# Build a real held-out 2Wiki test set by extracting per-question graphs from
# the dev passages (paper Appendix F). GPU job.
#
# PREREQUISITE (run once on a LOGIN node, needs internet):
#   source $SCRATCH/EoG/dlvenv/bin/activate   # or any env with pandas+pyarrow
#   python $REPO/data/prep_2wiki_dev.py --out $SCRATCH/eog/data/2wiki_dev_raw.jsonl --n 1000
#
# Then:  export REPO=$PWD ; sbatch narval/sbatch_extract_graphs.sh
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-extract-graphs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail
REPO=${REPO:-$SCRATCH/EoG}
RAW=${RAW:-$SCRATCH/eog/data/2wiki_dev_raw.jsonl}
OUT=${OUT:-$SCRATCH/eog/data/2wiki_test_graphs.jsonl}
MODEL=${MODEL:-$SCRATCH/models/Qwen2.5-7B-Instruct}   # paper uses Gemma-2-9b-it (gated)
N=${N:-1000}                                          # #questions to build
BATCH=${BATCH:-16}

source "$REPO/narval/setup_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf TMPDIR=$SLURM_TMPDIR

if [ ! -f "$RAW" ]; then
  echo "ERROR: $RAW not found. Run data/prep_2wiki_dev.py on a login node first."
  exit 1
fi

python "$REPO/data/extract_graphs.py" \
  --input "$RAW" --output "$OUT" --model "$MODEL" --n "$N" --batch_size "$BATCH"
echo "done -> $OUT"
