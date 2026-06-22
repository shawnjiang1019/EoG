#!/bin/bash
# =============================================================================
# Held-out eval on the extracted-graph test set. Run once per checkpoint:
#   export REPO=$PWD
#   MODEL=$SCRATCH/eog/ckpt/2wiki_sft_v2/hf   sbatch -J eog-test-full narval/sbatch_eval_test.sh
#   MODEL=$SCRATCH/eog/ckpt/2wiki_sft_lora/hf sbatch -J eog-test-lora narval/sbatch_eval_test.sh
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-eval-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

set -uo pipefail
REPO=${REPO:-$SCRATCH/EoG}
MODEL=${MODEL:?set MODEL=<merged hf dir>}
DATA=${DATA:-$SCRATCH/eog/data/2wiki_test_graphs.jsonl}   # from extract_graphs.py
N=${N:-0}                                                 # 0 = all rows in the test set

source "$REPO/narval/setup_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf TMPDIR=$SLURM_TMPDIR

cd "$REPO/narval"   # so eval_test.py can import eval_sft (same dir)
python eval_test.py --model "$MODEL" --data "$DATA" --n "$N"
