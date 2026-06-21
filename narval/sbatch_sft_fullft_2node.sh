#!/bin/bash
# =============================================================================
# EoG Stage-1 (SFT) -- FULL fine-tune across 2 nodes x 4 A100-40GB = 8 GPUs.
# FSDP shards the optimizer/param/grad state over 8 GPUs (~14 GB/GPU instead of
# ~28 GB on 4), leaving room for full-length logits -> max_length=8192, NO
# truncation, no memory tricks.  Submit:  export REPO=$PWD ; sbatch this
#
# Trade-off vs the single-node script: cross-node FSDP comms (network, not
# NVLink) are slower per step, and 2 full nodes wait longer in the queue.
# =============================================================================
#SBATCH --account=def-enaskt
#SBATCH --job-name=eog-sftft-2node
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1            # one torchrun launcher per node
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G                     # per node
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

export REPO=${REPO:-$SCRATCH/EoG}
export MODEL_DIR=${MODEL_DIR:-$SCRATCH/models/Qwen2.5-7B-Instruct}
export DATA=${DATA:-$SCRATCH/eog/data/2wiki_sft_train.parquet}
export SAVE=${SAVE:-$SCRATCH/eog/ckpt/2wiki_sft_2node}

# The parquet must already exist: building it here would race across the 2 nodes.
# (It's created by the single-node sbatch, or run kg_qa_sft_process.py once.)
if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA not found. Build it once first (single-node sbatch or kg_qa_sft_process.py)."
  exit 1
fi

# Rendezvous: first node hosts the c10d store; same host/port on both nodes.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "rendezvous = $MASTER_ADDR:$MASTER_PORT over $SLURM_NNODES nodes"

# One launcher per node (srun), each builds its own NVMe venv + stages the model,
# then joins the c10d rendezvous. Single quotes: vars expand on each node.
srun --ntasks-per-node=1 bash -c '
  set -uo pipefail
  source "$REPO/narval/setup_env.sh"               # venv on THIS node (NVMe)
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$SCRATCH/hf
  export TOKENIZERS_PARALLELISM=false TMPDIR=$SLURM_TMPDIR
  export PYTORCH_ALLOC_CONF=expandable_segments:True

  LM=$SLURM_TMPDIR/base_model
  echo "[stage node $SLURM_NODEID] copying model -> $LM"
  cp -r "$MODEL_DIR" "$LM"

  cd "$REPO"
  torchrun \
    --nnodes=$SLURM_NNODES --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.max_length=8192 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    optim.lr=2e-5 \
    model.partial_pretrain="$LM" \
    model.lora_rank=0 \
    model.strategy=fsdp2 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    trainer.default_local_dir="$SAVE" \
    trainer.project_name="eog_sft" \
    trainer.experiment_name="2wiki_qwen7b_fullft_2node" \
    trainer.logger=console \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=1 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.n_gpus_per_node=4 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false
'
echo "2-node full-FT SFT done -> $SAVE"

# If NCCL hangs at startup (cross-node), the usual fix is to pin the interface:
#   export NCCL_SOCKET_IFNAME=<iface>   (and/or NCCL_IB_HCA) -- check `ip a` on a node.
