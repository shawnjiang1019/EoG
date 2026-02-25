set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_kg_qa_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="<path_to_train_data>" \
    data.val_files="<path_to_test_data>" \
    data.max_length=19999 \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    optim.lr=2e-5 \
    model.partial_pretrain="<path_to_model>" \
    model.lora_rank=0 \
    model.lora_alpha=16 \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name="<project_name>" \
    trainer.experiment_name="<experiment_name>" \
    trainer.logger=console \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
