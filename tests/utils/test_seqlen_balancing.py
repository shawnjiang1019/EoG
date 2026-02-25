

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl import DataProto
from verl.utils.model import create_random_mask
from verl.utils.seqlen_balancing import (
    ceildiv,
    get_reverse_idx,
    prepare_dynamic_batch,
    rearrange_micro_batches,
    restore_dynamic_batch,
)

def test_seqlen_balancing():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5
    )
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, micro_bsz_idx_lst = rearrange_micro_batches(dataproto.batch, max_token_len=300)
    batch = torch.cat(micro_batches)
    micro_bsz_idx = []
    for idx in micro_bsz_idx_lst:
        micro_bsz_idx.extend(idx)
    reverse_idx_map = get_reverse_idx(micro_bsz_idx)
    reverse_idx_map = torch.tensor(reverse_idx_map)
    new_batch = batch[reverse_idx_map]
    torch.testing.assert_close(new_batch, dataproto.batch)

def test_dynamic_batch():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5
    )
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, micro_bsz_idx_lst = prepare_dynamic_batch(dataproto, max_token_len=300)
    input_ids = torch.cat([micro_batch.batch["input_ids"] for micro_batch in micro_batches], dim=0)
    input_ids = restore_dynamic_batch(input_ids, micro_bsz_idx_lst)
    torch.testing.assert_close(input_ids, dataproto.batch["input_ids"])

def _worker(rank, world_size, init_method, max_token_len, use_same_dp, min_mb):

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, 10, (20 + rank * 5, 100), device=f"cuda:{rank}")
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_left_padding=0.1,
        max_ratio_of_valid_token=0.9,
        min_ratio_of_valid_token=0.5,
    )
    dp = {"input_ids": input_ids, "attention_mask": attention_mask}
    proto = DataProto.from_single_dict(dp)
    batch = proto.batch

    micros, idx_lst = rearrange_micro_batches(
        batch,
        max_token_len=max_token_len,
        dp_group=dist.group.WORLD,
        same_micro_num_in_dp=use_same_dp,
        min_num_micro_batch=min_mb,
    )

    seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)
    total_seqlen = seq_len_effective.sum().item()
    local = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))

    if min_mb is not None:
        expected = max(local, min_mb)
        assert len(micros) == expected
    if use_same_dp:

        counts = [torch.zeros(1, device=f"cuda:{rank}") for _ in range(world_size)]
        counts[rank].fill_(local)
        dist.all_gather(counts, counts[rank])
        expected = max(int(c.item()) for c in counts)
        assert len(micros) == expected
    else:

        assert len(micros) == local

    flat = torch.cat(micros, dim=0)
    idx = []
    for sub in idx_lst:
        idx.extend(sub)
    inv = get_reverse_idx(idx)
    inv = torch.tensor(inv, device=flat.device)
    reconstructed = flat[inv]
    torch.testing.assert_close(reconstructed, batch)

    dist.destroy_process_group()

def test_dataproto_split_uneven():

    input_ids = torch.randint(low=0, high=10, size=(10, 5))
    attention_mask = torch.ones(10, 5)
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)

    splits = dataproto.split(3)
    assert len(splits) == 4
    assert len(splits[0]) == 3
    assert len(splits[1]) == 3
    assert len(splits[2]) == 3
    assert len(splits[3]) == 1

    reconstructed = DataProto.concat(splits)
    torch.testing.assert_close(reconstructed.batch["input_ids"], dataproto.batch["input_ids"])
    torch.testing.assert_close(reconstructed.batch["attention_mask"], dataproto.batch["attention_mask"])

    splits = dataproto.split(10)
    assert len(splits) == 1
    assert len(splits[0]) == 10

    splits = dataproto.split(15)
    assert len(splits) == 1
    assert len(splits[0]) == 10

    import numpy as np

    data_with_non_tensor = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": np.array([f"label_{i}" for i in range(10)], dtype=object),
    }
    dataproto_with_non_tensor = DataProto.from_single_dict(data_with_non_tensor)

    splits = dataproto_with_non_tensor.split(3)
    assert len(splits) == 4
    assert len(splits[0]) == 3
    assert len(splits[1]) == 3
    assert len(splits[2]) == 3
    assert len(splits[3]) == 1

    reconstructed = DataProto.concat(splits)
    np.testing.assert_array_equal(
        reconstructed.non_tensor_batch["labels"], dataproto_with_non_tensor.non_tensor_batch["labels"]
    )

def test_seqlen_balancing_distributed_params(tmp_path):
    world_size = 2
    init_file = tmp_path / "dist_init"
    init_file.write_text("")
    init_method = f"file://{init_file}"

    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, False, 4),
        nprocs=world_size,
        join=True,
    )

    mp.spawn(
        _worker,
        args=(world_size, init_method, 300, True, None),
        nprocs=world_size,
        join=True,
    )
