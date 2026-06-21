"""Generic training loop shared by every reasoner variant.

It only ever calls `reasoner.compute_loss(batch)`, so it is agnostic to whether
inputs are text token-ids or spliced `inputs_embeds`. Single-device baseline;
wrap with accelerate/FSDP later for multi-GPU without changing variants.
"""
from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader

from ..base.reasoner import BaseReasoner
from ..base.schema import ReasonerExample
from .data import ReasonerDataset


def train(reasoner: BaseReasoner, examples: list[ReasonerExample]) -> None:
    cfg = reasoner.config
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reasoner.build_model()
    reasoner.to(device)
    reasoner.train()

    dataset = ReasonerDataset(examples, reasoner)
    collator = reasoner.build_collator()
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    accum = max(1, cfg.train_batch_size // cfg.micro_batch_size)
    steps_per_epoch = max(1, len(loader) // accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)

    optim = torch.optim.AdamW(
        reasoner.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    global_step = 0
    for epoch in range(cfg.epochs):
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
                loss = reasoner.compute_loss(batch) / accum
            loss.backward()

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(reasoner.trainable_parameters(), cfg.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(
                        f"epoch {epoch} step {global_step}/{total_steps} "
                        f"loss {loss.item() * accum:.4f} lr {sched.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

        if cfg.save_every_epoch:
            ckpt = os.path.join(cfg.output_dir, f"epoch_{epoch}")
            reasoner.save(ckpt)
            print(f"saved {ckpt}", flush=True)

    reasoner.save(os.path.join(cfg.output_dir, "final"))
    print(f"done -> {os.path.join(cfg.output_dir, 'final')}", flush=True)
