from __future__ import annotations

import argparse
from typing import Iterable

import torch
from torch.utils.data import DataLoader


def train_steps(
    model,
    dataloader: DataLoader,
    *,
    learning_rate: float,
    max_steps: int,
    device: str,
    warmup_ratio: float = 0.03,
    grad_clip: float = 1.0,
) -> None:
    model.to(device)
    model.train()
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    warmup_steps = max(1, int(max_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, max_steps - warmup_steps)
        return max(0.0, float(max_steps - step) / float(remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    step = 0
    while step < max_steps:
        for batch in dataloader:
            optim.zero_grad(set_to_none=True)
            allowed = {
                "input_ids",
                "attention_mask",
                "labels",
                "images",
                "pixel_values",
                "image_grid_thw",
                "retrieved_input_ids",
                "retrieved_attention_mask",
            }
            tensor_batch = {k: v.to(device) for k, v in batch.items() if k in allowed and torch.is_tensor(v)}
            out = model(**tensor_batch)
            loss = out.loss if hasattr(out, "loss") else out["loss"]
            if loss is None:
                raise RuntimeError("Model did not return a training loss.")
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), grad_clip)
            optim.step()
            scheduler.step()
            print(f"step={step} loss={float(loss.detach().cpu()):.4f}")
            step += 1
            if step >= max_steps:
                return
