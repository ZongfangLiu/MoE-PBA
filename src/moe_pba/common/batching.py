from __future__ import annotations

from typing import Any

import torch


def batch_to_tensors(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(batch, torch.Tensor):
        input_ids = batch
        attention_mask = None
        labels_mask = None
    elif isinstance(batch, dict):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels_mask = batch.get("labels_mask")
    else:
        input_ids = batch.input_ids
        attention_mask = getattr(batch, "attention_mask", None)
        labels_mask = getattr(batch, "labels_mask", None)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if labels_mask is not None and labels_mask.dim() == 1:
        labels_mask = labels_mask.unsqueeze(0)
    return input_ids, attention_mask, labels_mask
