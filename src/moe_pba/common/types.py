from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerSpec:
    layer_idx: int
    num_experts: int
    top_k: int


@dataclass
class CalibrationContext:
    batches: list[Any]
    fingerprint: str
    dataset_tag: str
    batch_count: int


@dataclass
class MethodResult:
    pruned_experts: dict[str, list[int]]
    plan_summary: list[dict[str, Any]]
    score_summary: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)


def clone_prune_map(pruned: dict[int | str, list[int]] | None) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if not pruned:
        return out
    for layer, experts in pruned.items():
        out[int(layer)] = sorted({int(idx) for idx in experts})
    return out


def merge_prune_maps(
    left: dict[int | str, list[int]] | None,
    right: dict[int | str, list[int]] | None,
) -> dict[int, list[int]]:
    out = clone_prune_map(left)
    for layer, experts in clone_prune_map(right).items():
        merged = set(out.get(layer, []))
        merged.update(int(idx) for idx in experts)
        out[layer] = sorted(merged)
    return out


def to_jsonable_prune_map(pruned: dict[int | str, list[int]]) -> dict[str, list[int]]:
    return {str(int(layer)): sorted(int(idx) for idx in experts) for layer, experts in clone_prune_map(pruned).items()}
