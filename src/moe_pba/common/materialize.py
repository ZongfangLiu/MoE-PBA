from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from moe_pba.common.modeling import (
    discover_moe_layers,
    supported_model_type,
    validate_uniform_checkpoint_support,
)
from moe_pba.model_util import get_moe, resolve_model_attrs

logger = logging.getLogger(__name__)


def _normalize_prune_map(pruned_experts: dict[int | str, Iterable[int]]) -> dict[int, list[int]]:
    normalized: dict[int, list[int]] = {}
    for raw_layer, raw_experts in pruned_experts.items():
        layer_idx = int(raw_layer)
        experts = [int(expert_idx) for expert_idx in raw_experts]
        if len(experts) != len(set(experts)):
            raise ValueError(f"Layer {layer_idx}: pruned expert indices must be unique.")
        normalized[layer_idx] = sorted(experts)
    return normalized


def _retained_indices(num_experts: int, pruned: Iterable[int], *, layer_idx: int) -> list[int]:
    pruned_set = {int(expert_idx) for expert_idx in pruned}
    invalid = sorted(expert_idx for expert_idx in pruned_set if expert_idx < 0 or expert_idx >= num_experts)
    if invalid:
        raise ValueError(
            f"Layer {layer_idx}: expert indices {invalid} are outside [0, {num_experts - 1}]."
        )
    retained = [expert_idx for expert_idx in range(num_experts) if expert_idx not in pruned_set]
    if not retained:
        raise ValueError(f"Layer {layer_idx}: cannot prune every expert.")
    return retained


def _replace_indexed_parameter(module: nn.Module, name: str, retained: list[int]) -> bool:
    value = getattr(module, name, None)
    if value is None:
        return False
    tensor = value.data if isinstance(value, nn.Parameter) else value
    if not torch.is_tensor(tensor) or tensor.ndim == 0:
        return False
    index = torch.tensor(retained, device=tensor.device, dtype=torch.long)
    sliced = tensor.index_select(0, index).clone()
    if isinstance(value, nn.Parameter):
        sliced = nn.Parameter(sliced, requires_grad=value.requires_grad)
    setattr(module, name, sliced)
    return True


def _slice_router(router: nn.Module, retained: list[int], kept: int) -> None:
    if not _replace_indexed_parameter(router, "weight", retained):
        raise ValueError(f"Unsupported router layout for {router.__class__.__name__}: missing indexed weight.")
    _replace_indexed_parameter(router, "bias", retained)
    if hasattr(router, "num_experts"):
        router.num_experts = kept
    if hasattr(router, "top_k"):
        router.top_k = min(int(router.top_k), kept)
    if hasattr(router, "out_features"):
        router.out_features = kept


def _slice_experts(experts: Any, retained: list[int], kept: int) -> Any:
    if isinstance(experts, nn.ModuleList):
        return nn.ModuleList([experts[expert_idx] for expert_idx in retained])

    sliced = False
    for name in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
        sliced = _replace_indexed_parameter(experts, name, retained) or sliced
    if not sliced:
        raise ValueError(f"Unsupported expert layout for {experts.__class__.__name__}.")
    if hasattr(experts, "num_experts"):
        experts.num_experts = kept
    return experts


def _slice_ernie_correction_bias(moe: nn.Module, router: nn.Module, retained: list[int]) -> None:
    statics = getattr(moe, "moe_statics", None)
    if statics is None:
        statics = getattr(router, "moe_statics", None)
    correction = getattr(statics, "e_score_correction_bias", None)
    if correction is None:
        return
    tensor = correction.data if isinstance(correction, nn.Parameter) else correction
    if not torch.is_tensor(tensor):
        raise ValueError("ERNIE router correction bias is not a tensor.")
    index = torch.tensor(retained, device=tensor.device, dtype=torch.long)
    sliced = tensor.index_select(-1, index).clone()
    if isinstance(correction, nn.Parameter):
        sliced = nn.Parameter(sliced, requires_grad=correction.requires_grad)
    statics.e_score_correction_bias = sliced


def _validate_uniform_plan(model, prune_map: dict[int, list[int]]) -> tuple[list[Any], int]:
    layers = discover_moe_layers(model)
    layer_indices = {int(layer.layer_idx) for layer in layers}
    unexpected = sorted(set(prune_map) - layer_indices)
    if unexpected:
        raise ValueError(f"Pruning plan contains non-MoE layer indices: {unexpected}.")

    prune_counts = {len(prune_map.get(int(layer.layer_idx), [])) for layer in layers}
    if len(prune_counts) != 1:
        details = {int(layer.layer_idx): len(prune_map.get(int(layer.layer_idx), [])) for layer in layers}
        raise ValueError(f"Uniform materialization requires the same prune count in every MoE layer: {details}.")
    prune_count = prune_counts.pop()

    expert_counts = {int(layer.num_experts) for layer in layers}
    if len(expert_counts) != 1:
        raise ValueError(
            "Uniform materialization requires the supported model to use one expert count across MoE layers."
        )
    kept = expert_counts.pop() - prune_count
    if kept <= 0:
        raise ValueError("Uniform materialization cannot prune every expert.")
    for layer in layers:
        if kept < int(layer.top_k):
            raise ValueError(
                f"Layer {layer.layer_idx}: kept experts {kept} is below top_k={layer.top_k}."
            )
    return layers, kept


def _update_global_config(model, model_type: str, kept: int) -> None:
    config = model.config
    if model_type in {"olmoe", "qwen3_moe"}:
        config.num_experts = kept
        config.num_experts_per_tok = min(int(config.num_experts_per_tok), kept)
    elif model_type == "deepseek_v2":
        config.n_routed_experts = kept
        config.num_experts_per_tok = min(int(config.num_experts_per_tok), kept)
    elif model_type == "ernie4_5_moe":
        config.moe_num_experts = kept
        config.moe_k = min(int(config.moe_k), kept)
    else:  # guarded by supported_model_type
        raise ValueError(f"Unsupported model_type={model_type!r}.")

    # Strip custom-code hints so the saved model uses its standard Transformers class.
    for config_field in ("num_experts_per_layer", "auto_map"):
        if hasattr(config, config_field):
            try:
                delattr(config, config_field)
            except AttributeError:
                pass

    # Store dtype metadata in the format accepted by the active Transformers version.
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None and first_parameter.is_floating_point():
        dtype = first_parameter.dtype
        dtype_name = str(dtype).removeprefix("torch.")
        if hasattr(config, "dtype"):
            config.dtype = dtype_name
        else:
            config.torch_dtype = dtype
            config.dtype = dtype_name


def compact_uniform_model(model, pruned_experts: dict[int | str, Iterable[int]]) -> dict[str, list[int]]:
    """Physically remove a uniform expert budget from a loaded Transformers model in place."""

    model_type = supported_model_type(model)
    prune_map = _normalize_prune_map(pruned_experts)
    layers, kept = _validate_uniform_plan(model, prune_map)
    prune_count = int(layers[0].num_experts) - kept
    validate_uniform_checkpoint_support(model, layers, prune_count)

    for layer in layers:
        layer_idx = int(layer.layer_idx)
        retained = _retained_indices(layer.num_experts, prune_map.get(layer_idx, []), layer_idx=layer_idx)
        if len(retained) != kept:
            raise ValueError(f"Layer {layer_idx}: expected {kept} retained experts, found {len(retained)}.")

        moe = get_moe(model, layer_idx)
        attrs = resolve_model_attrs(model, moe)
        router = getattr(moe, attrs["router"])
        experts = getattr(moe, attrs["experts"])
        _slice_router(router, retained, kept)
        setattr(moe, attrs["experts"], _slice_experts(experts, retained, kept))

        if model_type == "ernie4_5_moe":
            _slice_ernie_correction_bias(moe, router, retained)
        if hasattr(moe, "num_experts"):
            moe.num_experts = kept
        if hasattr(moe, "top_k"):
            moe.top_k = min(int(moe.top_k), kept)
        if hasattr(moe, "num_experts_per_tok"):
            moe.num_experts_per_tok = min(int(moe.num_experts_per_tok), kept)
        if hasattr(moe, "experts_per_rank"):
            moe.experts_per_rank = kept
        if hasattr(model, "num_experts"):
            model.num_experts = kept
        if hasattr(model, "num_experts_per_tok"):
            model.num_experts_per_tok = min(int(model.num_experts_per_tok), kept)
        if hasattr(moe, "config"):
            moe.config = model.config

    _update_global_config(model, model_type, kept)
    return {str(layer_idx): experts for layer_idx, experts in sorted(prune_map.items())}


def materialize_uniform_model(
    model,
    tokenizer,
    pruned_experts: dict[int | str, Iterable[int]],
    target_dir: pathlib.Path,
) -> None:
    """Compact and save a normal, reloadable Transformers checkpoint."""

    target_dir.mkdir(parents=True, exist_ok=True)
    normalized = compact_uniform_model(model, pruned_experts)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    with open(target_dir / "pruned_experts.json", "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
        handle.write("\n")
    logger.info("Saved standard uniform checkpoint with %d MoE layers to %s.", len(normalized), target_dir)
