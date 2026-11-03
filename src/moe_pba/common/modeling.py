from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

from moe_pba.model_util import MODEL_ATTRS, get_moe, resolve_model_attrs
from moe_pba.common.types import LayerSpec

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = {"olmoe", "qwen3_moe", "ernie4_5_moe", "deepseek_v2"}


def supported_model_type(model) -> str:
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Supported: {sorted(SUPPORTED_MODEL_TYPES)}"
        )
    return model_type


def _validate_deepseek_groups(model, kept: int) -> None:
    if supported_model_type(model) != "deepseek_v2":
        return
    topk_method = str(getattr(model.config, "topk_method", "greedy") or "greedy")
    groups = int(getattr(model.config, "n_group", 1) or 1)
    if topk_method != "greedy" and groups > 1:
        raise ValueError(
            "Uniform checkpoint compaction does not preserve DeepSeek grouped routing semantics. "
            "Only topk_method='greedy' or n_group=1 is supported."
        )
    if groups > 1 and kept % groups != 0:
        raise ValueError(f"Retained experts ({kept}) must be divisible by DeepSeek n_group={groups}.")


def validate_uniform_checkpoint_support(model, layers: list[LayerSpec], prune_count: int) -> int:
    """Reject plans that search successfully but cannot become a native uniform checkpoint."""

    expert_counts = {int(layer.num_experts) for layer in layers}
    if len(expert_counts) != 1:
        raise ValueError(
            "Uniform checkpoint materialization requires one expert count across all MoE layers."
        )
    kept = expert_counts.pop() - int(prune_count)
    if kept <= 0:
        raise ValueError("Uniform checkpoint materialization cannot prune every expert.")
    for layer in layers:
        if kept < int(layer.top_k):
            raise ValueError(
                f"Layer {layer.layer_idx}: kept experts {kept} is below top_k={layer.top_k}."
            )
    _validate_deepseek_groups(model, kept)
    return kept


def validate_method_support(
    model,
    *,
    layers: list[LayerSpec] | None = None,
    prune_count: int | None = None,
) -> str:
    model_type = supported_model_type(model)
    if layers is not None and prune_count is not None:
        validate_uniform_checkpoint_support(model, layers, prune_count)
    return model_type


def model_short_name(model_name: str | pathlib.Path) -> str:
    return str(model_name).rstrip("/").split("/")[-1]


def discover_moe_layers(model) -> list[LayerSpec]:
    if model.__class__.__name__ not in MODEL_ATTRS:
        raise ValueError(
            f"Model class {model.__class__.__name__} is not registered in MODEL_ATTRS."
        )

    specs: list[LayerSpec] = []
    for layer_idx, _layer in enumerate(model.model.layers):
        try:
            moe = get_moe(model, layer_idx)
        except Exception:
            continue
        model_attrs = resolve_model_attrs(model, moe)
        experts = getattr(moe, model_attrs["experts"], None)
        router = getattr(moe, model_attrs["router"], None)
        if experts is None or router is None:
            continue

        if model_attrs["fused"]:
            if not hasattr(experts, model_attrs["gate_proj"]) or not hasattr(experts, model_attrs["down_proj"]):
                continue
            num_experts = int(getattr(experts, "num_experts", getattr(experts, model_attrs["gate_proj"]).shape[0]))
        else:
            try:
                num_experts = len(experts)
            except TypeError:
                continue
        top_k = int(
            getattr(moe, "top_k", None)
            or getattr(router, "top_k", None)
            or getattr(model.config, model_attrs["num_experts_per_tok"], 1)
            or 1
        )
        specs.append(
            LayerSpec(
                layer_idx=layer_idx,
                num_experts=int(num_experts),
                top_k=max(1, min(int(top_k), int(num_experts))),
            )
        )
    if not specs:
        raise RuntimeError("No supported MoE layers were discovered.")
    return specs


def max_prunable_per_layer(layers: list[LayerSpec]) -> dict[int, int]:
    return {layer.layer_idx: max(layer.num_experts - layer.top_k, 0) for layer in layers}


def resolve_prune_count(layers: list[LayerSpec], moe_pba_args) -> int:
    cached = getattr(moe_pba_args, "resolved_prune_count", None)
    if cached is not None:
        return int(cached)

    ratio = getattr(moe_pba_args, "sparsity_ratio", None)
    if ratio is None:
        raise ValueError("moe_pba requires sparsity_ratio to be set.")

    if not layers:
        raise RuntimeError("Cannot resolve sparsity_ratio without discovered MoE layers.")

    ratio = float(ratio)
    if not math.isfinite(ratio):
        raise ValueError("sparsity_ratio must be finite.")
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("sparsity_ratio must be within [0, 1].")

    resolved_counts: list[int] = []
    per_layer_limits = max_prunable_per_layer(layers)
    for layer in layers:
        prune_count = int(math.floor(ratio * float(layer.num_experts)))
        max_prunable = int(per_layer_limits[layer.layer_idx])
        if prune_count > max_prunable:
            raise ValueError(
                "sparsity_ratio resolves to an invalid prune count for layer "
                f"{layer.layer_idx}: ratio={ratio:g} -> prune_count={prune_count}, "
                f"but the layer can prune at most {max_prunable} experts while preserving top_k={layer.top_k}."
            )
        resolved_counts.append(prune_count)

    unique_counts = sorted(set(int(count) for count in resolved_counts))
    if len(unique_counts) != 1:
        raise ValueError(
            "sparsity_ratio resolves to different per-layer prune counts across MoE layers "
            f"({unique_counts}). Current moe_pba methods require a uniform per-layer prune count."
        )

    resolved = int(unique_counts[0])
    try:
        moe_pba_args.resolved_prune_count = resolved
    except Exception:
        pass
    return resolved


def build_sparsity_tag(moe_pba_args) -> str:
    ratio = getattr(moe_pba_args, "sparsity_ratio", None)
    if ratio is None:
        resolved = getattr(moe_pba_args, "resolved_prune_count", None)
        if resolved is None:
            return "ratio-missing"
        return f"ratio-unknown-prune{int(resolved)}"
    return f"ratio{float(ratio):.3f}"


def build_output_dir(model_name: str, method: str, run_name: str, explicit_output_dir: str | None = None) -> pathlib.Path:
    if explicit_output_dir:
        return pathlib.Path(explicit_output_dir)
    return pathlib.Path("artifacts") / "moe_pba" / model_short_name(model_name) / method / run_name


def completed_output_exists(output_dir: pathlib.Path) -> bool:
    required_files = {
        "config.json",
        "pruned_experts.json",
        "pruning_plan.json",
        "search_metadata.json",
        "moe_pba_args.yaml",
        "run_request.json",
    }
    model_weight_files = {"model.safetensors", "pytorch_model.bin"}
    sharded_weight_index_files = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json"}

    if not output_dir.exists():
        return False
    if not all((output_dir / name).exists() for name in required_files):
        return False
    has_single_weight_file = any((output_dir / name).exists() for name in model_weight_files)
    has_sharded_weight_files = any((output_dir / name).exists() for name in sharded_weight_index_files) and (
        any(output_dir.glob("model-*-of-*.safetensors"))
        or any(output_dir.glob("pytorch_model-*-of-*.bin"))
    )
    if not (has_single_weight_file or has_sharded_weight_files):
        return False
    if not any((output_dir / name).exists() for name in tokenizer_files):
        return False
    return True


def build_plan_summary(
    layers: list[LayerSpec],
    pruned_experts: dict[str, list[int]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for layer in layers:
        pruned = len(pruned_experts.get(str(layer.layer_idx), []))
        kept = int(layer.num_experts - pruned)
        summary.append(
            {
                "layer": int(layer.layer_idx),
                "total_experts": int(layer.num_experts),
                "pruned_experts": int(pruned),
                "kept_experts": int(kept),
                "prune_ratio": (float(pruned) / float(layer.num_experts)) if layer.num_experts > 0 else 0.0,
            }
        )
    return summary
