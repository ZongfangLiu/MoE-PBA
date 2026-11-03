from typing import Any

import torch
import torch.nn as nn

MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "OlmoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
}


def _tensor_ndim(value: Any) -> int | None:
    if isinstance(value, nn.Parameter):
        value = value.data
    if torch.is_tensor(value):
        return value.ndim
    return None


def _looks_like_fused_experts(experts: Any) -> bool:
    if experts is None:
        return False
    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        return False
    gate_dim = _tensor_ndim(gate_up_proj)
    down_dim = _tensor_ndim(down_proj)
    if gate_dim is None or down_dim is None:
        return False
    return gate_dim >= 3 and down_dim >= 3


def resolve_model_attrs(model: nn.Module, moe: nn.Module | None = None) -> dict[str, Any]:
    base = MODEL_ATTRS[model.__class__.__name__]
    if base.get("fused", False):
        return base
    if moe is None:
        try:
            moe = get_moe(model, 0)
        except Exception:
            moe = None
    experts = None
    if moe is not None:
        try:
            experts = getattr(moe, base["experts"])
        except Exception:
            experts = None
    if not _looks_like_fused_experts(experts):
        return base
    updated = dict(base)
    updated["fused"] = True
    if getattr(experts, "gate_up_proj", None) is not None:
        updated["gate_proj"] = "gate_up_proj"
        updated["up_proj"] = "gate_up_proj"
    if getattr(experts, "down_proj", None) is not None:
        updated["down_proj"] = "down_proj"
    if hasattr(model, "config") and hasattr(model.config, "num_local_experts"):
        updated["num_experts"] = "num_local_experts"
    return updated


def _looks_like_moe_module(module: Any, base_attrs: dict[str, Any]) -> bool:
    if module is None:
        return False
    experts_attr = str(base_attrs.get("experts", "experts"))
    router_attr = str(base_attrs.get("router", "router"))
    if hasattr(module, experts_attr) and hasattr(module, router_attr):
        return True
    if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
        return True
    return False


def get_moe(model, layer):
    base_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    if base_attrs is None:
        raise ValueError(
            f"Model class {model.__class__.__name__} is not registered in MODEL_ATTRS."
        )

    layer_module = model.model.layers[layer]
    moe_attr_name = base_attrs["moe_block"]
    candidate_attrs = []
    for attr_name in (
        moe_attr_name,
        "mlp",
        "block_sparse_moe",
        "feed_forward",
    ):
        if attr_name not in candidate_attrs:
            candidate_attrs.append(attr_name)

    fallback_names: list[str] = []
    for attr_name in candidate_attrs:
        moe = getattr(layer_module, attr_name, None)
        if _looks_like_moe_module(moe, base_attrs):
            return moe
        if moe is not None:
            fallback_names.append(attr_name)

    if hasattr(layer_module, moe_attr_name):
        return getattr(layer_module, moe_attr_name)

    raise AttributeError(
        f"{layer_module.__class__.__name__} has no compatible MoE block for "
        f"{model.__class__.__name__} layer {layer}. Checked {candidate_attrs}; "
        f"present-but-incompatible={fallback_names}."
    )
