from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

from moe_pba.model_util import MODEL_ATTRS, get_moe, resolve_model_attrs

logger = logging.getLogger(__name__)


class _FnHandle:
    def __init__(self, fn):
        self._fn = fn

    def remove(self):
        self._fn()


def _cleanup_handles(handles):
    for handle in reversed(handles):
        try:
            handle.remove()
        except Exception:
            pass


def _install_temp_router_masks(model: nn.Module, pruned_experts_info: dict[int, list[int]]) -> list[Any]:
    """
    Temporarily prevent pruned experts from being selected by supported MoE routers.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    top_k_default = int(getattr(model.config, model_attrs["num_experts_per_tok"]))
    num_experts_default = int(getattr(model.config, model_attrs["num_experts"]))
    handles: list[Any] = []
    warned_layers: set[tuple[int, str]] = set()

    def _as_valid_indices(pruned: list[int] | None, num_experts: int) -> list[int]:
        if not pruned:
            return []
        return sorted({int(i) for i in pruned if 0 <= int(i) < int(num_experts)})

    def _warn_masked_selected(layer: int, router_name: str, selected: torch.Tensor, valid: list[int]) -> None:
        if not valid:
            return
        key = (int(layer), str(router_name))
        if key in warned_layers:
            return
        if (not torch.is_tensor(selected)) or selected.numel() == 0:
            return
        if selected.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return
        idx = torch.tensor(valid, device=selected.device, dtype=selected.dtype)
        try:
            bad_mask = torch.isin(selected, idx)
        except Exception:
            bad_mask = (selected.unsqueeze(-1) == idx).any(dim=-1)
        if not bool(bad_mask.any()):
            return
        bad_vals = torch.unique(selected[bad_mask]).detach().cpu().tolist()
        logger.warning(
            "Masked experts were still activated during search (layer=%d, router=%s, masked=%s, count=%d).",
            layer,
            router_name,
            bad_vals[:8],
            len(bad_vals),
        )
        warned_layers.add(key)

    def _collect_tensors(value: Any, bucket: list[torch.Tensor]) -> None:
        if torch.is_tensor(value):
            bucket.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _collect_tensors(item, bucket)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect_tensors(item, bucket)

    def _inspect_router_output(output: Any, warn_fn, top_k: int, num_experts: int) -> None:
        if warn_fn is None:
            return
        tensors: list[torch.Tensor] = []
        _collect_tensors(output, tensors)
        int_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        for tensor in tensors:
            if (not torch.is_tensor(tensor)) or tensor.numel() == 0 or tensor.dtype not in int_types:
                continue
            try:
                if int(tensor.min().item()) < 0 or int(tensor.max().item()) >= int(num_experts):
                    continue
            except Exception:
                continue
            if tensor.ndim > 0 and int(tensor.shape[-1]) <= max(int(top_k), 1):
                warn_fn(tensor)
                return
        for tensor in tensors:
            if (not torch.is_tensor(tensor)) or tensor.numel() == 0:
                continue
            if (not torch.is_floating_point(tensor)) or tensor.ndim == 0:
                continue
            if int(tensor.shape[-1]) != int(num_experts):
                continue
            k = min(int(top_k), int(num_experts))
            if k <= 0:
                return
            try:
                warn_fn(torch.topk(tensor, k=k, dim=-1).indices)
            except Exception:
                return
            return

    def _mask_logits_inplace(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return logits
        if (not torch.is_floating_point(logits)) or logits.shape[-1] <= int(indices.max().item()):
            return logits
        masked = logits.clone()
        masked.index_fill_(-1, indices, torch.finfo(masked.dtype).min)
        return masked

    def _resolve_pruned_tensor(experts_to_prune: Any, default_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        experts = default_indices if experts_to_prune is None else experts_to_prune
        if isinstance(experts, (list, tuple, set)):
            experts = torch.tensor(list(experts), dtype=torch.long)
        elif not torch.is_tensor(experts):
            experts = torch.tensor([int(experts)], dtype=torch.long)
        return experts.to(device=device, dtype=torch.long).reshape(-1)

    def _try_install_ernie_bias_mask(moe: nn.Module, router: nn.Module, valid: list[int]) -> bool:
        if not valid:
            return False
        statics = getattr(moe, "moe_statics", None) or getattr(router, "moe_statics", None)
        bias = getattr(statics, "e_score_correction_bias", None) if statics is not None else None
        if not isinstance(bias, (nn.Parameter, torch.Tensor)):
            return False
        bias_tensor = bias.data if isinstance(bias, nn.Parameter) else bias
        original = bias_tensor.detach().clone()
        idx = torch.tensor(valid, device=bias_tensor.device, dtype=torch.long)
        bias_tensor.index_fill_(-1, idx, torch.finfo(bias_tensor.dtype).min)

        def _restore():
            try:
                bias_tensor.copy_(original)
            except Exception:
                pass

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_linear_gate_patch(router: nn.Module, valid: list[int]) -> bool:
        if not valid or not isinstance(router, nn.Linear):
            return False
        original_forward = router.forward
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_forward(hidden_states, *args, experts_to_prune=None, **kwargs):
            router_input = hidden_states
            weight = getattr(router, "weight", None)
            if (
                torch.is_tensor(router_input)
                and torch.is_tensor(weight)
                and torch.is_floating_point(router_input)
                and torch.is_floating_point(weight)
                and router_input.dtype != weight.dtype
            ):
                router_input = router_input.to(dtype=weight.dtype)
            logits = original_forward(router_input, *args, **kwargs)
            if torch.is_tensor(logits):
                experts = _resolve_pruned_tensor(experts_to_prune, idx_cpu, logits.device)
                logits = _mask_logits_inplace(logits, experts)
            return logits

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            router.forward = original_forward  # type: ignore[assignment]

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_deepseek_v2_gate_patch(router: nn.Module, valid: list[int], warn_fn) -> bool:
        """Mask DeepSeek-V2 router logits before internal routing."""

        if not valid or router.__class__.__name__ != "DeepseekV2MoEGate" or not hasattr(router, "weight"):
            return False
        original_forward = router.forward
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_forward(hidden_states, *args, experts_to_prune=None, **kwargs):
            if args or kwargs:
                return original_forward(hidden_states, *args, **kwargs)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
            logits = torch.nn.functional.linear(
                flat_hidden_states.to(dtype=torch.float32),
                router.weight.to(dtype=torch.float32),
                None,
            )
            experts = _resolve_pruned_tensor(experts_to_prune, idx_cpu, logits.device)
            logits = _mask_logits_inplace(logits, experts)
            scores = logits.softmax(dim=-1, dtype=torch.float32)

            topk_method = str(getattr(router, "topk_method", "greedy"))
            if topk_method == "greedy":
                topk_weight, topk_idx = torch.topk(scores, k=int(router.top_k), dim=-1, sorted=False)
            elif topk_method == "group_limited_greedy":
                num_group = int(getattr(router, "num_group"))
                num_experts = int(getattr(router, "num_experts"))
                topk_group = int(getattr(router, "topk_group"))
                group_scores = scores.view(batch_size * seq_len, num_group, -1).max(dim=-1).values
                group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(batch_size * seq_len, num_group, num_experts // num_group)
                    .reshape(batch_size * seq_len, -1)
                )
                topk_weight, topk_idx = torch.topk(
                    scores.masked_fill(~score_mask.bool(), 0.0),
                    k=int(router.top_k),
                    dim=-1,
                    sorted=False,
                )
            else:
                raise NotImplementedError(f"Unsupported DeepSeek-V2 topk_method: {topk_method}")

            topk_weight = topk_weight * float(getattr(router, "routed_scaling_factor", 1.0))
            if warn_fn is not None:
                warn_fn(topk_idx)
            return topk_idx, topk_weight

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            router.forward = original_forward  # type: ignore[assignment]

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_ernie_topkrouter_patch(router: nn.Module, valid: list[int], warn_fn) -> bool:
        """Mask ERNIE 5.x router logits before softmax, matching physical compaction."""

        if not valid or router.__class__.__name__ != "Ernie4_5_MoeTopKRouter" or not hasattr(router, "weight"):
            return False
        original_forward = router.forward
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_forward(hidden_states, *args, experts_to_prune=None, **kwargs):
            if args or kwargs:
                return original_forward(hidden_states, *args, **kwargs)
            device_type = (
                hidden_states.device.type
                if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                router_logits = torch.nn.functional.linear(
                    hidden_states.to(dtype=torch.float32),
                    router.weight.to(dtype=torch.float32),
                )
                experts = _resolve_pruned_tensor(experts_to_prune, idx_cpu, router_logits.device)
                router_logits = _mask_logits_inplace(router_logits, experts)
                routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float32)
                corrected_scores = router.moe_statics(routing_weights)
                selected_experts = torch.topk(corrected_scores, int(router.top_k), dim=-1).indices
                routing_weights = torch.gather(routing_weights, dim=-1, index=selected_experts)
                routing_weights = routing_weights / torch.clamp(
                    routing_weights.sum(dim=-1, keepdim=True),
                    min=float(getattr(router, "norm_min", 1e-20)),
                )
            routing_weights = routing_weights.to(dtype=hidden_states.dtype)
            if warn_fn is not None:
                warn_fn(selected_experts)
            return router_logits, selected_experts, routing_weights

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            router.forward = original_forward  # type: ignore[assignment]

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_olmoe_topkrouter_patch(router: nn.Module, valid: list[int], warn_fn) -> bool:
        if not valid or router.__class__.__name__ not in {"OlmoeTopKRouter", "Qwen3MoeTopKRouter"}:
            return False
        if not hasattr(router, "weight") or not hasattr(router, "top_k"):
            return False
        original_forward = router.forward
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_forward(hidden_states, *args, experts_to_prune=None, **kwargs):
            hidden_dim = int(getattr(router, "hidden_dim", hidden_states.shape[-1]))
            top_k_local = int(getattr(router, "top_k"))
            norm_topk_prob = bool(getattr(router, "norm_topk_prob", False))
            hs = hidden_states.reshape(-1, hidden_dim)
            router_logits = torch.nn.functional.linear(hs, router.weight, getattr(router, "bias", None))
            experts = _resolve_pruned_tensor(experts_to_prune, idx_cpu, router_logits.device)
            router_logits = _mask_logits_inplace(router_logits, experts)
            router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_scores, router_indices = torch.topk(router_logits, top_k_local, dim=-1)
            if norm_topk_prob:
                router_scores /= router_scores.sum(dim=-1, keepdim=True)
            router_scores = router_scores.to(router_logits.dtype)
            if warn_fn is not None:
                warn_fn(router_indices)
            return router_logits, router_scores, router_indices

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            router.forward = original_forward  # type: ignore[assignment]

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_deepseek_route_patch(moe: nn.Module, router: nn.Module, valid: list[int], warn_fn) -> bool:
        route_tokens_to_experts = getattr(moe, "route_tokens_to_experts", None)
        if not valid or not isinstance(router, nn.Linear) or not callable(route_tokens_to_experts):
            return False
        original_route_tokens_to_experts = route_tokens_to_experts
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_route_tokens_to_experts(self, router_logits, *args, experts_to_prune=None, **kwargs):
            if torch.is_tensor(router_logits):
                experts = _resolve_pruned_tensor(experts_to_prune, idx_cpu, router_logits.device)
                router_logits = _mask_logits_inplace(router_logits, experts)
            out = original_route_tokens_to_experts(router_logits, *args, **kwargs)
            if warn_fn is not None and isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                warn_fn(out[0])
            return out

        moe.route_tokens_to_experts = MethodType(patched_route_tokens_to_experts, moe)  # type: ignore[method-assign]

        def _restore():
            moe.route_tokens_to_experts = original_route_tokens_to_experts  # type: ignore[method-assign]

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_deepseek_moegate_patch(router: nn.Module, valid: list[int], warn_fn) -> bool:
        if not valid or router.__class__.__name__ != "MoEGate" or not hasattr(router, "weight"):
            return False
        original_forward = router.forward
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_forward(hidden_states, *args, experts_to_prune=None, **kwargs):
            bsz, seq_len, hidden_dim = hidden_states.shape
            flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
            logits = torch.nn.functional.linear(flat_hidden_states.type(torch.float32), router.weight.type(torch.float32), None)
            experts = _resolve_pruned_tensor(experts_to_prune, idx_cpu, logits.device)
            logits = _mask_logits_inplace(logits, experts)
            if getattr(router, "scoring_func", "softmax") != "softmax":
                raise NotImplementedError(f"Unsupported scoring function for masked MoEGate: {getattr(router, 'scoring_func', None)}")
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            if router.topk_method == "greedy":
                topk_weight, topk_idx = torch.topk(scores, k=router.top_k, dim=-1, sorted=False)
            elif router.topk_method == "group_limited_greedy":
                group_scores = scores.view(bsz * seq_len, router.n_group, -1).max(dim=-1).values
                group_idx = torch.topk(group_scores, k=router.topk_group, dim=-1, sorted=False)[1]
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(bsz * seq_len, router.n_group, router.n_routed_experts // router.n_group)
                    .reshape(bsz * seq_len, -1)
                )
                topk_weight, topk_idx = torch.topk(scores.masked_fill(~score_mask.bool(), 0.0), k=router.top_k, dim=-1, sorted=False)
            else:
                raise NotImplementedError(f"Unsupported topk_method for masked MoEGate: {router.topk_method}")

            if router.top_k > 1 and getattr(router, "norm_topk_prob", False):
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                topk_weight = topk_weight * router.routed_scaling_factor
            aux_loss = None
            if warn_fn is not None:
                warn_fn(topk_idx)
            return topk_idx, topk_weight, aux_loss, logits

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            router.forward = original_forward  # type: ignore[assignment]

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_ernie_topkrouter_warning(router: nn.Module, warn_fn) -> bool:
        if warn_fn is None or router.__class__.__name__ != "Ernie4_5_MoeTopKRouter":
            return False
        original_forward = router.forward

        def patched_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            try:
                if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
                    warn_fn(out[1])
            except Exception:
                pass
            return out

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            router.forward = original_forward  # type: ignore[assignment]

        handles.append(_FnHandle(_restore))
        return True

    for layer, pruned in pruned_experts_info.items():
        moe = get_moe(model, layer)
        router = getattr(moe, model_attrs["router"])
        num_experts_layer = num_experts_default
        valid = _as_valid_indices(pruned, num_experts_layer)
        top_k = min(top_k_default, num_experts_layer)

        if valid and (num_experts_layer - len(valid)) < top_k:
            raise ValueError(
                f"Layer {layer}: pruned too many experts for top_k={top_k} "
                f"(num_experts={num_experts_layer}, pruned={len(valid)})."
            )

        warn_fn = None
        if valid:
            def _warn_selected(selected, layer=layer, router_name=router.__class__.__name__, valid=valid):
                _warn_masked_selected(layer, router_name, selected, valid)

            warn_fn = _warn_selected

        ernie_bias_mask_installed = _try_install_ernie_bias_mask(moe, router, valid)
        router_mask_installed = False
        selection_checked_by_patch = False
        if _try_install_olmoe_topkrouter_patch(router, valid, warn_fn):
            router_mask_installed = True
            selection_checked_by_patch = True
        elif _try_install_deepseek_v2_gate_patch(router, valid, warn_fn):
            router_mask_installed = True
            selection_checked_by_patch = True
        elif _try_install_ernie_topkrouter_patch(router, valid, warn_fn):
            router_mask_installed = True
            selection_checked_by_patch = True
        elif _try_install_deepseek_route_patch(moe, router, valid, warn_fn):
            router_mask_installed = True
            selection_checked_by_patch = True
        elif _try_install_deepseek_moegate_patch(router, valid, warn_fn):
            router_mask_installed = True
            selection_checked_by_patch = True
        elif _try_install_linear_gate_patch(router, valid):
            router_mask_installed = True
        elif ernie_bias_mask_installed:
            router_mask_installed = True

        if not router_mask_installed:
            raise RuntimeError(
                "Router mask installation failed for "
                f"{model.__class__.__name__} layer {layer} "
                f"(router={router.__class__.__name__})."
            )

        if warn_fn is not None and not selection_checked_by_patch:
            _try_install_ernie_topkrouter_warning(router, warn_fn)

            def _warn_hook(_module, _inp, out, warn_fn=warn_fn, top_k=top_k, num_experts_layer=num_experts_layer):
                try:
                    _inspect_router_output(out, warn_fn, top_k, num_experts_layer)
                except Exception:
                    pass

            handles.append(router.register_forward_hook(_warn_hook))

    return handles


def _zero_output_like(output: Any) -> Any:
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    if isinstance(output, tuple):
        return tuple(_zero_output_like(item) for item in output)
    if isinstance(output, list):
        return [_zero_output_like(item) for item in output]
    if isinstance(output, dict):
        return {key: _zero_output_like(item) for key, item in output.items()}
    return output


class TemporaryExpertMask:
    """
    Internal candidate-masking primitive for greedy-search scoring.

    It guarantees both:
    - masked experts cannot be selected by routing
    - masked experts contribute zero output even if routing is forced or bypassed
    """

    def __init__(self, model: nn.Module, pruned_experts_info: dict[int, list[int]] | dict[str, list[int]]):
        self.model = model
        self.pruned_experts_info = {int(k): sorted({int(v) for v in vals}) for k, vals in pruned_experts_info.items()}
        self._router_handles: list[Any] = []
        self._expert_handles: list[Any] = []
        self._restore_actions: list[Any] = []

    def __enter__(self):
        self._router_handles = _install_temp_router_masks(self.model, self.pruned_experts_info)
        self._expert_handles = self._install_runtime_expert_masks()
        return self

    def __exit__(self, exc_type, exc, tb):
        _cleanup_handles(self._router_handles)
        self._router_handles = []
        _cleanup_handles(self._expert_handles)
        self._expert_handles = []
        for action in reversed(self._restore_actions):
            try:
                action()
            except Exception:
                logger.exception("Failed restoring temporary expert mask state.")
        self._restore_actions = []

    def _register_restore(self, fn):
        self._restore_actions.append(fn)

    def _install_runtime_expert_masks(self) -> list[Any]:
        handles: list[Any] = []
        for layer_idx, indices in sorted(self.pruned_experts_info.items()):
            if not indices:
                continue
            moe = get_moe(self.model, layer_idx)
            model_attrs = resolve_model_attrs(self.model, moe)
            experts = getattr(moe, model_attrs["experts"], None)
            if experts is None:
                continue

            if model_attrs["fused"]:
                num_experts = getattr(experts, "num_experts", None)
                if num_experts is None:
                    for attr in (model_attrs["gate_proj"], model_attrs["up_proj"], model_attrs["down_proj"]):
                        tensor = getattr(experts, attr, None)
                        if torch.is_tensor(tensor) and tensor.ndim >= 1:
                            num_experts = int(tensor.shape[0])
                            break
                if num_experts is None:
                    continue
                original_forward = experts.forward
                valid_indices = sorted({int(idx) for idx in indices if 0 <= int(idx) < int(num_experts)})
                idx_cpu = torch.tensor(valid_indices, dtype=torch.long)

                def patched_forward(*args, _orig=original_forward, _idx_cpu=idx_cpu, **kwargs):
                    selected = None
                    weights = None
                    selected_from_kwargs = None
                    weights_from_kwargs = None

                    if len(args) >= 3 and torch.is_tensor(args[1]) and torch.is_tensor(args[2]):
                        selected = args[1]
                        weights = args[2]
                    else:
                        for key in ("selected_experts", "top_k_index", "topk_idx"):
                            value = kwargs.get(key)
                            if torch.is_tensor(value):
                                selected = value
                                selected_from_kwargs = key
                                break
                        for key in ("routing_weights", "top_k_weights", "topk_weight"):
                            value = kwargs.get(key)
                            if torch.is_tensor(value):
                                weights = value
                                weights_from_kwargs = key
                                break

                    if (
                        selected is None
                        or weights is None
                        or not torch.is_tensor(selected)
                        or not torch.is_tensor(weights)
                        or _idx_cpu.numel() == 0
                    ):
                        return _orig(*args, **kwargs)

                    experts_to_zero = _idx_cpu.to(device=selected.device, dtype=torch.long)
                    try:
                        selected_mask = torch.isin(selected.to(dtype=torch.long), experts_to_zero)
                    except Exception:
                        selected_mask = (selected.to(dtype=torch.long).unsqueeze(-1) == experts_to_zero).any(dim=-1)
                    if not bool(selected_mask.any()):
                        return _orig(*args, **kwargs)

                    masked_weights = weights.masked_fill(selected_mask.to(device=weights.device), 0)
                    if len(args) >= 3 and torch.is_tensor(args[1]) and torch.is_tensor(args[2]):
                        patched_args = list(args)
                        patched_args[2] = masked_weights
                        return _orig(*patched_args, **kwargs)

                    patched_kwargs = dict(kwargs)
                    if selected_from_kwargs is not None:
                        patched_kwargs[selected_from_kwargs] = selected
                    if weights_from_kwargs is not None:
                        patched_kwargs[weights_from_kwargs] = masked_weights
                    return _orig(*args, **patched_kwargs)

                experts.forward = patched_forward  # type: ignore[assignment]

                def _restore(experts=experts, original_forward=original_forward):
                    try:
                        experts.forward = original_forward  # type: ignore[assignment]
                    except Exception:
                        pass

                self._register_restore(_restore)
                continue

            if not isinstance(experts, (nn.ModuleList, list, tuple)):
                continue

            for idx in sorted({int(expert_idx) for expert_idx in indices if 0 <= int(expert_idx) < len(experts)}):
                expert_module = experts[idx]

                def _zero_hook(_module, _args, output):
                    return _zero_output_like(output)

                handles.append(expert_module.register_forward_hook(_zero_hook))
        return handles
