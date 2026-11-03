from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from moe_pba.common.batching import batch_to_tensors
from moe_pba.common.masking import TemporaryExpertMask
from moe_pba.model_util import get_moe

logger = logging.getLogger(__name__)
_REPLAY_CACHE_META_KEY = "__replay_cache_meta__"


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def module_device(module, *, fallback: torch.device | None = None) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback if fallback is not None else torch.device("cpu")


def move_batch_to_device(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    def _move(value: Any) -> Any:
        if not torch.is_tensor(value):
            return value
        return value.to(
            device,
            non_blocking=value.device.type == "cpu" and device.type == "cuda",
        )

    if isinstance(batch, dict):
        return {key: _move(value) for key, value in batch.items()}
    input_ids, attention_mask, labels_mask = batch_to_tensors(batch)
    out = {"input_ids": _move(input_ids)}
    if attention_mask is not None:
        out["attention_mask"] = _move(attention_mask)
    if labels_mask is not None:
        out["labels_mask"] = _move(labels_mask)
    return out


def _forward_without_kv_cache(model, **kwargs):
    try:
        return model(use_cache=False, **kwargs)
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        return model(**kwargs)


def _detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _move_nested_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    return value


def _device_cache_key(device: torch.device) -> str:
    if device.index is None:
        return str(device.type)
    return f"{device.type}:{int(device.index)}"


def _replay_cache_meta(replay_cache: list[dict[str, Any]]) -> dict[str, Any]:
    if not replay_cache or not isinstance(replay_cache[0], dict):
        return {}
    meta = replay_cache[0].get(_REPLAY_CACHE_META_KEY)
    if not isinstance(meta, dict):
        meta = {}
        replay_cache[0][_REPLAY_CACHE_META_KEY] = meta
    hot_by_device = meta.get("hot_by_device")
    if not isinstance(hot_by_device, dict):
        hot_by_device = {}
        meta["hot_by_device"] = hot_by_device
    oom_devices = meta.get("oom_devices")
    if not isinstance(oom_devices, set):
        oom_devices = set(oom_devices or [])
        meta["oom_devices"] = oom_devices
    return meta


def _promote_replay_sample_to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: _move_nested_to_device(value, device)
        for key, value in sample.items()
        if key != _REPLAY_CACHE_META_KEY
    }


def _maybe_get_hot_replay_cache(
    replay_cache: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]] | None:
    if device.type != "cuda" or not replay_cache:
        return None

    meta = _replay_cache_meta(replay_cache)
    if not meta:
        return None
    device_key = _device_cache_key(device)
    hot_by_device = meta["hot_by_device"]
    if device_key in hot_by_device:
        return hot_by_device[device_key]
    if device_key in meta["oom_devices"]:
        return None

    promoted: list[dict[str, Any]] = []
    try:
        for sample in replay_cache:
            promoted.append(_promote_replay_sample_to_device(sample, device))
    except BaseException as exc:
        if not (
            isinstance(exc, torch.cuda.OutOfMemoryError)
            or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())
        ):
            raise
        meta["oom_devices"].add(device_key)
        promoted.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning(
            "Replay-cache hot promotion OOM on %s; falling back to cold-cache transfers for this cache.",
            device_key,
        )
        return None

    hot_by_device[device_key] = promoted
    return promoted


def release_replay_cache_hot(
    replay_cache: list[dict[str, Any]] | None,
    *,
    device: torch.device | None = None,
) -> None:
    if not replay_cache:
        return
    meta = _replay_cache_meta(replay_cache)
    if not meta:
        return
    hot_by_device = meta.get("hot_by_device")
    if not isinstance(hot_by_device, dict):
        return
    if device is None:
        hot_by_device.clear()
    else:
        hot_by_device.pop(_device_cache_key(device), None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_replay_cache_collection_hot(
    replay_caches: dict[Any, list[dict[str, Any]]] | list[list[dict[str, Any]]] | None,
    *,
    device: torch.device | None = None,
) -> None:
    if replay_caches is None:
        return
    if isinstance(replay_caches, dict):
        iterable = replay_caches.values()
    else:
        iterable = replay_caches
    for replay_cache in iterable:
        release_replay_cache_hot(replay_cache, device=device)


def _extract_hidden_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError(f"Unsupported MoE output type for replay scoring: {type(output)!r}")


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported model output type for causal scoring: {type(output)!r}")


def _causal_valid_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_mask = torch.ones_like(input_ids[..., 1:], dtype=torch.bool)
    if attention_mask is not None:
        valid_mask &= attention_mask[..., 1:].to(dtype=torch.bool)
    if labels_mask is not None:
        valid_mask &= labels_mask[..., 1:].to(dtype=torch.bool)
    if not torch.any(valid_mask):
        raise RuntimeError("No valid tokens remain after applying attention/labels masks.")
    return valid_mask


def compute_causal_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    valid_mask = _causal_valid_mask(
        input_ids,
        attention_mask=attention_mask,
        labels_mask=labels_mask,
    )

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = valid_mask.view(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])


def score_model_nll(model, batches: list[Any], pruned_experts_info: dict[int, list[int]] | None = None) -> float:
    device = model_device(model)
    total_loss = 0.0
    count = 0
    model_was_training = model.training
    model.eval()
    ctx = TemporaryExpertMask(model, pruned_experts_info or {})
    with torch.inference_mode():
        with ctx:
            for batch in batches:
                moved = move_batch_to_device(batch, device)
                outputs = _forward_without_kv_cache(
                    model,
                    input_ids=moved["input_ids"],
                    attention_mask=moved.get("attention_mask"),
                )
                loss = compute_causal_ce_loss(
                    _extract_logits(outputs),
                    moved["input_ids"],
                    attention_mask=moved.get("attention_mask"),
                    labels_mask=moved.get("labels_mask"),
                )
                total_loss += float(loss.detach().cpu().item())
                count += 1
    if model_was_training:
        model.train()
    return total_loss / max(count, 1)


def capture_all_layer_replay_caches(
    model,
    batches: list[Any],
    layer_indices: list[int],
    pruned_experts_info: dict[int, list[int]] | None = None,
    *,
    capture_inputs: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    """Capture complete MoE outputs, including any shared-expert branches."""
    device = model_device(model)
    caches: dict[int, list[dict[str, Any]]] = {int(layer_idx): [] for layer_idx in layer_indices}
    originals = []
    current_attention_mask: torch.Tensor | None = None
    current_num_examples = 0

    for layer_idx in layer_indices:
        moe = get_moe(model, int(layer_idx))
        original_forward = moe.forward

        def patched_forward(*args, _layer_idx=int(layer_idx), _orig=original_forward, **kwargs):
            output = _orig(*args, **kwargs)
            entry = {
                "output": _extract_hidden_tensor(output).detach().cpu(),
                "num_examples": current_num_examples,
            }
            if capture_inputs:
                entry["args"] = _detach_to_cpu(args)
                entry["kwargs"] = _detach_to_cpu(kwargs)
                if current_attention_mask is not None:
                    entry["attention_mask"] = current_attention_mask.detach().cpu()
            caches[_layer_idx].append(entry)
            return output

        originals.append((moe, original_forward))
        moe.forward = patched_forward  # type: ignore[assignment]

    model_was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            with TemporaryExpertMask(model, pruned_experts_info or {}):
                for batch in batches:
                    moved = move_batch_to_device(batch, device)
                    current_attention_mask = moved.get("attention_mask")
                    current_num_examples = int(moved["input_ids"].shape[0])
                    _forward_without_kv_cache(
                        model,
                        input_ids=moved["input_ids"],
                        attention_mask=moved.get("attention_mask"),
                    )
    finally:
        current_attention_mask = None
        for moe, original_forward in reversed(originals):
            moe.forward = original_forward  # type: ignore[assignment]
        if model_was_training:
            model.train()
    return caches


def build_prefix_conditioned_replay_cache(
    full_reference_cache: list[dict[str, Any]],
    prefix_input_cache: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine dense reference outputs with inputs captured under a pruned prefix."""

    if len(full_reference_cache) != len(prefix_input_cache):
        raise ValueError(
            "Full-reference and prefix-input replay caches must contain the same number of samples: "
            f"{len(full_reference_cache)} != {len(prefix_input_cache)}."
        )

    combined: list[dict[str, Any]] = []
    for full_sample, prefix_sample in zip(full_reference_cache, prefix_input_cache):
        if full_sample["num_examples"] != prefix_sample["num_examples"]:
            raise ValueError("Full-reference and prefix-input replay batches must contain the same number of examples.")
        sample = {
            key: value
            for key, value in prefix_sample.items()
            if key != _REPLAY_CACHE_META_KEY
        }
        sample["output"] = full_sample["output"]
        combined.append(sample)
    return combined


def _compute_hidden_replay_squared_l2(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
    *,
    attention_mask: torch.Tensor | None = None,
) -> float:
    """Sum squared errors over the replay batch's valid tokens and hidden dimensions."""
    reference = reference.to(device=device, dtype=torch.float32)
    candidate = candidate.to(device=device, dtype=torch.float32)
    if reference.shape != candidate.shape:
        raise ValueError(
            "Reference and candidate tensors must have matching shapes for squared L2 replay scoring: "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}."
        )
    difference = candidate - reference
    squared_error = difference.square()
    if attention_mask is not None:
        if squared_error.ndim < 2 or attention_mask.numel() * squared_error.shape[-1] != squared_error.numel():
            raise ValueError(
                "Attention mask must match the replay output's token dimensions: "
                f"{tuple(attention_mask.shape)} vs {tuple(squared_error.shape)}."
            )
        valid_tokens = attention_mask.to(device=device, dtype=torch.bool).reshape(squared_error.shape[:-1])
        squared_error.masked_fill_(~valid_tokens.unsqueeze(-1), 0)
    return float(squared_error.sum().item())


def score_layer_replay_squared_l2(
    model,
    layer_idx: int,
    replay_cache: list[dict[str, Any]],
    pruned_experts_info: dict[int, list[int]] | None = None,
) -> float:
    """Mean per-example squared L2 drift of complete MoE outputs from the full-model reference.

    Sum over valid tokens and hidden dimensions, then divide by the total number
    of examples, including when the last microbatch is smaller than the others.
    """
    if not replay_cache:
        return 0.0

    moe = get_moe(model, layer_idx)
    device = module_device(moe, fallback=model_device(model))
    total = 0.0
    count = 0
    model_was_training = model.training
    hot_samples = _maybe_get_hot_replay_cache(replay_cache, device)
    model.eval()
    try:
        with torch.inference_mode():
            with TemporaryExpertMask(model, pruned_experts_info or {}):
                samples_iter = hot_samples if hot_samples is not None else replay_cache
                for sample in samples_iter:
                    num_examples = int(sample["num_examples"])
                    if num_examples <= 0:
                        raise ValueError("Replay batches must contain at least one example.")
                    if hot_samples is not None:
                        args = sample["args"]
                        kwargs = sample["kwargs"]
                    else:
                        args = _move_nested_to_device(sample["args"], device)
                        kwargs = _move_nested_to_device(sample["kwargs"], device)

                    reference = sample["output"]
                    candidate_output = moe(*args, **kwargs)
                    candidate = _extract_hidden_tensor(candidate_output).detach()

                    total += _compute_hidden_replay_squared_l2(
                        reference, candidate, device, attention_mask=sample.get("attention_mask")
                    )
                    count += num_examples
    finally:
        if model_was_training:
            model.train()
    return total / max(count, 1)
