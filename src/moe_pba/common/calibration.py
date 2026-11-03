from __future__ import annotations

import gc
import hashlib
import logging
import math
from typing import Any

import torch

from moe_pba.data import DATASET_REGISTRY
from moe_pba.args import GreedySearchArgs
from moe_pba.common.batching import batch_to_tensors
from moe_pba.common.types import CalibrationContext

logger = logging.getLogger(__name__)


def _normalize_calib_microbatch_size(raw_value: Any) -> str:
    normalized = str(raw_value or "8").strip().lower()
    if normalized in {"", "auto", "0"}:
        return "auto"
    try:
        resolved = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"calib_microbatch_size must be a positive integer or `auto`. Received {raw_value!r}."
        ) from exc
    if resolved <= 0:
        raise ValueError("calib_microbatch_size must be a positive integer or `auto`.")
    return str(resolved)


def _batch_token_length(batch: Any) -> int:
    input_ids, _attention_mask, _labels_mask = batch_to_tensors(batch)
    return int(input_ids.shape[-1])


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _select_auto_microbatch_size(
    model,
    search_batches: list[Any],
    *,
    pad_token_id: int,
    max_try: int | None = None,
) -> int:
    if model is None:
        raise ValueError("calib_microbatch_size=auto requires the loaded model.")
    if not search_batches:
        return 1

    from moe_pba.common.scoring import score_model_nll

    ordered_batches = sorted(search_batches, key=_batch_token_length, reverse=True)
    limit = len(ordered_batches) if max_try is None else max(1, min(int(max_try), len(ordered_batches)))
    if limit < 2:
        return 1

    def _try_microbatch(size: int) -> bool:
        probe_batches = _rebatch(list(ordered_batches), size, pad_token_id=pad_token_id)
        try:
            _ = score_model_nll(model, probe_batches, None)
            return True
        except BaseException as exc:
            if not _is_cuda_oom(exc):
                raise
            return False
        finally:
            try:
                del probe_batches
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    last_ok = 1
    candidate = 2
    while candidate <= limit and _try_microbatch(candidate):
        last_ok = candidate
        candidate *= 2

    lo = last_ok
    hi = min(limit, candidate - 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _try_microbatch(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _apply_auto_microbatch_headroom(largest_fit: int) -> int:
    largest_fit = max(1, int(largest_fit))
    if largest_fit <= 1:
        return 1
    headroom = max(1, largest_fit // 2)
    return 1 << (headroom.bit_length() - 1)


def _collate_microbatch(items: list[Any], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot collate an empty microbatch.")

    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor | None] = []
    labels_mask_list: list[torch.Tensor | None] = []
    max_len = 0
    need_attention_mask = False
    need_labels_mask = False

    for item in items:
        input_ids, attention_mask, labels_mask = batch_to_tensors(item)
        input_ids = input_ids.detach().cpu()
        input_ids_list.append(input_ids)
        max_len = max(max_len, int(input_ids.shape[1]))

        if attention_mask is not None:
            need_attention_mask = True
            attention_mask = attention_mask.detach().cpu()
        attention_mask_list.append(attention_mask)

        if labels_mask is not None:
            need_labels_mask = True
            labels_mask = labels_mask.detach().cpu()
        labels_mask_list.append(labels_mask)

    def _pad_2d(t: torch.Tensor, value: int) -> torch.Tensor:
        pad = max_len - int(t.shape[1])
        if pad <= 0:
            return t
        return torch.nn.functional.pad(t, (0, pad), value=value)

    batch_input_ids = torch.cat([_pad_2d(t, pad_token_id) for t in input_ids_list], dim=0)

    batch_attention_mask: torch.Tensor | None = None
    if need_attention_mask or any(int(t.shape[1]) != max_len for t in input_ids_list):
        masks = []
        for idx, ids in enumerate(input_ids_list):
            attn = attention_mask_list[idx]
            if attn is None:
                attn = torch.ones_like(ids, dtype=torch.long)
            masks.append(_pad_2d(attn.to(dtype=torch.long), 0))
        batch_attention_mask = torch.cat(masks, dim=0)

    batch_labels_mask: torch.Tensor | None = None
    if need_labels_mask:
        masks = []
        for idx, ids in enumerate(input_ids_list):
            mask = labels_mask_list[idx]
            if mask is None:
                mask = torch.ones_like(ids, dtype=torch.long)
            masks.append(_pad_2d(mask.to(dtype=torch.long), 0))
        batch_labels_mask = torch.cat(masks, dim=0)

    out: dict[str, torch.Tensor] = {"input_ids": batch_input_ids}
    if batch_attention_mask is not None:
        out["attention_mask"] = batch_attention_mask
    if batch_labels_mask is not None:
        out["labels_mask"] = batch_labels_mask
    return out


def _rebatch(search_batches: list[Any], microbatch_size: int, pad_token_id: int) -> list[Any]:
    if microbatch_size <= 1:
        return search_batches
    out: list[Any] = []
    for i in range(0, len(search_batches), microbatch_size):
        out.append(_collate_microbatch(search_batches[i : i + microbatch_size], pad_token_id=pad_token_id))
    return out


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_dataset_list(value: str | None, fallback: str | None = None) -> list[str]:
    raw = value or fallback
    if not raw:
        return []
    items = _split_csv(raw)
    dataset_names = items if items else [raw.strip()]
    unsupported = [name for name in dataset_names if name not in DATASET_REGISTRY]
    if unsupported:
        raise ValueError(
            f"Unsupported calibration dataset(s): {unsupported}. "
            f"Supported: {list(DATASET_REGISTRY)}"
        )
    return dataset_names


def _parse_weight_list(weights: str | None, count: int) -> list[float]:
    if not weights:
        return [1.0] * count
    items = _split_csv(weights)
    if len(items) != count:
        raise ValueError(
            "calib_dataset_weights must match calib_dataset_name count "
            f"(expected {count}, got {len(items)})"
        )
    out: list[float] = []
    for item in items:
        weight = float(item)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("calib_dataset_weights must contain finite, positive values.")
        out.append(weight)
    return out


def _allocate_dataset_samples(total: int, weights: list[float]) -> list[int]:
    if total <= 0:
        raise ValueError("calib_samples must be > 0.")
    if not weights:
        raise ValueError("At least one calibration dataset weight is required.")
    if total < len(weights):
        raise ValueError(
            "calib_samples must be at least the number of calibration datasets "
            f"({total} < {len(weights)})."
        )
    counts = [1] * len(weights)
    remaining = total - len(weights)
    if remaining <= 0:
        return counts
    total_weight = sum(weights)
    raw = [remaining * (w / total_weight) for w in weights]
    floors = [int(x) for x in raw]
    counts = [base + inc for base, inc in zip(counts, floors)]
    leftover = remaining - sum(floors)
    if leftover > 0:
        fractional = sorted(
            enumerate(raw),
            key=lambda item: item[1] - floors[item[0]],
            reverse=True,
        )
        for idx in range(leftover):
            counts[fractional[idx % len(fractional)][0]] += 1
    return counts


def _mix_short_dataset_name(dataset_name: str) -> str:
    base = dataset_name.split("/")[-1]
    base_lower = base.lower()
    if base_lower.startswith("tulu-3"):
        return "tulu"
    if "evol-codealpaca" in base_lower:
        return "evol"
    return base.replace("/", "_")


def build_calibration_tag(dataset_names: list[str], weights: list[float]) -> str:
    parts: list[str] = []
    for dataset_name, weight in zip(dataset_names, weights):
        parts.append(_mix_short_dataset_name(dataset_name))
        if abs(weight - round(weight)) < 1e-6:
            parts.append(str(int(round(weight))))
        else:
            parts.append(f"{weight:g}")
    return "_".join(parts)


def resolve_calibration_tag(search_args: GreedySearchArgs) -> str:
    dataset_names = _parse_dataset_list(search_args.calib_dataset_name)
    if not dataset_names:
        raise RuntimeError("No calibration dataset specified. Set calib_dataset_name explicitly for moe_pba.")
    weights = _parse_weight_list(search_args.calib_dataset_weights, len(dataset_names))
    return build_calibration_tag(dataset_names, weights)


def fingerprint_batches(search_batches: list[Any]) -> str:
    hasher = hashlib.sha256()
    for batch in search_batches:
        input_ids, attention_mask, labels_mask = batch_to_tensors(batch)
        for tensor in (input_ids, attention_mask, labels_mask):
            if tensor is None:
                hasher.update(b"<none>")
                continue
            hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
            hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _load_raw_dataset(dataset_name: str, dataset_config: str | None, split: str):
    from datasets import load_dataset

    if dataset_config:
        try:
            return load_dataset(dataset_name, name=dataset_config, split=split)
        except ValueError:
            return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, split=split)


def _resolve_max_input_len(tokenizer, obs_args) -> int:
    max_input_len = getattr(obs_args, "model_max_length", None)
    if max_input_len is None:
        max_input_len = getattr(tokenizer, "model_max_length", None)
    if max_input_len is None:
        raise ValueError("Could not resolve max input length for calibration.")
    return int(max_input_len)


def _build_standard_batches(
    tokenizer,
    search_args: GreedySearchArgs,
    obs_args,
    dataset_names: list[str],
    targets: list[int],
    dataset_config: str | None,
    split: str,
) -> list[Any]:
    batches: list[Any] = []
    max_input_len = _resolve_max_input_len(tokenizer, obs_args)
    for dataset_name, target_total in zip(dataset_names, targets):
        proc_cls = DATASET_REGISTRY.get(dataset_name)
        if proc_cls is None:
            raise ValueError(
                f"No DatasetProcessor registered for calibration dataset '{dataset_name}'. "
                f"Supported: {list(DATASET_REGISTRY.keys())}"
            )
        raw_ds = _load_raw_dataset(dataset_name, dataset_config, split)

        processor = proc_cls(
            dataset=raw_ds,
            tokenizer=tokenizer,
            pack_samples=search_args.calib_pack_samples,
            max_input_len=max_input_len,
            split=split,
            truncate=obs_args.truncate,
        )
        # Sampling uses the process-global RNG initialized by main._set_seed().
        dataset_batches = processor.get_processed_dataset(
            num_samples=target_total,
            return_loss_mask=True,
        )
        if not dataset_batches:
            raise ValueError(
                f"Calibration dataset '{dataset_name}' produced no usable sequences. "
                "Check that the split contains samples and that model_max_length/truncate "
                "allow nonempty inputs."
            )
        batches.extend(dataset_batches)
    return batches


def prepare_calibration_context(
    tokenizer,
    search_args: GreedySearchArgs,
    ds_args,
    obs_args,
    *,
    model=None,
) -> CalibrationContext:
    dataset_config = search_args.calib_dataset_config_name or ds_args.dataset_config_name
    split = search_args.calib_dataset_split or ds_args.split
    dataset_names = _parse_dataset_list(search_args.calib_dataset_name)
    if not dataset_names:
        raise RuntimeError("No calibration dataset specified. Set calib_dataset_name explicitly for moe_pba.")
    weights = _parse_weight_list(search_args.calib_dataset_weights, len(dataset_names))
    targets = _allocate_dataset_samples(search_args.calib_samples, weights)
    requested_microbatch = _normalize_calib_microbatch_size(search_args.calib_microbatch_size)

    logger.info(
        "Loading greedy-search calibration dataset(s)=%s config=%s split=%s mode=standard",
        ",".join(dataset_names),
        dataset_config,
        split,
    )

    batches = _build_standard_batches(
        tokenizer,
        search_args,
        obs_args,
        dataset_names,
        targets,
        dataset_config,
        split,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    if requested_microbatch == "auto":
        probed_microbatch_size = _select_auto_microbatch_size(
            model,
            list(batches),
            pad_token_id=int(pad_token_id),
            max_try=64,
        )
        microbatch_size = _apply_auto_microbatch_headroom(probed_microbatch_size)
    else:
        microbatch_size = int(requested_microbatch)
    old_batch_count = len(batches)
    batches = _rebatch(list(batches), microbatch_size, pad_token_id=pad_token_id)
    if requested_microbatch == "auto":
        logger.info(
            "Using calib_microbatch_size=%d (auto=True, probed_max=%d, safety_margin=2x): %d -> %d calibration batches",
            microbatch_size,
            probed_microbatch_size,
            old_batch_count,
            len(batches),
        )
    else:
        logger.info(
            "Using calib_microbatch_size=%d (auto=False): %d -> %d calibration batches",
            microbatch_size,
            old_batch_count,
            len(batches),
        )

    return CalibrationContext(
        batches=batches,
        fingerprint=fingerprint_batches(batches),
        dataset_tag=build_calibration_tag(dataset_names, weights),
        batch_count=len(batches),
    )
