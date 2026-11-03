from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

from moe_pba.args import (
    MOE_PBA_CROSS_LAYER_MODE,
    MOE_PBA_INCLUDE_SHARED_EXPERTS,
    MOE_PBA_INPUT_CONDITIONING,
    MOE_PBA_METRIC,
    MOE_PBA_OBJECTIVE,
    MOE_PBA_REFERENCE,
    MOE_PBA_SCORE_REDUCTION,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def build_run_request(
    *,
    model_name: str,
    method: str,
    run_args: Any,
    ds_args: Any,
    obs_args: Any,
    model_args: Any,
    moe_pba_args: Any,
) -> dict[str, Any]:
    moe = _as_dict(moe_pba_args)
    # These values are runtime/output controls, not part of the search identity.
    moe.pop("gpus", None)
    moe.pop("output_dir", None)
    moe.pop("resolved_prune_count", None)
    payload = {
        "model_name": str(model_name),
        "method": str(method),
        "seed": int(getattr(run_args, "seed", 42)),
        "model_args": _as_dict(model_args),
        "dataset_args": _as_dict(ds_args),
        "observer_args": _as_dict(obs_args),
        "moe_pba_args": moe,
        "algorithm": {
            "metric": MOE_PBA_METRIC,
            "score_reduction": MOE_PBA_SCORE_REDUCTION,
            "objective": MOE_PBA_OBJECTIVE,
            "reference": MOE_PBA_REFERENCE,
            "cross_layer_mode": MOE_PBA_CROSS_LAYER_MODE,
            "input_conditioning": MOE_PBA_INPUT_CONDITIONING,
            "include_shared_experts": MOE_PBA_INCLUDE_SHARED_EXPERTS,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "version": 2,
        "request_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "parameters": payload,
    }


def write_run_request(
    output_dir: pathlib.Path,
    request: dict[str, Any],
    *,
    calibration_fingerprint: str | None = None,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_request.json"
    existing: dict[str, Any] | None = None
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Cannot read existing run request at {path}.") from exc
        if existing.get("request_fingerprint") != request.get("request_fingerprint"):
            raise RuntimeError(
                f"Output directory {output_dir} belongs to a different search request. "
                "Choose a new --output_dir or remove the incomplete run explicitly."
            )
        old_calibration = existing.get("calibration_fingerprint")
        if old_calibration and calibration_fingerprint and old_calibration != calibration_fingerprint:
            raise RuntimeError(
                f"Calibration fingerprint mismatch for {output_dir}; refusing to reuse stale search artifacts."
            )
    elif any((output_dir / name).exists() for name in ("search_history.json", "config.json", "pruned_experts.json")):
        raise RuntimeError(
            f"Output directory {output_dir} contains artifacts without run_request.json. "
            "Choose a new --output_dir to avoid mixing runs."
        )

    merged = dict(existing or request)
    if calibration_fingerprint is not None:
        merged["calibration_fingerprint"] = str(calibration_fingerprint)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
