from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any

import yaml

from moe_pba.common.materialize import materialize_uniform_model


def dump_args_to_yaml(output_dir: pathlib.Path, **all_args: Any) -> None:
    payload = {}
    for key, value in all_args.items():
        try:
            payload[key] = dataclasses.asdict(value)
        except TypeError:
            payload[key] = value
    with open(output_dir / "moe_pba_args.yaml", "w") as f:
        yaml.dump(payload, f, default_flow_style=False)


def _history_search_started_at(history: Any) -> float | None:
    if not isinstance(history, list):
        return None

    starts: list[float] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for key in ("search_start_time", "search_started_unix_time", "started_unix_time"):
            value = entry.get(key)
            if value is None:
                continue
            try:
                starts.append(float(value))
            except (TypeError, ValueError):
                continue
    if not starts:
        return None
    return min(starts)


def save_method_result(
    model,
    tokenizer,
    output_dir: pathlib.Path,
    method_result,
    method_name: str,
    model_args,
    ds_args,
    obs_args,
    run_args,
    moe_pba_args,
    search_started_at: float | None = None,
    search_finished_at: float | None = None,
    run_request: dict[str, Any] | None = None,
) -> pathlib.Path:
    started_at = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    pruned_experts = method_result.pruned_experts

    resolved_search_finished_at = None
    search_duration_seconds = None
    search_duration_hours = None
    effective_search_started_at = search_started_at
    score_summary_started_at = method_result.score_summary.get("search_started_unix_time")
    if score_summary_started_at is not None:
        effective_search_started_at = float(score_summary_started_at)
    elif method_result.history:
        history_started_at = _history_search_started_at(method_result.history)
        if history_started_at is not None:
            effective_search_started_at = float(history_started_at)

    if effective_search_started_at is not None:
        resolved_search_finished_at = started_at if search_finished_at is None else float(search_finished_at)
        search_duration_seconds = max(
            0.0,
            float(resolved_search_finished_at) - float(effective_search_started_at),
        )
        search_duration_hours = search_duration_seconds / 3600.0
        method_result.score_summary["search_started_unix_time"] = float(effective_search_started_at)
        method_result.score_summary["search_finished_unix_time"] = float(resolved_search_finished_at)
        method_result.score_summary["search_duration_seconds"] = float(search_duration_seconds)
        method_result.score_summary["search_duration_hours"] = float(search_duration_hours)

    materialize_uniform_model(model, tokenizer, pruned_experts, output_dir)

    with open(output_dir / "pruning_plan.json", "w") as f:
        json.dump(
            {
                "method": method_name,
                "plan": method_result.plan_summary,
                "score_summary": method_result.score_summary,
            },
            f,
            indent=2,
        )
    with open(output_dir / "search_metadata.json", "w") as f:
        request_fingerprint = None if run_request is None else run_request.get("request_fingerprint")
        json.dump(
            {
                "method": method_name,
                "output_dir": str(output_dir),
                "calibration_fingerprint": method_result.score_summary.get("calibration_fingerprint"),
                "budget_mode": "per_layer",
                "checkpoint_format": "transformers_uniform",
                "run_request_fingerprint": request_fingerprint,
                "run_args": dataclasses.asdict(run_args),
                "moe_pba_args": dataclasses.asdict(moe_pba_args),
                "algorithm": {
                    key: method_result.score_summary[key]
                    for key in (
                        "objective", "metric", "score_reduction", "reference", "cross_layer_mode",
                        "input_conditioning", "include_shared_experts", "within_layer_ordering",
                    )
                },
                "search_started_unix_time": effective_search_started_at,
                "search_finished_unix_time": resolved_search_finished_at,
                "search_duration_seconds": search_duration_seconds,
                "search_duration_hours": search_duration_hours,
                "save_started_unix_time": started_at,
            },
            f,
            indent=2,
        )
    if method_result.history:
        with open(output_dir / "search_history.json", "w") as f:
            json.dump(method_result.history, f, indent=2)

    dump_args_to_yaml(
        output_dir,
        run_args=run_args,
        ds_args=ds_args,
        obs_args=obs_args,
        model_args=model_args,
        moe_pba_args=moe_pba_args,
    )
    return output_dir
