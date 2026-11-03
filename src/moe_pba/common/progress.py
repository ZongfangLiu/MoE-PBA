from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

from moe_pba.args import MOE_PBA_INPUT_CONDITIONING, MOE_PBA_METRIC, MOE_PBA_SCORE_REDUCTION

logger = logging.getLogger(__name__)


def progress_log_every(total: int, updates: int = 4) -> int:
    total = max(1, int(total))
    updates = max(1, int(updates))
    return max(1, (total + updates - 1) // updates)


def persist_search_history(
    output_dir: str | pathlib.Path | None,
    history: list[dict[str, Any]],
) -> None:
    if output_dir is None:
        return

    target_dir = pathlib.Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target_dir / "search_history.json.tmp"
    final_path = target_dir / "search_history.json"

    try:
        with open(tmp_path, "w") as f:
            json.dump(history, f, indent=2)
        tmp_path.replace(final_path)
    except Exception:
        logger.exception("Failed to persist incremental moe_pba history to %s", final_path)


def load_search_history(output_dir: str | pathlib.Path | None) -> list[dict[str, Any]]:
    if output_dir is None:
        return []

    final_path = pathlib.Path(output_dir) / "search_history.json"
    if not final_path.exists():
        return []

    try:
        with open(final_path) as f:
            payload = json.load(f)
    except Exception:
        logger.exception("Failed to load moe_pba history from %s", final_path)
        return []

    if not isinstance(payload, list):
        logger.warning("Ignoring malformed moe_pba history at %s: expected a JSON list.", final_path)
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def restore_completed_layer_history(
    output_dir: str | pathlib.Path | None,
    *,
    layers: list[Any],
    expected_objective: str,
    prune_count: int,
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    raw_history = load_search_history(output_dir)
    if not raw_history:
        return [], {}

    restored_history: list[dict[str, Any]] = []
    committed: dict[int, list[int]] = {}
    expected_layer_indices = [int(getattr(layer, "layer_idx")) for layer in layers]

    for position, entry in enumerate(raw_history):
        if entry.get("metric") != MOE_PBA_METRIC or entry.get("score_reduction") != MOE_PBA_SCORE_REDUCTION:
            raise ValueError(
                f"Resume history at {output_dir} uses an incompatible or unspecified scoring metric. "
                f"Expected {MOE_PBA_METRIC} with {MOE_PBA_SCORE_REDUCTION} reduction. "
                "Choose a new output directory or resume a run with matching scoring settings."
            )
        if position >= len(expected_layer_indices):
            logger.warning("Ignoring moe_pba resume history with extra completed layers in %s.", output_dir)
            return [], {}

        if str(entry.get("status", "")) != "completed":
            break
        if str(entry.get("objective", "")) != expected_objective:
            logger.warning(
                "Ignoring moe_pba resume history at %s due to objective mismatch: expected %s, found %s.",
                output_dir,
                expected_objective,
                entry.get("objective"),
            )
            return [], {}

        input_conditioning = str(entry.get("input_conditioning", MOE_PBA_INPUT_CONDITIONING))
        if input_conditioning != MOE_PBA_INPUT_CONDITIONING or not entry.get("include_shared_experts", True):
            raise ValueError(
                f"Resume history at {output_dir} uses incompatible scoring. "
                "MoE-PBA requires prefix conditioning and shared-expert outputs."
            )

        try:
            layer_idx = int(entry.get("layer"))
        except Exception:
            logger.warning("Ignoring malformed moe_pba resume history at %s: invalid layer id.", output_dir)
            return [], {}

        if layer_idx != expected_layer_indices[position]:
            logger.warning(
                "Ignoring moe_pba resume history at %s due to layer-order mismatch: expected layer %d, found %d.",
                output_dir,
                expected_layer_indices[position],
                layer_idx,
            )
            return [], {}

        selected_raw = entry.get("selected", [])
        if not isinstance(selected_raw, list):
            logger.warning("Ignoring malformed moe_pba resume history at %s: selected must be a list.", output_dir)
            return [], {}
        try:
            selected_values = [int(idx) for idx in selected_raw]
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed moe_pba resume history at %s: invalid expert id.", output_dir)
            return [], {}
        selected = sorted(set(selected_values))
        layer_num_experts = int(getattr(layers[position], "num_experts"))
        invalid = [idx for idx in selected if idx < 0 or idx >= layer_num_experts]
        if len(selected) != len(selected_values) or invalid:
            logger.warning(
                "Ignoring malformed moe_pba resume history at %s: duplicate or out-of-range experts on layer %d.",
                output_dir,
                layer_idx,
            )
            return [], {}
        if len(selected) != int(prune_count):
            logger.warning(
                "Ignoring moe_pba resume history at %s due to prune-count mismatch on layer %d: expected %d, found %d.",
                output_dir,
                layer_idx,
                int(prune_count),
                len(selected),
            )
            return [], {}

        normalized_entry = dict(entry)
        normalized_entry["layer"] = layer_idx
        normalized_entry["selected"] = selected
        normalized_entry["input_conditioning"] = MOE_PBA_INPUT_CONDITIONING
        normalized_entry["include_shared_experts"] = True
        restored_history.append(normalized_entry)
        committed[layer_idx] = selected

    return restored_history, committed
