from __future__ import annotations

import logging
import math
import time
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
from moe_pba.common.modeling import build_plan_summary, max_prunable_per_layer, resolve_prune_count
from moe_pba.common.progress import (
    persist_search_history,
    progress_log_every,
    restore_completed_layer_history,
)
from moe_pba.common.scoring import (
    release_replay_cache_collection_hot,
    release_replay_cache_hot,
)
from moe_pba.common.types import MethodResult, merge_prune_maps, to_jsonable_prune_map
from moe_pba.methods.moe_pba import (
    _precompute_full_reference_caches,
    _prepare_prefix_conditioned_replay_cache,
    _score_candidate,
)

logger = logging.getLogger(__name__)


def _total_candidates_for_layer(available_count: int, prune_count: int) -> int:
    available_count = max(0, int(available_count))
    prune_count = max(0, int(prune_count))
    if available_count == 0 or prune_count == 0:
        return 0
    total = 0
    for selected_count in range(min(prune_count, available_count)):
        total += max(0, available_count - selected_count)
    return total


def run(model, calibration_context, layers, moe_pba_args, *, output_dir=None):
    max_prunable = max_prunable_per_layer(layers)
    prune_count = resolve_prune_count(layers, moe_pba_args)
    score_key = MOE_PBA_OBJECTIVE

    history, committed = restore_completed_layer_history(
        output_dir,
        layers=layers,
        expected_objective=MOE_PBA_OBJECTIVE,
        prune_count=prune_count,
    )
    resumed_layer_count = len(history)
    resumed_pruned_total = sum(len(indices) for indices in committed.values())

    if prune_count == 0:
        pruned = {str(int(layer.layer_idx)): [] for layer in layers}
        return MethodResult(
            pruned_experts=pruned,
            plan_summary=build_plan_summary(layers, pruned),
            score_summary={
                "objective": MOE_PBA_OBJECTIVE,
                "metric": MOE_PBA_METRIC,
                "score_reduction": MOE_PBA_SCORE_REDUCTION,
                "within_layer_ordering": "greedy",
                "reference": MOE_PBA_REFERENCE,
                "cross_layer_mode": MOE_PBA_CROSS_LAYER_MODE,
                "input_conditioning": MOE_PBA_INPUT_CONDITIONING,
                "include_shared_experts": MOE_PBA_INCLUDE_SHARED_EXPERTS,
                "calibration_fingerprint": calibration_context.fingerprint,
            },
        )

    if resumed_layer_count > 0:
        next_layer = None if resumed_layer_count >= len(layers) else int(layers[resumed_layer_count].layer_idx)
        logger.info(
            "Resuming MoE-PBA greedy search: completed_layers=%d/%d resumed_pruned_total=%d next_layer=%s",
            resumed_layer_count,
            len(layers),
            resumed_pruned_total,
            "none" if next_layer is None else str(next_layer),
        )
    else:
        logger.info(
            "Starting MoE-PBA greedy search: layers=%d prune_count_per_layer=%d calibration_batches=%d "
            "include_shared_experts=%s input_conditioning=%s",
            len(layers),
            prune_count,
            len(calibration_context.batches),
            MOE_PBA_INCLUDE_SHARED_EXPERTS,
            MOE_PBA_INPUT_CONDITIONING,
        )
    persist_search_history(output_dir, history)

    remaining_layers = layers[resumed_layer_count:]
    full_reference_caches = _precompute_full_reference_caches(
        model,
        calibration_context.batches,
        layers,
        start_position=resumed_layer_count,
    )

    for layer_pos, layer in enumerate(remaining_layers, start=resumed_layer_count + 1):
        if prune_count <= 0:
            continue
        if prune_count > max_prunable[layer.layer_idx]:
            raise ValueError(
                f"Requested prune_count={prune_count} exceeds max prunable experts "
                f"for layer {layer.layer_idx} (max={max_prunable[layer.layer_idx]})."
            )

        layer_started_at = time.time()
        layer_idx = int(layer.layer_idx)
        replay_cache = _prepare_prefix_conditioned_replay_cache(
            model,
            calibration_context.batches,
            layer_idx=layer_idx,
            committed=committed,
            full_reference_cache=full_reference_caches[layer_idx],
        )
        already_pruned = set(committed.get(layer_idx, []))
        available = [expert_idx for expert_idx in range(layer.num_experts) if expert_idx not in already_pruned]
        if prune_count > len(available):
            raise ValueError(
                f"Requested prune_count={prune_count} exceeds remaining experts for layer {layer_idx} "
                f"(remaining={len(available)})."
            )

        candidate_total = _total_candidates_for_layer(len(available), prune_count)
        log_every = progress_log_every(candidate_total)
        layer_history: dict[str, Any] = {
            "layer": layer_idx,
            "status": "running",
            "objective": MOE_PBA_OBJECTIVE,
            "metric": MOE_PBA_METRIC,
            "score_reduction": MOE_PBA_SCORE_REDUCTION,
            "candidate_count": candidate_total,
            "candidates_scored": 0,
            "replay_samples": len(replay_cache),
            "selected": committed.get(layer_idx, []),
            "started_unix_time": layer_started_at,
            "reference": MOE_PBA_REFERENCE,
            "cross_layer_mode": MOE_PBA_CROSS_LAYER_MODE,
            "input_conditioning": MOE_PBA_INPUT_CONDITIONING,
            "include_shared_experts": MOE_PBA_INCLUDE_SHARED_EXPERTS,
            "within_layer_ordering": "greedy",
            "greedy_steps": [],
        }
        history.append(layer_history)
        persist_search_history(output_dir, history)

        logger.info(
            "[%d/%d] Layer %d: replay_samples=%d candidates=%d committed_total=%d",
            layer_pos,
            len(layers),
            layer_idx,
            len(replay_cache),
            candidate_total,
            sum(len(indices) for indices in committed.values()),
        )

        selected: list[int] = []
        candidate_scores: list[dict[str, Any]] = []
        best_so_far_score: float | None = None
        best_so_far_selected: list[int] | None = None

        while len(selected) < prune_count:
            remaining = [expert_idx for expert_idx in available if expert_idx not in selected]
            stage_best_score = None
            stage_best_expert = None
            stage_scores = []

            for expert_idx in remaining:
                current_selected = sorted(selected + [int(expert_idx)])
                current_layer_mask = {layer_idx: current_selected}
                mask = merge_prune_maps(committed, current_layer_mask)
                score = _score_candidate(
                    model,
                    layer_idx,
                    replay_cache,
                    mask,
                )
                if not math.isfinite(score):
                    raise RuntimeError(
                        "MoE-PBA-G produced a non-finite score for "
                        f"layer {layer_idx}, expert {expert_idx}: {score}."
                    )
                candidate_scores.append(
                    {
                        "step": len(selected) + 1,
                        "candidate": int(expert_idx),
                        "selected_before": sorted(selected),
                        "selected_after": current_selected,
                        score_key: float(score),
                    }
                )
                stage_scores.append((float(score), int(expert_idx), current_selected))

                if stage_best_score is None or score < stage_best_score:
                    stage_best_score = float(score)
                    stage_best_expert = int(expert_idx)
                if best_so_far_score is None or score < best_so_far_score:
                    best_so_far_score = float(score)
                    best_so_far_selected = current_selected

                layer_history["candidates_scored"] = len(candidate_scores)
                layer_history["elapsed_seconds"] = max(0.0, time.time() - layer_started_at)
                layer_history["best_so_far"] = (
                    None if best_so_far_selected is None else {"selected": best_so_far_selected, score_key: best_so_far_score}
                )
                if candidate_total == 0 or len(candidate_scores) == candidate_total or (len(candidate_scores) % log_every) == 0:
                    logger.info(
                        "[%d/%d] Layer %d progress %d/%d best_selected=%s best_%s=%.6f elapsed=%.1fs",
                        layer_pos,
                        len(layers),
                        layer_idx,
                        len(candidate_scores),
                        candidate_total,
                        "none" if best_so_far_selected is None else str(best_so_far_selected),
                        score_key,
                        float(best_so_far_score if best_so_far_score is not None else 0.0),
                        layer_history["elapsed_seconds"],
                    )
                    persist_search_history(output_dir, history)

            stage_scores.sort(key=lambda item: (item[0], item[1]))
            if stage_best_expert is None or stage_best_score is None:
                raise RuntimeError(f"MoE-PBA greedy could not select the next expert for layer {layer_idx}.")
            selected = sorted(selected + [stage_best_expert])
            layer_history["selected"] = selected
            layer_history["greedy_steps"].append(
                {
                    "step": len(selected),
                    "mode": "scored_greedy",
                    "selected": selected,
                    "best_candidate": int(stage_best_expert),
                    score_key: float(stage_best_score),
                    "candidates": [
                        {
                            "candidate": expert_idx,
                            "selected_after": selected_after,
                            score_key: score,
                        }
                        for score, expert_idx, selected_after in stage_scores
                    ],
                }
            )
            persist_search_history(output_dir, history)
            logger.info(
                "[%d/%d] Layer %d selected step=%d expert=%d selected=%s %s=%.6f",
                layer_pos,
                len(layers),
                layer_idx,
                len(selected),
                stage_best_expert,
                selected,
                score_key,
                stage_best_score,
            )

        committed[layer_idx] = selected
        layer_elapsed = max(0.0, time.time() - layer_started_at)
        layer_history.update(
            {
                "status": "completed",
                "completed_unix_time": time.time(),
                "elapsed_seconds": layer_elapsed,
                "candidate_scores": candidate_scores,
                "selected": committed[layer_idx],
            }
        )
        persist_search_history(output_dir, history)
        logger.info(
            "[%d/%d] Layer %d complete: selected=%s best=%s elapsed=%.1fs",
            layer_pos,
            len(layers),
            layer_idx,
            committed[layer_idx],
            candidate_scores[: min(3, len(candidate_scores))],
            layer_elapsed,
        )
        release_replay_cache_hot(replay_cache)
        release_replay_cache_collection_hot(full_reference_caches)

    pruned = to_jsonable_prune_map(committed)
    plan_summary = build_plan_summary(layers, pruned)
    release_replay_cache_collection_hot(full_reference_caches)
    logger.info(
        "Finished MoE-PBA greedy search: total_pruned=%d",
        sum(len(indices) for indices in committed.values()),
    )
    return MethodResult(
        pruned_experts=pruned,
        plan_summary=plan_summary,
        score_summary={
            "objective": MOE_PBA_OBJECTIVE,
            "metric": MOE_PBA_METRIC,
            "score_reduction": MOE_PBA_SCORE_REDUCTION,
            "within_layer_ordering": "greedy",
            "reference": MOE_PBA_REFERENCE,
            "cross_layer_mode": MOE_PBA_CROSS_LAYER_MODE,
            "input_conditioning": MOE_PBA_INPUT_CONDITIONING,
            "include_shared_experts": MOE_PBA_INCLUDE_SHARED_EXPERTS,
            "calibration_fingerprint": calibration_context.fingerprint,
        },
        history=history,
    )
