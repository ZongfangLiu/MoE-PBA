from __future__ import annotations

import logging
import math
import time

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
    build_prefix_conditioned_replay_cache,
    capture_all_layer_replay_caches,
    release_replay_cache_collection_hot,
    release_replay_cache_hot,
    score_layer_replay_squared_l2,
)
from moe_pba.common.types import MethodResult, merge_prune_maps, to_jsonable_prune_map

logger = logging.getLogger(__name__)


def _score_candidate(
    model,
    layer_idx: int,
    replay_cache: list[dict[str, object]],
    mask: dict[int, list[int]],
) -> float:
    return float(
        score_layer_replay_squared_l2(
            model,
            layer_idx,
            replay_cache,
            mask,
        )
    )


def _precompute_full_reference_caches(
    model,
    calibration_batches,
    layers,
    *,
    start_position: int,
) -> dict[int, list[dict[str, object]]]:
    remaining_layer_indices = [int(layer.layer_idx) for layer in layers[int(start_position) :]]
    if not remaining_layer_indices:
        return {}
    logger.info(
        "Precomputing full-model reference replay caches for remaining_layers=%d",
        len(remaining_layer_indices),
    )
    return capture_all_layer_replay_caches(
        model,
        calibration_batches,
        remaining_layer_indices,
        capture_inputs=False,
    )


def _prepare_prefix_conditioned_replay_cache(
    model,
    calibration_batches,
    *,
    layer_idx: int,
    committed: dict[int, list[int]],
    full_reference_cache: list[dict[str, object]],
) -> list[dict[str, object]]:
    prefix_input_cache = capture_all_layer_replay_caches(
        model,
        calibration_batches,
        [int(layer_idx)],
        pruned_experts_info=committed,
    )[int(layer_idx)]
    return build_prefix_conditioned_replay_cache(
        full_reference_cache,
        prefix_input_cache,
    )


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
                "within_layer_ordering": "leave_one_out",
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
            "Resuming MoE-PBA search: completed_layers=%d/%d resumed_pruned_total=%d next_layer=%s",
            resumed_layer_count,
            len(layers),
            resumed_pruned_total,
            "none" if next_layer is None else str(next_layer),
        )
    else:
        logger.info(
            "Starting MoE-PBA search: layers=%d prune_count_per_layer=%d calibration_batches=%d "
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
        candidate_total = max(0, layer.num_experts - len(already_pruned))
        log_every = progress_log_every(candidate_total)
        layer_history = {
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
            "within_layer_ordering": "leave_one_out",
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

        best_score = None
        best_expert = None
        candidate_scores = []
        for expert_idx in range(layer.num_experts):
            if expert_idx in already_pruned:
                continue
            current_layer_mask = {layer_idx: [expert_idx]}
            mask = merge_prune_maps(committed, current_layer_mask)
            score = _score_candidate(
                model,
                layer_idx,
                replay_cache,
                mask,
            )
            if not math.isfinite(score):
                raise RuntimeError(
                    f"MoE-PBA produced a non-finite score for layer {layer_idx}, expert {expert_idx}: {score}."
                )
            candidate_scores.append((float(score), int(expert_idx)))
            if best_score is None or score < best_score:
                best_score = float(score)
                best_expert = int(expert_idx)

            layer_history["candidates_scored"] = len(candidate_scores)
            layer_history["elapsed_seconds"] = max(0.0, time.time() - layer_started_at)
            layer_history["best_so_far"] = None if best_expert is None else {"expert": best_expert, score_key: float(best_score)}
            if len(candidate_scores) == candidate_total or (len(candidate_scores) % log_every) == 0:
                logger.info(
                    "[%d/%d] Layer %d progress %d/%d best_expert=%s best_%s=%.6f elapsed=%.1fs",
                    layer_pos,
                    len(layers),
                    layer_idx,
                    len(candidate_scores),
                    candidate_total,
                    "none" if best_expert is None else str(best_expert),
                    score_key,
                    float(best_score if best_score is not None else 0.0),
                    layer_history["elapsed_seconds"],
                )
                persist_search_history(output_dir, history)

        candidate_scores.sort(key=lambda item: (item[0], item[1]))
        committed[layer_idx] = sorted(
            set(committed.get(layer_idx, [])) | {expert_idx for _, expert_idx in candidate_scores[:prune_count]}
        )
        layer_elapsed = max(0.0, time.time() - layer_started_at)
        layer_history.update(
            {
                "status": "completed",
                "completed_unix_time": time.time(),
                "elapsed_seconds": layer_elapsed,
                "candidate_scores": [
                    {"expert": int(expert_idx), score_key: float(score)}
                    for score, expert_idx in candidate_scores
                ],
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
            [
                {"expert": int(expert_idx), score_key: float(score)}
                for score, expert_idx in candidate_scores[: min(3, len(candidate_scores))]
            ],
            layer_elapsed,
        )
        release_replay_cache_hot(replay_cache)
        release_replay_cache_collection_hot(full_reference_caches)

    pruned = to_jsonable_prune_map(committed)
    plan_summary = build_plan_summary(layers, pruned)
    release_replay_cache_collection_hot(full_reference_caches)
    logger.info("Finished MoE-PBA search: total_pruned=%d", sum(len(indices) for indices in committed.values()))
    return MethodResult(
        pruned_experts=pruned,
        plan_summary=plan_summary,
        score_summary={
            "objective": MOE_PBA_OBJECTIVE,
            "metric": MOE_PBA_METRIC,
            "score_reduction": MOE_PBA_SCORE_REDUCTION,
            "within_layer_ordering": "leave_one_out",
            "reference": MOE_PBA_REFERENCE,
            "cross_layer_mode": MOE_PBA_CROSS_LAYER_MODE,
            "input_conditioning": MOE_PBA_INPUT_CONDITIONING,
            "include_shared_experts": MOE_PBA_INCLUDE_SHARED_EXPERTS,
            "calibration_fingerprint": calibration_context.fingerprint,
        },
        history=history,
    )
