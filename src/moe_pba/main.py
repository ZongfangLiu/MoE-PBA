from __future__ import annotations

import gc
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from moe_pba.args import (
    DatasetArgs,
    GreedySearchArgs,
    MOE_PBA_CROSS_LAYER_MODE,
    MOE_PBA_METRIC,
    MOE_PBA_REFERENCE,
    ModelArgs,
    ObserverArgs,
    RunArgs,
)
from moe_pba.runtime_cleanup import release_runtime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_RECURSION_LIMIT = 3000


def _configure_recursion_limit() -> None:
    if sys.getrecursionlimit() < _RECURSION_LIMIT:
        sys.setrecursionlimit(_RECURSION_LIMIT)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _install_run_log_handler(output_dir: Path) -> Path:
    log_path = output_dir / "run.log"
    root_logger = logging.getLogger()
    resolved_path = log_path.resolve()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == resolved_path:
                    return log_path
            except Exception:
                continue

    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    logger.info("Attached per-run log file at %s", log_path)
    return log_path


def _configure_moe_pba_runtime(moe_pba_args: GreedySearchArgs) -> None:
    if not moe_pba_args.gpus:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != moe_pba_args.gpus:
        logger.warning(
            "moe_pba --gpus=%s requested but CUDA_VISIBLE_DEVICES=%s. "
            "Prefer launching via `python -m moe_pba --gpus ...` so bootstrap happens before torch import.",
            moe_pba_args.gpus,
            visible,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = moe_pba_args.gpus
        visible = moe_pba_args.gpus

    logger.info(
        "moe_pba runtime configured: gpus=%s visible_cuda_devices=%s",
        moe_pba_args.gpus,
        visible,
    )


def _build_run_name(method: str, args: GreedySearchArgs, calibration_tag: str, seed: int) -> str:
    from moe_pba.common.modeling import build_sparsity_tag

    parts = [
        f"fit_{method}",
        f"seed_{seed}",
        build_sparsity_tag(args),
        f"samples{int(args.calib_samples)}",
        f"metric_{MOE_PBA_METRIC}",
    ]

    parts.append(f"ref{MOE_PBA_REFERENCE}")
    parts.append(f"crosslayer{MOE_PBA_CROSS_LAYER_MODE}")

    parts.append(f"calib_{calibration_tag}")
    return "-".join(parts)


def main() -> None:
    _configure_recursion_limit()
    parser = HfArgumentParser(
        (
            RunArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            GreedySearchArgs,
        )
    )
    run_args, ds_args, obs_args, model_args, moe_pba_args = parser.parse_args_into_dataclasses()
    _configure_moe_pba_runtime(moe_pba_args)

    from moe_pba.common.calibration import prepare_calibration_context, resolve_calibration_tag
    from moe_pba.common.run_identity import build_run_request, write_run_request
    from moe_pba.common.modeling import (
        build_output_dir,
        completed_output_exists,
        discover_moe_layers,
        resolve_prune_count,
        validate_method_support,
    )
    from moe_pba.common.save_and_eval import save_method_result
    from moe_pba.methods import METHOD_REGISTRY

    _set_seed(run_args.seed)
    model_name = model_args.model_name
    calibration_tag = resolve_calibration_tag(moe_pba_args)
    run_name = _build_run_name(moe_pba_args.method, moe_pba_args, calibration_tag, run_args.seed)
    output_dir = build_output_dir(
        model_name,
        moe_pba_args.method,
        run_name,
        moe_pba_args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = _install_run_log_handler(output_dir)
    logger.info("Starting moe_pba run: output_dir=%s", output_dir)
    logger.info("Run log path: %s", run_log_path)
    run_request = build_run_request(
        model_name=model_name,
        method=moe_pba_args.method,
        run_args=run_args,
        ds_args=ds_args,
        obs_args=obs_args,
        model_args=model_args,
        moe_pba_args=moe_pba_args,
    )
    write_run_request(output_dir, run_request)

    local_files_only = bool(model_args.local_files_only)
    trust_remote_code = bool(model_args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    logger.info("Loading model for moe_pba: model_name=%s device_map=auto torch_dtype=auto", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    validate_method_support(model)
    layers = discover_moe_layers(model)
    resolved_prune_count = resolve_prune_count(layers, moe_pba_args)
    validate_method_support(
        model,
        layers=layers,
        prune_count=resolved_prune_count,
    )
    logger.info(
        "Resolved moe_pba sparsity: ratio=%s prune_count=%d",
        getattr(moe_pba_args, "sparsity_ratio", None),
        resolved_prune_count,
    )

    calibration_context = prepare_calibration_context(
        tokenizer,
        moe_pba_args,
        ds_args,
        obs_args,
        model=model,
    )
    run_name = _build_run_name(
        moe_pba_args.method,
        moe_pba_args,
        calibration_context.dataset_tag,
        run_args.seed,
    )
    output_dir = build_output_dir(
        model_name,
        moe_pba_args.method,
        run_name,
        moe_pba_args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_request(
        output_dir,
        run_request,
        calibration_fingerprint=calibration_context.fingerprint,
    )
    if completed_output_exists(output_dir):
        logger.info("Detected completed moe_pba output at %s; skipping search/materialization.", output_dir)
        logger.info("MOE_PBA_OUTPUT_DIR=%s", output_dir)
        print(f"MOE_PBA_OUTPUT_DIR={output_dir}")
        release_runtime(model, logger=logger)
        return
    logger.info(
        "Running moe_pba method=%s model=%s layers=%d batches=%d output=%s",
        moe_pba_args.method,
        model_name,
        len(layers),
        calibration_context.batch_count,
        output_dir,
    )

    runner = METHOD_REGISTRY[moe_pba_args.method]
    search_started_at = time.time()
    method_result = runner(
        model=model,
        calibration_context=calibration_context,
        layers=layers,
        moe_pba_args=moe_pba_args,
        output_dir=output_dir,
    )
    search_finished_at = time.time()
    logger.info(
        "Completed moe_pba method=%s in %.2f seconds (%.4f hours).",
        moe_pba_args.method,
        max(0.0, search_finished_at - search_started_at),
        max(0.0, search_finished_at - search_started_at) / 3600.0,
    )

    save_method_result(
        model=model,
        tokenizer=tokenizer,
        output_dir=Path(output_dir),
        method_result=method_result,
        method_name=moe_pba_args.method,
        model_args=model_args,
        ds_args=ds_args,
        obs_args=obs_args,
        run_args=run_args,
        moe_pba_args=moe_pba_args,
        search_started_at=search_started_at,
        search_finished_at=search_finished_at,
        run_request=run_request,
    )
    logger.info("MOE_PBA_OUTPUT_DIR=%s", output_dir)
    print(f"MOE_PBA_OUTPUT_DIR={output_dir}")

    release_runtime(model, logger=logger)
    model = None
    tokenizer = None
    calibration_context = None
    method_result = None
    layers = None
    gc.collect()
    release_runtime(logger=logger)


if __name__ == "__main__":
    main()
