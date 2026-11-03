from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urlparse


PAPER_TASKS: dict[str, tuple[str, ...]] = {
    "lm-eval": ("mmlu", "openbookqa"),
    "evalscope": ("gsm8k", "math_500"),
    "evalplus": ("humaneval", "mbpp"),
    "livecodebench": ("livecodebench",),
}
FAMILY_ORDER = tuple(PAPER_TASKS)
TASK_TO_FAMILY = {
    task: family
    for family, tasks in PAPER_TASKS.items()
    for task in tasks
}
TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "paper": tuple(TASK_TO_FAMILY),
    "all": tuple(TASK_TO_FAMILY),
    **PAPER_TASKS,
    "lm_eval": PAPER_TASKS["lm-eval"],
    "lcb": PAPER_TASKS["livecodebench"],
}
DEFAULT_SEED = 42
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_PARALLEL_REQUESTS = 32
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_LCB_START_DATE = "2025-01-01"
DEFAULT_LCB_END_DATE = "2025-07-31"


@dataclass(frozen=True)
class EvaluationPlan:
    model: str
    output_dir: Path
    tasks: tuple[str, ...]
    backend: str
    base_url: str | None
    seed: int
    context_length: int
    max_new_tokens: int
    trust_remote_code: bool
    tensor_parallel_size: int
    lcb_start_date: str
    lcb_end_date: str
    parallel_requests: int = DEFAULT_PARALLEL_REQUESTS
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE

    @property
    def families(self) -> tuple[str, ...]:
        selected = set(self.tasks)
        return tuple(
            family
            for family in FAMILY_ORDER
            if selected.intersection(PAPER_TASKS[family])
        )


def _normalize_tasks(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return tuple(TASK_TO_FAMILY)
    raw_values = (values,) if isinstance(values, str) else values
    tokens = [
        token.strip().lower()
        for value in raw_values
        for token in value.split(",")
        if token.strip()
    ]
    if not tokens:
        raise ValueError("--tasks must contain at least one task")

    tasks: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        normalized = token.replace("math-500", "math_500")
        if normalized in TASK_TO_FAMILY:
            tasks.append(normalized)
        elif normalized in TASK_GROUPS:
            tasks.extend(TASK_GROUPS[normalized])
        else:
            unknown.append(token)
    if unknown:
        raise ValueError(
            "unsupported task(s): " + ", ".join(unknown) + "; choose from " + ", ".join(TASK_TO_FAMILY)
        )
    # Preserve user order while ensuring a benchmark is not run twice.
    return tuple(dict.fromkeys(tasks))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _normalize_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized.removesuffix("/chat/completions")
    elif normalized.endswith("/completions"):
        normalized = normalized.removesuffix("/completions")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an http(s) URL")
    return normalized


def _resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "vllm"
    return requested


def _parse_lcb_date(value: str, option: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{option} must be a valid date in YYYY-MM-DD format") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ValueError(f"{option} must be a valid date in YYYY-MM-DD format")
    return parsed


def _validate_lcb_date_range(start_date: str, end_date: str) -> None:
    start = _parse_lcb_date(start_date, "--lcb-start-date")
    end = _parse_lcb_date(end_date, "--lcb-end-date")
    if start > end:
        raise ValueError("--lcb-start-date must be on or before --lcb-end-date")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run zero-shot benchmark evaluation.",
        epilog="The vLLM backend requires an existing OpenAI-compatible server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for manifests and benchmark outputs.")
    parser.add_argument(
        "--tasks",
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Space- or comma-separated tasks/families. Default is the full paper suite: "
            "mmlu,openbookqa,gsm8k,math_500,humaneval,mbpp,livecodebench."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "hf", "vllm"),
        default="auto",
        help="auto defaults to vllm for every benchmark family; use hf explicitly for checkpoint evaluation.",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible vLLM URL (for example http://127.0.0.1:8000/v1); required by vllm runs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--context-length", type=_positive_int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument("--parallel-requests", type=_positive_int, default=DEFAULT_PARALLEL_REQUESTS)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--lcb-start-date", default=DEFAULT_LCB_START_DATE)
    parser.add_argument("--lcb-end-date", default=DEFAULT_LCB_END_DATE)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow benchmark loaders to execute model repository code (default: disabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write evaluation_config.json without importing or running optional packages.",
    )
    return parser


def _plan_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> EvaluationPlan:
    try:
        tasks = _normalize_tasks(args.tasks)
        base_url = _normalize_base_url(args.base_url)
        _validate_lcb_date_range(args.lcb_start_date, args.lcb_end_date)
    except ValueError as exc:
        parser.error(str(exc))
    backend = _resolve_backend(args.backend)
    if backend == "hf" and "livecodebench" in tasks:
        parser.error("LiveCodeBench requires --backend vllm; it has no supported Hugging Face evaluation path")
    if backend == "vllm" and not base_url:
        parser.error("--base-url is required when the resolved backend is vllm")
    if args.context_length <= args.max_new_tokens:
        parser.error("--context-length must be greater than --max-new-tokens")

    return EvaluationPlan(
        model=args.model,
        output_dir=args.output_dir,
        tasks=tasks,
        backend=backend,
        base_url=base_url,
        seed=args.seed,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        lcb_start_date=args.lcb_start_date,
        lcb_end_date=args.lcb_end_date,
        parallel_requests=args.parallel_requests,
        eval_batch_size=args.eval_batch_size,
    )


def _family_tasks(plan: EvaluationPlan, family: str) -> tuple[str, ...]:
    selected = set(plan.tasks)
    return tuple(task for task in PAPER_TASKS[family] if task in selected)


def _worker_base(plan: EvaluationPlan, family: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "moe_pba.eval",
        "_worker",
        "--family",
        family,
        "--model",
        plan.model,
        "--output-dir",
        str(plan.output_dir),
        "--tasks",
        ",".join(_family_tasks(plan, family)),
        "--backend",
        plan.backend,
        "--seed",
        str(plan.seed),
        "--context-length",
        str(plan.context_length),
        "--max-new-tokens",
        str(plan.max_new_tokens),
        "--tensor-parallel-size",
        str(plan.tensor_parallel_size),
        "--parallel-requests",
        str(plan.parallel_requests),
        "--eval-batch-size",
        str(plan.eval_batch_size),
        "--lcb-start-date",
        plan.lcb_start_date,
        "--lcb-end-date",
        plan.lcb_end_date,
    ]
    if plan.base_url:
        command.extend(("--base-url", plan.base_url))
    if plan.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _described_commands(plan: EvaluationPlan) -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "tasks": list(_family_tasks(plan, family)),
            "command": _worker_base(plan, family),
        }
        for family in plan.families
    ]


def _vllm_server_command(plan: EvaluationPlan) -> list[str] | None:
    if plan.backend != "vllm":
        return None
    command = [
        "vllm",
        "serve",
        plan.model,
        "--max-model-len",
        str(plan.context_length),
        "--tensor-parallel-size",
        str(plan.tensor_parallel_size),
        "--seed",
        str(plan.seed),
    ]
    if plan.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _config_payload(plan: EvaluationPlan, *, dry_run: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "preset": "paper",
        "model": plan.model,
        "output_dir": str(plan.output_dir),
        "tasks": list(plan.tasks),
        "families": list(plan.families),
        "backend": plan.backend,
        "base_url": plan.base_url,
        "zero_shot": True,
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": plan.seed,
            "context_length": plan.context_length,
            "max_new_tokens": plan.max_new_tokens,
        },
        "trust_remote_code": plan.trust_remote_code,
        "tensor_parallel_size": plan.tensor_parallel_size,
        "parallel_requests": plan.parallel_requests,
        "eval_batch_size": plan.eval_batch_size,
        "livecodebench": {
            "start_date": plan.lcb_start_date,
            "end_date": plan.lcb_end_date,
        },
        "strict_failures": True,
        "dry_run": dry_run,
        "server_command": _vllm_server_command(plan),
        "commands": _described_commands(plan),
    }


def _write_json(path: Path, payload: dict[str, Any], *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not atomic:
        path.write_text(serialized, encoding="utf-8")
        return

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _request_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the persisted benchmark request, excluding dry-run execution mode."""

    request = dict(config)
    request.pop("dry_run", None)
    return request


def _ensure_output_config_compatible(config_path: Path, config: dict[str, Any]) -> None:
    if not config_path.exists():
        return
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Output directory contains an unreadable {config_path.name}; use a new --output-dir"
        ) from exc
    if not isinstance(existing, dict) or _request_config(existing) != _request_config(config):
        raise RuntimeError(
            f"Output directory already contains {config_path.name} for a different evaluation request; "
            "use a new --output-dir"
        )


def _run_driver(plan: EvaluationPlan, *, dry_run: bool) -> int:
    config = _config_payload(plan, dry_run=dry_run)
    config_path = plan.output_dir / "evaluation_config.json"
    status_path = plan.output_dir / "evaluation_status.json"
    _ensure_output_config_compatible(config_path, config)

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    # Mark the new attempt incomplete before writing its manifest.
    if status_path.exists():
        status_path.unlink()
    _write_json(config_path, config)

    if config["server_command"]:
        print("VLLM_SERVER " + subprocess.list2cmdline(config["server_command"]))
    for item in config["commands"]:
        print("COMMAND " + subprocess.list2cmdline(item["command"]))
    if dry_run:
        return 0

    for item in config["commands"]:
        subprocess.run(item["command"], check=True)

    _write_json(
        status_path,
        {
            "schema_version": 1,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": plan.model,
            "backend": plan.backend,
            "tasks": list(plan.tasks),
        },
        atomic=True,
    )
    return 0


def _require_module(module: str, install_hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"{module} is required for this benchmark. {install_hint}")


def _run_lm_eval(plan: EvaluationPlan) -> None:
    _require_module("lm_eval", "Install a compatible lm-eval release with its API extra.")
    tasks = _family_tasks(plan, "lm-eval")
    output = plan.output_dir / "lm_eval"
    if plan.backend == "hf":
        model_type = "hf"
        model_args = {
            "pretrained": plan.model,
            "dtype": "auto",
            "max_length": plan.context_length,
            "trust_remote_code": plan.trust_remote_code,
        }
    else:
        model_type = "local-completions"
        model_args = {
            "model": plan.model,
            "base_url": f"{plan.base_url}/completions",
            "tokenizer": plan.model,
            "tokenizer_backend": "huggingface",
            "tokenized_requests": False,
            "max_length": plan.context_length,
            "max_gen_toks": plan.max_new_tokens,
            "num_concurrent": plan.parallel_requests,
            "timeout": 1200,
            "max_retries": 10,
            "seed": plan.seed,
            "trust_remote_code": plan.trust_remote_code,
        }
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        model_type,
        "--model_args",
        ",".join(f"{key}={value}" for key, value in model_args.items()),
        "--tasks",
        *tasks,
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto" if plan.backend == "hf" else "1",
        "--gen_kwargs",
        f"do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks={plan.max_new_tokens}",
        "--seed",
        str(plan.seed),
        "--output_path",
        str(output),
        "--log_samples",
    ]
    subprocess.run(command, check=True)


@contextmanager
def _preserve_evalscope_greedy_config():
    """Keep EvalScope from rewriting an explicit greedy temperature of zero."""

    try:
        from evalscope.utils import model_utils
    except Exception:
        yield
        return

    original = getattr(model_utils, "fix_do_sample_warning", None)
    if original is None:
        yield
        return

    def preserve(generation_config: Any) -> None:
        if getattr(generation_config, "temperature", None) == 0:
            generation_config.do_sample = False

    model_utils.fix_do_sample_warning = preserve
    try:
        yield
    finally:
        model_utils.fix_do_sample_warning = original


def _run_evalscope(plan: EvaluationPlan) -> None:
    _require_module("evalscope", "Install a compatible EvalScope release.")
    from evalscope.config import TaskConfig
    from evalscope.constants import EvalType
    from evalscope.run import run_task

    tasks = _family_tasks(plan, "evalscope")
    generation: dict[str, Any] = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": plan.max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    model_args: dict[str, Any] = {}
    if plan.backend == "hf":
        model_args.update(device_map="auto", torch_dtype="auto", model_max_length=plan.context_length)
        generation["max_length"] = plan.context_length
        eval_type = EvalType.CHECKPOINT
        api_url = None
    else:
        generation["max_tokens"] = plan.max_new_tokens
        eval_type = EvalType.SERVICE
        api_url = plan.base_url

    task_config = TaskConfig(
        model=plan.model,
        model_args=model_args,
        generation_config=generation,
        datasets=list(tasks),
        dataset_args={task: {"few_shot_num": 0} for task in tasks},
        api_url=api_url,
        api_key="EMPTY",
        timeout=3600,
        work_dir=str(plan.output_dir / "evalscope"),
        eval_batch_size=plan.eval_batch_size,
        eval_type=eval_type,
        seed=plan.seed,
    )
    with _preserve_evalscope_greedy_config():
        run_task(task_config)


def _run_evalplus(plan: EvaluationPlan) -> None:
    _require_module("evalplus", "Install a compatible EvalPlus release.")
    from evalplus import codegen as evalplus_codegen
    from evalplus import config as evalplus_config
    from evalplus import evaluate as evalplus_evaluate
    from evalplus.provider import base as evalplus_base

    evalplus_config.MAX_NEW_TOKENS = plan.max_new_tokens
    evalplus_base.MAX_NEW_TOKENS = plan.max_new_tokens
    original_decoder_init = evalplus_base.DecoderBase.__init__
    original_make_model = evalplus_codegen.make_model
    model_name_lower = plan.model.lower()
    enable_thinking = "qwen" not in model_name_lower and "glm-4.5" not in model_name_lower
    model_alias = f"moe-pba-{hashlib.sha256(plan.model.encode('utf-8')).hexdigest()[:16]}"

    def decoder_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("max_new_tokens", plan.max_new_tokens)
        original_decoder_init(self, *args, **kwargs)

    def make_model(*args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if args_list and args_list[0] == model_alias:
            args_list[0] = plan.model
        if kwargs.get("model") == model_alias:
            kwargs["model"] = plan.model
        return original_make_model(*args_list, **kwargs)

    evalplus_base.DecoderBase.__init__ = decoder_init
    evalplus_codegen.make_model = make_model

    root = plan.output_dir / "evalplus"
    backend = "hf" if plan.backend == "hf" else "openai"
    try:
        for task in _family_tasks(plan, "evalplus"):
            kwargs: dict[str, Any] = {
                "model": model_alias,
                "root": str(root),
                "dataset": task,
                "backend": backend,
                "bs": 1,
                "n_samples": 1,
                "greedy": True,
                "temperature": 0.0,
                "trust_remote_code": plan.trust_remote_code,
                # Disable thinking mode for Qwen/GLM and preserve other model defaults.
                "enable_thinking": enable_thinking,
                "parallel_tasks": plan.parallel_requests,
                "output_file": str(root / f"{task}_results.json"),
            }
            if plan.backend == "hf":
                kwargs.update(device_map="auto", attn_implementation="sdpa", dtype="bfloat16")
            else:
                kwargs["base_url"] = plan.base_url
            evalplus_evaluate.evaluate(**kwargs)
    finally:
        evalplus_base.DecoderBase.__init__ = original_decoder_init
        evalplus_codegen.make_model = original_make_model


def _run_livecodebench(plan: EvaluationPlan) -> None:
    if plan.backend != "vllm" or not plan.base_url:
        raise RuntimeError("LiveCodeBench requires an OpenAI-compatible vLLM server")
    _require_module(
        "lcb_runner",
        "Run `bash scripts/install.sh` from the MoE-PBA source checkout.",
    )

    from lcb_runner.lm_styles import LMStyle, LanguageModel, LanguageModelStore
    from lcb_runner.runner.main import get_args_dict, main as lcb_main

    model_key = plan.model
    if model_key not in LanguageModelStore:
        LanguageModelStore[model_key] = LanguageModel(
            model_key,
            Path(model_key).name,
            LMStyle.ReapBase,
            datetime.now(timezone.utc).replace(tzinfo=None),
            link=None,
        )
    args = get_args_dict(
        model=model_key,
        local_model_path=plan.model,
        n=1,
        codegen_n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=plan.max_new_tokens,
        output_path=str(plan.output_dir / "livecodebench"),
        enable_thinking=False,
        base_url=plan.base_url,
        start_date=plan.lcb_start_date,
        end_date=plan.lcb_end_date,
        evaluate=True,
        timeout=120,
        trust_remote_code=plan.trust_remote_code,
        tensor_parallel_size=plan.tensor_parallel_size,
    )
    lcb_main(args)


WORKERS: dict[str, Callable[[EvaluationPlan], None]] = {
    "lm-eval": _run_lm_eval,
    "evalscope": _run_evalscope,
    "evalplus": _run_evalplus,
    "livecodebench": _run_livecodebench,
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Benchmark produced an unreadable JSON result: {path}") from exc


def _validate_family_outputs(plan: EvaluationPlan, family: str) -> None:
    """Reject successful worker exits that did not leave benchmark results."""

    if family == "lm-eval":
        candidates = sorted((plan.output_dir / "lm_eval").rglob("results_*.json"))
        if not candidates:
            raise RuntimeError("lm-eval completed without writing an aggregated results JSON file")
        payload = _read_json(candidates[-1])
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict) or not payload["results"]:
            raise RuntimeError(f"lm-eval results are empty or malformed: {candidates[-1]}")
        return

    if family == "evalscope":
        root = plan.output_dir / "evalscope"
        for task in _family_tasks(plan, family):
            candidates = sorted(root.rglob(f"{task}.json"))
            if not candidates:
                raise RuntimeError(f"EvalScope completed without writing a report for {task}")
            payload = _read_json(candidates[-1])
            if not isinstance(payload, dict) or not payload:
                raise RuntimeError(f"EvalScope report is empty or malformed: {candidates[-1]}")
        return

    if family == "evalplus":
        root = plan.output_dir / "evalplus"
        for task in _family_tasks(plan, family):
            result_path = root / f"{task}_results.json"
            payload = _read_json(result_path) if result_path.exists() else None
            if not isinstance(payload, dict) or not payload:
                raise RuntimeError(f"EvalPlus completed without a valid result for {task}: {result_path}")
        return

    if family == "livecodebench":
        candidates = sorted((plan.output_dir / "livecodebench").glob("*_eval.json"))
        if not candidates:
            raise RuntimeError("LiveCodeBench completed without writing an evaluation result JSON file")
        payload = _read_json(candidates[-1])
        if not isinstance(payload, (dict, list)) or not payload:
            raise RuntimeError(f"LiveCodeBench result is empty or malformed: {candidates[-1]}")
        return

    raise ValueError(f"Unknown evaluation family: {family}")


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", choices=FAMILY_ORDER, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--backend", choices=("hf", "vllm"), required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--context-length", type=_positive_int, required=True)
    parser.add_argument("--max-new-tokens", type=_positive_int, required=True)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, required=True)
    parser.add_argument("--parallel-requests", type=_positive_int, required=True)
    parser.add_argument("--eval-batch-size", type=_positive_int, required=True)
    parser.add_argument("--lcb-start-date", required=True)
    parser.add_argument("--lcb-end-date", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _run_worker(argv: Sequence[str]) -> int:
    parser = _worker_parser()
    args = parser.parse_args(argv)
    try:
        tasks = _normalize_tasks(args.tasks)
        base_url = _normalize_base_url(args.base_url)
        _validate_lcb_date_range(args.lcb_start_date, args.lcb_end_date)
    except ValueError as exc:
        parser.error(str(exc))
    plan = EvaluationPlan(
        model=args.model,
        output_dir=args.output_dir,
        tasks=tasks,
        backend=args.backend,
        base_url=base_url,
        seed=args.seed,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        lcb_start_date=args.lcb_start_date,
        lcb_end_date=args.lcb_end_date,
        parallel_requests=args.parallel_requests,
        eval_batch_size=args.eval_batch_size,
    )
    expected_family_tasks = set(PAPER_TASKS[args.family])
    if not set(plan.tasks).issubset(expected_family_tasks):
        parser.error(f"worker {args.family} received tasks outside its benchmark family")
    WORKERS[args.family](plan)
    _validate_family_outputs(plan, args.family)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_worker":
        return _run_worker(arguments[1:])
    parser = _build_parser()
    args = parser.parse_args(arguments)
    plan = _plan_from_args(args, parser)
    return _run_driver(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
