from __future__ import annotations

import importlib.metadata
import inspect
import json
import pathlib
import subprocess
import sys
import urllib.parse
from collections.abc import Iterable


EXPECTED_VERSIONS = {
    "accelerate": "1.13.0",
    "datasets": "3.6.0",
    "fsspec": "2025.3.0",
    "grpcio-reflection": "1.71.2",
    "grpcio-status": "1.71.2",
    "huggingface-hub": "1.9.0",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "pyarrow": "23.0.1",
    "torch": "2.9.1",
    "transformers": "5.0.0",
    "vllm": "0.14.1",
    "lm-eval": "0.4.11",
    "modelscope": "1.34.0",
    "openai": "2.30.0",
    "oss2": "2.19.1",
    "protobuf": "5.29.6",
}

EXPECTED_COMMITS = {
    "evalplus": "59e9263ab2aa6d8cbfe3229cd05039de3bc43d67",
    "evalscope": "d3bec2b94a81eec2d1cc15af2a29547a4f5c5368",
    "livecodebench": "581bed959cafd2c209f2c5eaab4ea190ee0dcacd",
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _check_versions() -> None:
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            _fail(f"{distribution}: expected {expected}, found {actual}")
        print(f"[ok] {distribution}=={actual}")


def _direct_url_commit(distribution: str) -> str | None:
    direct_url = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    if not direct_url:
        return None
    payload = json.loads(direct_url)
    commit = payload.get("vcs_info", {}).get("commit_id")
    if commit:
        return commit
    if not payload.get("dir_info", {}).get("editable"):
        return None
    parsed = urllib.parse.urlparse(payload.get("url", ""))
    if parsed.scheme != "file":
        return None
    source_dir = pathlib.Path(urllib.parse.unquote(parsed.path))
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _check_vcs_sources() -> None:
    for distribution, expected in EXPECTED_COMMITS.items():
        actual = _direct_url_commit(distribution)
        if actual != expected:
            _fail(f"{distribution}: expected git commit {expected}, found {actual or 'no direct_url metadata'}")
        print(f"[ok] {distribution}@{actual[:12]}")


def _require_parameters(callable_object: object, required: Iterable[str], label: str) -> None:
    parameters = inspect.signature(callable_object).parameters
    missing = sorted(set(required) - set(parameters))
    if missing:
        _fail(f"{label} is missing required parameters: {', '.join(missing)}")


def _check_evaluation_apis() -> None:
    from evalplus.codegen import run_codegen
    from evalplus.evaluate import evaluate
    from evalplus.provider.base import DecoderBase
    from evalscope.config import TaskConfig
    from evalscope.run import run_task
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.runner.main import get_args_dict
    from modelscope.msdatasets import MsDataset

    _require_parameters(run_codegen, {"parallel_tasks", "enable_thinking", "base_url"}, "EvalPlus run_codegen")
    _require_parameters(evaluate, {"output_file"}, "EvalPlus evaluate")
    _require_parameters(DecoderBase.__init__, {"max_new_tokens"}, "EvalPlus DecoderBase")
    _require_parameters(get_args_dict, {"base_url", "enable_thinking", "timeout"}, "LiveCodeBench get_args_dict")
    if not hasattr(LMStyle, "ReapBase"):
        _fail("LiveCodeBench LMStyle.ReapBase is unavailable")
    if not callable(TaskConfig) or not callable(run_task):
        _fail("EvalScope TaskConfig/run_task API is unavailable")
    if not callable(MsDataset):
        _fail("ModelScope MsDataset API is unavailable")
    print("[ok] EvalPlus, EvalScope, ModelScope datasets, and LiveCodeBench evaluation APIs")


def _check_vllm_native_import() -> None:
    import torch
    import transformers
    import vllm
    import vllm._C  # noqa: F401

    print(
        "[ok] vLLM native extension import "
        f"(vllm={vllm.__version__}, torch={torch.__version__}, transformers={transformers.__version__})"
    )


def _check_pip_conflicts() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    required_overrides = {"grpcio-reflection", "protobuf", "transformers"}
    expected_overrides: dict[str, str] = {}
    unexpected = []
    for line in lines:
        normalized = line.lower()
        if (
            normalized.startswith("vllm 0.14.1 has requirement transformers<5")
            and "transformers 5.0.0" in normalized
        ):
            expected_overrides["transformers"] = line
        elif (
            normalized.startswith("vllm 0.14.1 has requirement protobuf>=6.30.0")
            and "protobuf 5.29.6" in normalized
        ):
            expected_overrides["protobuf"] = line
        elif (
            normalized.startswith("vllm 0.14.1 has requirement grpcio-reflection>=1.76.0")
            and "grpcio-reflection 1.71.2" in normalized
        ):
            expected_overrides["grpcio-reflection"] = line
        elif normalized != "no broken requirements found.":
            unexpected.append(line)
    if unexpected:
        _fail("Unexpected dependency conflicts:\n  " + "\n  ".join(unexpected))
    missing = sorted(required_overrides - expected_overrides.keys())
    if missing:
        _fail("Expected vLLM metadata overrides were not reported: " + ", ".join(missing))
    for line in expected_overrides.values():
        print("[expected override] " + line)


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        _fail(f"Python 3.12 is required; found {sys.version.split()[0]}")
    _check_versions()
    _check_vcs_sources()
    _check_evaluation_apis()
    _check_vllm_native_import()
    _check_pip_conflicts()
    print("Environment verification passed.")


if __name__ == "__main__":
    main()
