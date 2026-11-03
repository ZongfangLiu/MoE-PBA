from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional


MOE_PBA_METRIC = "squared_l2"
MOE_PBA_SCORE_REDUCTION = "mean_per_example"
MOE_PBA_OBJECTIVE = "layer_output_drift"
MOE_PBA_REFERENCE = "full"
MOE_PBA_CROSS_LAYER_MODE = "dynamic"
MOE_PBA_INPUT_CONDITIONING = "committed_prefix"
MOE_PBA_INCLUDE_SHARED_EXPERTS = True

GREEDY_SEARCH_METHODS = (
    "moe_pba",
    "moe_pba_greedy",
)

def normalize_gpus(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    if normalized == "":
        return None
    parts = [part.strip() for part in normalized.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"gpus must be a comma-separated list like `0` or `4,5`. Received {raw_value!r}.")
    try:
        resolved = [str(int(part)) for part in parts]
    except ValueError as exc:
        raise ValueError(f"gpus must be a comma-separated list like `0` or `4,5`. Received {raw_value!r}.") from exc
    return ",".join(resolved)


def normalize_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(raw_value)


def normalize_method(raw_value: Any) -> str:
    normalized = str(raw_value or "moe_pba").strip().lower().replace("-", "_")
    aliases = {
        "moepba": "moe_pba",
        "moe_pba": "moe_pba",
        "moepba_greedy": "moe_pba_greedy",
        "moe_pba_greedy": "moe_pba_greedy",
    }
    method = aliases.get(normalized, normalized)
    if method not in GREEDY_SEARCH_METHODS:
        raise ValueError(f"method must be one of {list(GREEDY_SEARCH_METHODS)}. Received {raw_value!r}.")
    return method


@dataclass
class RunArgs:
    seed: int = field(default=42, metadata={"help": "Random seed."})


@dataclass
class DatasetArgs:
    dataset_name: str = field(default="theblackcat102/evol-codealpaca-v1", metadata={"help": "Fallback dataset name."})
    dataset_config_name: str = field(default="all", metadata={"help": "Fallback dataset config."})
    split: str = field(default="train", metadata={"help": "Fallback dataset split."})


@dataclass
class ObserverArgs:
    model_max_length: Optional[int] = field(default=4096, metadata={"help": "Maximum calibration sequence length."})
    truncate: bool = field(default=False, metadata={"help": "Truncate long calibration samples instead of skipping."})


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        metadata={"help": "Model name or local model path."},
    )
    local_files_only: bool = field(
        default=False,
        metadata={"help": "Load the model and tokenizer only from the local Hugging Face cache."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow model repositories to execute custom code. Not required for supported native models."},
    )


@dataclass
class GreedySearchArgs:
    gpus: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated physical GPU ids, for example `0` or `4,5`."},
    )
    method: str = field(
        default="moe_pba",
        metadata={"help": "Search method: `moe_pba` or `moe_pba_greedy`."},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional output directory. Defaults to artifacts/moe_pba/<model>/<method>/<run>."},
    )
    sparsity_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Uniform ratio of experts to prune per MoE layer."},
    )
    resolved_prune_count: Optional[int] = field(default=None, init=False, repr=False)
    calib_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Calibration dataset name. Multiple names can be comma-separated."},
    )
    calib_dataset_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated calibration dataset weights."},
    )
    calib_dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Calibration dataset config."})
    calib_dataset_split: str = field(default="train", metadata={"help": "Calibration dataset split."})
    calib_samples: int = field(default=128, metadata={"help": "Total calibration sequences across all datasets."})
    calib_pack_samples: bool = field(default=True, metadata={"help": "Pack calibration samples into full sequences."})
    calib_microbatch_size: str = field(default="8", metadata={"help": "Calibration microbatch size or `auto`."})

    def __post_init__(self) -> None:
        self.gpus = normalize_gpus(self.gpus)
        self.method = normalize_method(self.method)

        if self.sparsity_ratio is not None:
            self.sparsity_ratio = float(self.sparsity_ratio)
            if not math.isfinite(self.sparsity_ratio) or not (0.0 <= self.sparsity_ratio <= 1.0):
                raise ValueError("sparsity_ratio must be finite and within [0, 1].")

        if self.calib_samples <= 0:
            raise ValueError("calib_samples must be positive.")
        self.calib_pack_samples = normalize_bool(self.calib_pack_samples)
