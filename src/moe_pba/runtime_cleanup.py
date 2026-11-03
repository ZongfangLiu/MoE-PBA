from __future__ import annotations

import gc
import logging

import torch

try:
    from accelerate.hooks import remove_hook_from_module
except Exception:
    def remove_hook_from_module(_module, recurse=True):
        return None


def log_visible_cuda_memory(logger: logging.Logger, prefix: str) -> None:
    if not torch.cuda.is_available():
        logger.info("%s: cuda unavailable", prefix)
        return
    snapshots: list[str] = []
    for device_idx in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            used_bytes = int(total_bytes - free_bytes)
            snapshots.append(
                f"cuda:{device_idx} free={free_bytes / (1024 ** 3):.2f}GiB "
                f"used={used_bytes / (1024 ** 3):.2f}GiB total={total_bytes / (1024 ** 3):.2f}GiB"
            )
        except Exception as exc:
            snapshots.append(f"cuda:{device_idx} mem-info-unavailable ({exc})")
    logger.info("%s: %s", prefix, "; ".join(snapshots))


def release_runtime(
    model=None,
    *,
    logger: logging.Logger | None = None,
    label: str = "Runtime cleanup",
) -> None:
    active_logger = logger or logging.getLogger(__name__)
    log_visible_cuda_memory(active_logger, f"{label} before release")

    if model is not None:
        try:
            remove_hook_from_module(model, recurse=True)
            active_logger.info("Removed accelerate hooks from model.")
        except Exception as exc:
            active_logger.warning("Failed to remove accelerate hooks from model: %s", exc)
        try:
            model.to("cpu")
            active_logger.info("Moved model to CPU.")
        except Exception as exc:
            active_logger.warning("Failed to move model to CPU: %s", exc)

    gc.collect()
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_idx):
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception as exc:
                active_logger.warning("CUDA cleanup failed on cuda:%d: %s", device_idx, exc)
    gc.collect()
    log_visible_cuda_memory(active_logger, f"{label} after release")
