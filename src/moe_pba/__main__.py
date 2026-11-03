from __future__ import annotations

import os
import sys

from .args import normalize_gpus


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"Missing value for {flag}.")
            return argv[idx + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _bootstrap_gpus(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> str | None:
    args = sys.argv[1:] if argv is None else list(argv)
    env = os.environ if environ is None else environ
    raw_value = _extract_flag_value(args, "--gpus")
    normalized = normalize_gpus(raw_value)
    if normalized is not None:
        env["CUDA_VISIBLE_DEVICES"] = normalized
    return normalized


def main() -> None:
    _bootstrap_gpus()
    from .main import main as _main

    _main()


if __name__ == "__main__":
    main()
