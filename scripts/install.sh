#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${PBA_VENV_DIR:-${repo_root}/.venv}"
livecodebench_commit="581bed959cafd2c209f2c5eaab4ea190ee0dcacd"
livecodebench_source="${venv_dir}/src/livecodebench"

run_with_retries() {
    local attempt=1
    local max_attempts=3
    local delay_seconds=5
    until "$@"; do
        if (( attempt >= max_attempts )); then
            printf 'Command failed after %d attempts:' "${max_attempts}" >&2
            printf ' %q' "$@" >&2
            printf '\n' >&2
            return 1
        fi
        printf 'Command failed (attempt %d/%d); retrying in %d seconds.\n' \
            "${attempt}" "${max_attempts}" "${delay_seconds}" >&2
        sleep "${delay_seconds}"
        ((attempt += 1))
        delay_seconds=$((delay_seconds * 2))
    done
}

find_python_312() {
    if [[ -n "${PBA_PYTHON:-}" ]]; then
        command -v "${PBA_PYTHON}" 2>/dev/null || printf '%s\n' "${PBA_PYTHON}"
        return
    fi
    if command -v python3.12 >/dev/null 2>&1; then
        command -v python3.12
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        uv python find 3.12 2>/dev/null && return
    fi
    return 1
}

if [[ -x "${venv_dir}/bin/python" ]]; then
    venv_python="${venv_dir}/bin/python"
else
    bootstrap_python="$(find_python_312)" || {
        printf '%s\n' \
            "Python 3.12 was not found. Install it or set PBA_PYTHON=/path/to/python3.12." >&2
        exit 1
    }
    "${bootstrap_python}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required; found {sys.version.split()[0]}")
PY
    "${bootstrap_python}" -m venv "${venv_dir}"
    venv_python="${venv_dir}/bin/python"
fi

"${venv_python}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"The existing .venv must use Python 3.12; found {sys.version.split()[0]}")
PY

run_with_retries "${venv_python}" -m pip install --upgrade \
    "pip==26.0.1" \
    "setuptools==79.0.1" \
    "wheel==0.46.3"

run_with_retries "${venv_python}" -m pip install --upgrade --editable "${repo_root}[evaluation]"

run_with_retries "${venv_python}" -m pip install --upgrade \
    --requirement "${repo_root}/requirements-livecodebench.txt"

mkdir -p "${venv_dir}/src"
if [[ -e "${livecodebench_source}" && ! -d "${livecodebench_source}/.git" ]]; then
    printf '%s\n' \
        "${livecodebench_source} exists but is not a Git checkout; move it aside and retry." >&2
    exit 1
fi
if [[ ! -d "${livecodebench_source}/.git" ]]; then
    run_with_retries git clone --no-checkout \
        https://github.com/mklasby/LiveCodeBench.git "${livecodebench_source}"
fi
run_with_retries git -C "${livecodebench_source}" fetch origin "${livecodebench_commit}"
run_with_retries git -C "${livecodebench_source}" checkout --detach --force "${livecodebench_commit}"
run_with_retries "${venv_python}" -m pip install --no-deps --editable "${livecodebench_source}"

run_with_retries "${venv_python}" -m pip install --upgrade --force-reinstall \
    --requirement "${repo_root}/requirements-transformers-override.txt"

"${venv_python}" "${repo_root}/scripts/check_environment.py"

printf '\nInstallation complete. Activate with:\n  source %s/bin/activate\n' "${venv_dir}"
