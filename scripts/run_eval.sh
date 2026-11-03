#!/usr/bin/env bash
set -Eeuo pipefail

# Run MoE-PBA or its greedy variant end to end:
# pruning -> vLLM startup -> domain-matched evaluation -> vLLM shutdown.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}"

MODEL=""
RATIO=""
DOMAIN="code"
METHOD="moe_pba"
GPUS=""
CALIB_MICROBATCH_SIZE="8"
PORT="8000"
OUTPUT_DIR=""
DRY_RUN="false"

SEED=42
CALIB_SAMPLES=128
MODEL_MAX_LENGTH=4096
EVAL_MAX_NEW_TOKENS=2048
PARALLEL_REQUESTS=32
EVAL_BATCH_SIZE=8
VLLM_MAX_NUM_SEQS=32
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_START_TIMEOUT_SECS=1800
VLLM_SHUTDOWN_TIMEOUT_SECS=60
VLLM_GENERATION_CONFIG='{"temperature":0.0,"top_p":1.0,"top_k":-1,"min_p":0.0}'

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval.sh \
    --model MODEL \
    --ratio RATIO \
    --domain code|math \
    --gpus GPU_IDS \
    [--calib-microbatch-size auto|N] \
    [--method moe_pba|moe_pba_greedy] \
    [--port PORT] \
    [--output-dir PATH] \
    [--dry-run]

The fixed settings are seed 42, 128 packed calibration samples, mean per-example squared L2 scoring,
full-model reference, prefix conditioning, shared experts included in scoring,
and vLLM evaluation. The default method is moe_pba; the default calibration microbatch size is 8.

Domain mapping:
  code  -> Evol-CodeAlpaca; HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench
  math  -> Tulu math; GSM8K, MATH-500
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  local option="$1" value="${2:-}"
  [[ -n "${value}" ]] || die "${option} requires a value"
}

while (( $# > 0 )); do
  case "$1" in
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --ratio)
      require_value "$1" "${2:-}"
      RATIO="$2"
      shift 2
      ;;
    --domain)
      require_value "$1" "${2:-}"
      DOMAIN="$2"
      shift 2
      ;;
    --method)
      require_value "$1" "${2:-}"
      METHOD="$2"
      shift 2
      ;;
    --gpus)
      require_value "$1" "${2:-}"
      GPUS="$2"
      shift 2
      ;;
    --calib-microbatch-size)
      require_value "$1" "${2:-}"
      CALIB_MICROBATCH_SIZE="$2"
      shift 2
      ;;
    --port)
      require_value "$1" "${2:-}"
      PORT="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "${MODEL}" ]] || die "--model is required"
[[ -n "${RATIO}" ]] || die "--ratio is required"
[[ -n "${GPUS}" ]] || die "--gpus is required"

[[ "${RATIO}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || die "--ratio must be numeric"
[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be a comma-separated list such as 0 or 0,1"
[[ "${PORT}" =~ ^[0-9]+$ ]] || die "--port must be an integer"
(( PORT >= 1 && PORT <= 65535 )) || die "--port must be between 1 and 65535"
if [[ "${CALIB_MICROBATCH_SIZE}" != "auto" ]]; then
  [[ "${CALIB_MICROBATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
    die "--calib-microbatch-size must be auto or a positive integer"
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
TENSOR_PARALLEL_SIZE="${#GPU_LIST[@]}"

case "${DOMAIN}" in
  code)
    CALIB_DATASET="theblackcat102/evol-codealpaca-v1"
    EVAL_TASKS=(humaneval mbpp livecodebench)
    ;;
  math)
    CALIB_DATASET="allenai/tulu-3-sft-personas-math"
    EVAL_TASKS=(gsm8k math_500)
    ;;
  *)
    die "--domain must be code or math"
    ;;
esac

case "${METHOD}" in
  moe_pba|moe_pba_greedy) ;;
  *)
    die "--method must be moe_pba or moe_pba_greedy"
    ;;
esac

MODEL_ARTIFACT_NAME="${MODEL##*/}"
MODEL_ARTIFACT_NAME="${MODEL_ARTIFACT_NAME//[^[:alnum:]_.-]/_}"
printf -v RATIO_TAG '%.3f' "${RATIO}"
RATIO_TAG="${RATIO_TAG//./_}"
RUN_NAME="fit_${METHOD}-seed_${SEED}-ratio${RATIO_TAG}-samples${CALIB_SAMPLES}"
RUN_NAME+="-metric_squared_l2-reffull-crosslayerdynamic-calib_${DOMAIN}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/artifacts/moe_pba"
  OUTPUT_DIR+="/${MODEL_ARTIFACT_NAME}/${METHOD}/${RUN_NAME}"
elif [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi
EVAL_DIR="${OUTPUT_DIR}/eval_vllm"
SERVER_LOG="${OUTPUT_DIR}/vllm.log"

SEARCH_COMMAND=(
  "${PYTHON_BIN}" -m moe_pba
  --gpus "${GPUS}"
  --model_name "${MODEL}"
  --trust_remote_code true
  --method "${METHOD}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --calib_dataset_name "${CALIB_DATASET}"
  --calib_samples "${CALIB_SAMPLES}"
  --calib_pack_samples true
  --calib_microbatch_size "${CALIB_MICROBATCH_SIZE}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --sparsity_ratio "${RATIO}"
)

SERVER_COMMAND=(
  "${VLLM_BIN}" serve "${OUTPUT_DIR}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --seed "${SEED}"
  --max-model-len "${MODEL_MAX_LENGTH}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MODEL_MAX_LENGTH}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --override-generation-config "${VLLM_GENERATION_CONFIG}"
  --served-model-name "${OUTPUT_DIR}"
  --trust-remote-code
)

EVAL_COMMAND=(
  "${PYTHON_BIN}" -m moe_pba.eval
  --model "${OUTPUT_DIR}"
  --output-dir "${EVAL_DIR}"
  --tasks "${EVAL_TASKS[@]}"
  --backend vllm
  --base-url "http://127.0.0.1:${PORT}/v1"
  --seed "${SEED}"
  --context-length "${MODEL_MAX_LENGTH}"
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --parallel-requests "${PARALLEL_REQUESTS}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --lcb-start-date 2025-01-01
  --lcb-end-date 2025-07-31
  --trust-remote-code
)

print_command() {
  local label="$1"
  shift
  printf '%s CUDA_VISIBLE_DEVICES=%q ' "${label}" "${GPUS}"
  printf '%q ' "$@"
  printf '\n'
}

if [[ "${DRY_RUN}" == "true" ]]; then
  print_command SEARCH "${SEARCH_COMMAND[@]}"
  print_command SERVER setsid "${SERVER_COMMAND[@]}"
  print_command EVAL "${EVAL_COMMAND[@]}"
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || die "${PYTHON_BIN} not found; run bash scripts/install.sh first"
[[ -x "${VLLM_BIN}" ]] || die "${VLLM_BIN} not found; run bash scripts/install.sh first"
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v setsid >/dev/null 2>&1 || die "setsid is required"

VLLM_PID=""

cleanup() {
  local elapsed=0
  trap - EXIT INT TERM
  [[ -n "${VLLM_PID}" ]] || return 0

  if kill -0 -- "-${VLLM_PID}" 2>/dev/null; then
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || true
    while kill -0 -- "-${VLLM_PID}" 2>/dev/null && \
      (( elapsed < VLLM_SHUTDOWN_TIMEOUT_SECS )); do
      sleep 2
      elapsed="$((elapsed + 2))"
    done
    if kill -0 -- "-${VLLM_PID}" 2>/dev/null; then
      kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
    fi
  fi
  wait "${VLLM_PID}" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Running pruning: model=%s method=%s ratio=%s domain=%s gpus=%s\n' \
  "${MODEL}" "${METHOD}" "${RATIO}" "${DOMAIN}" "${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${SEARCH_COMMAND[@]}"

if curl --fail --silent --max-time 2 \
  "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  die "port ${PORT} is already serving a vLLM instance"
fi

printf 'Starting vLLM: port=%s tensor_parallel_size=%s log=%s\n' \
  "${PORT}" "${TENSOR_PARALLEL_SIZE}" "${SERVER_LOG}"
CUDA_VISIBLE_DEVICES="${GPUS}" setsid "${SERVER_COMMAND[@]}" >"${SERVER_LOG}" 2>&1 &
VLLM_PID="$!"

STARTED_AT="$(date +%s)"
while true; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    printf 'vLLM exited before becoming ready. Last log lines:\n' >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if curl --fail --silent --max-time 5 \
    "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  NOW="$(date +%s)"
  if (( NOW - STARTED_AT >= VLLM_START_TIMEOUT_SECS )); then
    printf 'Timed out waiting for vLLM. Last log lines:\n' >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

printf 'Running evaluation: tasks=%s\n' "${EVAL_TASKS[*]}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${EVAL_COMMAND[@]}"
printf 'Completed: output=%s evaluation=%s\n' "${OUTPUT_DIR}" "${EVAL_DIR}"
