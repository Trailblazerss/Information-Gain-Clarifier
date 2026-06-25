#!/usr/bin/env bash
# Paper-faithful τ-Bench baseline runner for Qwen3-8B / None.
#
# This wrapper:
# - restarts the Qwen3-8B vLLM server on port 18000
# - enables Qwen3 reasoning at the request layer
# - runs the raw τ-Bench tool-calling baseline with agent temp 0.01 and user temp 1.0
# - averages seeds 0, 1, and 2 by default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TAU_BENCH_ROOT="${TAU_BENCH_ROOT:-${PROJECT_ROOT}/../tau-bench}"
CONDA_BASE="${CONDA_BASE:-/home/lizhaofeng/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ig_pipeline}"
PYTHON_BIN="${PYTHON_BIN:-}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-8B}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-Qwen/Qwen3-8B}"
QWEN_PORT="${QWEN_PORT:-18000}"
QWEN_GPU_ID="${QWEN_GPU_ID:-3}"
QWEN_MEM_UTIL="${QWEN_MEM_UTIL:-0.85}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-40960}"
VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-hermes}"
FORCE_QWEN_RESTART="${FORCE_QWEN_RESTART:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

if [ -z "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="${TAU_BENCH_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://localhost:${QWEN_PORT}/v1"
export OPENAI_API_BASE="${OPENAI_BASE_URL}"

stop_qwen_server() {
    if pgrep -f "vllm.entrypoints.openai.api_server.*--port ${QWEN_PORT}" >/dev/null; then
        echo "[stop] vLLM on :${QWEN_PORT}"
        pkill -f "vllm.entrypoints.openai.api_server.*--port ${QWEN_PORT}" || true
        sleep "${VLLM_STOP_WAIT_SEC:-10}"
    fi
}

wait_for_server() {
    for _ in $(seq 1 120); do
        if curl -s -o /dev/null -w '%{http_code}' "http://localhost:${QWEN_PORT}/v1/models" | grep -qx "200"; then
            return 0
        fi
        sleep 5
    done
    return 1
}

if [ "${FORCE_QWEN_RESTART}" = "1" ]; then
    stop_qwen_server
fi

code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${QWEN_PORT}/v1/models" 2>/dev/null || echo 000)"
if [ "${code}" != "200" ]; then
    echo "[boot] Qwen3-8B server on :${QWEN_PORT} (CUDA_VISIBLE_DEVICES=${QWEN_GPU_ID})"
    CUDA_VISIBLE_DEVICES="${QWEN_GPU_ID}" nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${QWEN_MODEL}" \
        --served-model-name "${QWEN_SERVED_NAME}" \
        --port "${QWEN_PORT}" \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "${QWEN_MEM_UTIL}" \
        --max-model-len "${QWEN_MAX_MODEL_LEN}" \
        --enable-auto-tool-choice \
        --tool-call-parser "${VLLM_TOOL_PARSER}" \
        > "${LOG_DIR}/vllm_${QWEN_PORT}.log" 2>&1 &
    echo "    pid=$! -> ${LOG_DIR}/vllm_${QWEN_PORT}.log"
fi

echo "[wait] waiting for Qwen server readiness..."
if ! wait_for_server; then
    echo "[fail] Qwen server did not become ready; see ${LOG_DIR}/vllm_${QWEN_PORT}.log" >&2
    tail -n 40 "${LOG_DIR}/vllm_${QWEN_PORT}.log" >&2 2>/dev/null || true
    exit 1
fi
echo "[ready] :${QWEN_PORT}"

BASELINE_LOG="${LOG_DIR}/base_qwen_eval.log"
: > "${BASELINE_LOG}"

if [ "$#" -eq 0 ]; then
    ENV_ARGS=(retail airline)
else
    ENV_ARGS=("$@")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/base_qwen_eval.py" \
    --envs "${ENV_ARGS[@]}" \
    --seeds ${SEEDS:-0 1 2} \
    --base-url "${OPENAI_BASE_URL}" \
    --log-root "${RESULT_ROOT}" \
    2>&1 | tee -a "${BASELINE_LOG}"
