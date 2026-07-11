#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/config.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
NUM_SHARDS="${NUM_SHARDS:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  conda activate "${CONDA_ENV_NAME}"
elif conda env list | awk '{print $1}' | grep -Fxq "acl2026"; then
  conda activate acl2026
else
  echo "Skipping conda activation. Set CONDA_ENV_NAME to activate a specific environment." >&2
fi

detect_num_shards() {
  if [[ -n "${NUM_SHARDS}" ]]; then
    echo "${NUM_SHARDS}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${gpu_count}" =~ ^[0-9]+$ ]] && [[ "${gpu_count}" -gt 0 ]]; then
      echo "${gpu_count}"
      return
    fi
  fi

  echo 1
}

SHARD_COUNT="$(detect_num_shards)"

run_step() {
  local script_name="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${script_name}"
  python "${ROOT_DIR}/src/${script_name}" --config "${CONFIG_PATH}"
}

run_sharded_step() {
  local script_name="$1"
  shift
  local pids=()
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${script_name} across ${SHARD_COUNT} shard(s)"
  for ((shard_idx=0; shard_idx<SHARD_COUNT; shard_idx++)); do
    python "${ROOT_DIR}/src/${script_name}" --config "${CONFIG_PATH}" --gpu "${shard_idx}" --shard "${shard_idx}" --num-shards "${SHARD_COUNT}" "$@" &
    pids+=("$!")
  done
  wait "${pids[@]}"
}

run_step "step1_preprocess.py"
run_step "step2_finetune_stance.py"
run_step "step2_predict_sentiment.py"
run_step "step3a_build_prompts.py"
run_step "step3b_expand_prompts_batch.py"
run_sharded_step "step3c_generate_images.py" --splits train,val,test
run_sharded_step "step4a_clip_score.py" --splits train,val,test
run_sharded_step "step4b_vlm_evaluate.py" --splits train,val,test
run_step "step4c_chatgpt_evaluate_batch.py"
run_step "step4d_pagerank_select.py"

echo "Pipeline completed."
