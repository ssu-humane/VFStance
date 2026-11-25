#!/usr/bin/env bash
# Run the full VFStance pipeline in one of the two reported configurations.
#
#   bash scripts/run_all.sh --config full --backend open --limit 2
#   bash scripts/run_all.sh --config text --backend api
#
#   full  VFStance          Stage 1 -> Stage 2 -> Stage 3 (article + generated image)
#   text  VFStance (Text)   Stage 1 -> Stage 3 (article + spec rendered as text)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/outputs"

CONFIG="full"
BACKEND="api"
SPLIT="test"
LIMIT=""
LVLM_MODEL=""
T2I_MODEL=""
DATA_PATH=""

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_all.sh [options]

  --config full|text      full: Stage 1 -> 2 -> 3 (VFStance, default)
                          text: Stage 1 -> 3 with the spec as text (VFStance (Text))
  --backend api|open      api: hosted closed models (default); open: open-weight models
  --split SPLIT           dataset split (default: test)
  --data-path PATH        dataset file (default: data/k_news_stance_mm/ko/k-news-stance-mm.json)
  --limit N               use only the first N articles (smoke test)
  --lvlm-model NAME       model for Stages 1 and 3
  --t2i-model NAME        text-to-image model for Stage 2
  -h, --help              show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --lvlm-model) LVLM_MODEL="$2"; shift 2 ;;
    --t2i-model) T2I_MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_all.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${CONFIG}" in full|text) ;; *) echo "run_all.sh: --config must be full or text" >&2; exit 2 ;; esac
case "${BACKEND}" in api|open) ;; *) echo "run_all.sh: --backend must be api or open" >&2; exit 2 ;; esac

STAGE1_ARGS=(--backend "${BACKEND}" --split "${SPLIT}")
STAGE2_ARGS=(--backend "${BACKEND}")
STAGE3_ARGS=(--backend "${BACKEND}" --split "${SPLIT}")
if [[ -n "${LIMIT}" ]]; then
  STAGE1_ARGS+=(--limit "${LIMIT}")
  STAGE2_ARGS+=(--limit "${LIMIT}")
  STAGE3_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${DATA_PATH}" ]]; then
  STAGE1_ARGS+=(--data-path "${DATA_PATH}")
  STAGE3_ARGS+=(--data-path "${DATA_PATH}")
fi
if [[ -n "${LVLM_MODEL}" ]]; then
  STAGE1_ARGS+=(--model "${LVLM_MODEL}")
  STAGE3_ARGS+=(--model "${LVLM_MODEL}")
fi
if [[ -n "${T2I_MODEL}" ]]; then
  STAGE2_ARGS+=(--model "${T2I_MODEL}")
fi

echo "[run_all] config=${CONFIG} backend=${BACKEND} split=${SPLIT} limit=${LIMIT:-all}"

# Each stage runs as its own process, so an open-weight model is released before
# the next stage loads its own: the GPU memory peak stays at one model.
bash "${SCRIPT_DIR}/stage1_visual_framing_annotation.sh" "${STAGE1_ARGS[@]}"

if [[ "${CONFIG}" == "full" ]]; then
  bash "${SCRIPT_DIR}/stage2_stance_aware_image_generation.sh" "${STAGE2_ARGS[@]}"
  STAGE3_ARGS+=(--input-mode multimodal --output "${OUTPUT_DIR}/predictions.jsonl")
else
  STAGE3_ARGS+=(--input-mode text --output "${OUTPUT_DIR}/predictions_text.jsonl")
fi

bash "${SCRIPT_DIR}/stage3_stance_detection.sh" "${STAGE3_ARGS[@]}"

echo "[run_all] finished config=${CONFIG} backend=${BACKEND}"
