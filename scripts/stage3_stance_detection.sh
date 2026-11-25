#!/usr/bin/env bash
# Stage 3: article + generated image (or spec text) -> stance (outputs/predictions.jsonl).
#
#   bash scripts/stage3_stance_detection.sh --backend open --input-mode multimodal --limit 5
#   bash scripts/stage3_stance_detection.sh --backend open --input-mode text --limit 5
#
# All arguments are forwarded to the stage script; see --help for the full list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m vfstance.stage3_stance_detection "$@"
