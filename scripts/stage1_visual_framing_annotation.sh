#!/usr/bin/env bash
# Stage 1: article -> visual framing spec (outputs/stage1.jsonl).
#
#   bash scripts/stage1_visual_framing_annotation.sh --backend open --limit 5
#
# All arguments are forwarded to the stage script; see --help for the full list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m vfstance.stage1_visual_framing_annotation "$@"
