#!/usr/bin/env bash
# Stage 2: visual framing spec -> generated image (outputs/images/id_{id}_image_1.png).
#
#   bash scripts/stage2_stance_aware_image_generation.sh --backend open --limit 5
#
# All arguments are forwarded to the stage script; see --help for the full list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m vfstance.stage2_stance_aware_image_generation "$@"
