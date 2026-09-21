#!/usr/bin/env bash
set -euo pipefail

# Ten evenly spaced members of the final 100-run experiment. The checkpoints
# are written directly into the final output directories and are reused when
# the remaining starts run.
bash scripts/evaluate_if2warm_starts.sh
bash scripts/run_if2warm_final100.sh --workers 10 --worker-index 0
IF2WARM_FINAL_ONLY=1 bash scripts/evaluate_if2warm_final100.sh \
  --workers 10 --worker-index 0

pushd ../ditlevsen >/dev/null
bash scripts/run_if2warm_final100.sh --workers 10 --worker-index 0
IF2WARM_FINAL_ONLY=1 bash scripts/evaluate_if2warm_final100.sh \
  --workers 10 --worker-index 0
popd >/dev/null
