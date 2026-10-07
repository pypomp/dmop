#!/usr/bin/env bash
set -euo pipefail

# Complete the 100-start fits without GPU contention, then perform the
# independent Euler-20 final and every-update evaluations and render figures.
bash scripts/run_if2warm_final100.sh

pushd ../ditlevsen >/dev/null
bash scripts/run_if2warm_final100.sh
popd >/dev/null

bash scripts/evaluate_if2warm_pipeline.sh
