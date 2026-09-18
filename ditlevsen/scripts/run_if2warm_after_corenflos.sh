#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f ../corenflos/results/if2warm_ifad097_budget_j100_final_100/checkpoints/fit_start099.npz ]]; do
  sleep 60
done

exec scripts/run_if2warm_final100.sh
