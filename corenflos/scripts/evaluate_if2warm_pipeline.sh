#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f ../ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100/checkpoints/fit_r20_start099.npz ]]; do
  sleep 60
done

bash scripts/evaluate_if2warm_starts.sh
bash scripts/evaluate_if2warm_final100.sh

pushd ../ditlevsen >/dev/null
bash scripts/evaluate_if2warm_final100.sh
popd >/dev/null

/home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.warm_plots
/home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.warm_audit
