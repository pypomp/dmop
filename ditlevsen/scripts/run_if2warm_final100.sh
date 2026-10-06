#!/usr/bin/env bash
set -euo pipefail

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.smc_benchmark \
  --stages fit \
  --output results/block_smc_guided_j100_if2warm_ifad097_budget_final_100 \
  --starts 100 \
  --starts-file results/reference/ifad097_comparable_post_if2.npz \
  --elapsed-time-offset-seconds 92.16042757034302 \
  --nsteps 20 \
  --transition block \
  --proposal guided \
  --particles 100 \
  --iterations 400 \
  --learning-rate 0.0005 \
  --learning-rate-decay-start 90 \
  --learning-rate-decay-exponent 0.3 \
  --burnin 30 \
  --gain-exponent 0.9 \
  --maximum-acceptable-invalid-fraction 0.8 \
  --likelihood-guard-particles 100 \
  --likelihood-guard-interval 10 \
  --maximum-guard-loglik-drop 5.0 \
  --maximum-elapsed-seconds 850.6569547653198 \
  --output-selection-seconds 942.8173823356628 \
  --trace-every-update \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1 \
  --eval-particles 5000 \
  --eval-replicates 36 \
  "$@"
