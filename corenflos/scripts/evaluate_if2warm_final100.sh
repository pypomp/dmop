#!/usr/bin/env bash
set -euo pipefail

stages=(final-eval trace-eval)
if [[ "${IF2WARM_FINAL_ONLY:-0}" == "1" ]]; then
  stages=(final-eval)
fi

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages "${stages[@]}" \
  --output results/if2warm_ifad097_budget_j100_final_100 \
  --starts 100 \
  --starts-file ../ditlevsen/results/reference/ifad097_comparable_post_if2.npz \
  --elapsed-time-offset-seconds 92.16042757034302 \
  --particles 100 \
  --epsilon 0.25 \
  --optimizer adam \
  --learning-rate 0.0005 \
  --learning-rate-decay 0.002 \
  --maximum-acceptable-invalid-fraction 1.0 \
  --maximum-updates 400 \
  --maximum-elapsed-seconds 850.6569547653198 \
  --output-selection-seconds 942.8173823356628 \
  --output-selection fixed-time \
  --eval-particles 5000 \
  --eval-replicates 36 \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1 \
  "$@"
