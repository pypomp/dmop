#!/usr/bin/env bash
set -euo pipefail

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages final-eval trace-eval \
  --output results/if2warm_ifad097_budget_j100_final_100 \
  --starts 100 \
  --starts-file ../ditlevsen/results/reference/ifad097_comparable_post_if2.npz \
  --elapsed-time-offset-seconds 244.44243359565735 \
  --particles 100 \
  --epsilon 0.25 \
  --optimizer adam \
  --learning-rate 0.02 \
  --learning-rate-decay 0.002 \
  --maximum-acceptable-invalid-fraction 1.0 \
  --maximum-updates 175 \
  --maximum-elapsed-seconds 614.6859633922577 \
  --output-selection-seconds 859.128396987915 \
  --output-selection fixed-time \
  --eval-particles 5000 \
  --eval-replicates 36 \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1
