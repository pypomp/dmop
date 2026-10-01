#!/usr/bin/env bash
set -euo pipefail

common=(
  --output results/recovery_start4_invalid1
  --starts 10
  --workers 10
  --worker-index 4
  --particles 100
  --epsilon 0.25
  --optimizer adam
  --learning-rate 0.02
  --learning-rate-decay 0.002
  --maximum-acceptable-invalid-fraction 1.0
  --maximum-updates 5000
  --maximum-elapsed-seconds 800
  --output-selection-seconds 700
  --output-selection maximum-pseudo
)

/home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages fit \
  "${common[@]}"

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages final-eval \
  --eval-particles 5000 \
  --eval-replicates 36 \
  "${common[@]}"
