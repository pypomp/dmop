#!/usr/bin/env bash
set -euo pipefail

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages fit \
  --output results/corenflos_j100_eps025_final_100 \
  --starts 100 \
  --particles 100 \
  --epsilon 0.25 \
  --optimizer adam \
  --learning-rate 0.02 \
  --learning-rate-decay 0.002 \
  --maximum-acceptable-invalid-fraction 1.0 \
  --maximum-updates 5000 \
  --maximum-elapsed-seconds 800 \
  --output-selection-seconds 700 \
  --output-selection maximum-pseudo
