#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f results/preflight_adaptive_eps025/checkpoints/fit_start000.npz ]]; do
  sleep 30
done

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages fit \
  --output results/preflight_adaptive_j250_eps025 \
  --starts 1 \
  --particles 250 \
  --epsilon 0.25 \
  --optimizer adam \
  --learning-rate 0.02 \
  --learning-rate-decay 0.002 \
  --maximum-updates 5000 \
  --maximum-elapsed-seconds 800 \
  --output-selection-seconds 700 \
  --output-selection maximum-pseudo
