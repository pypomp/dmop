#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f results/corenflos_j100_eps025_pilot_10/checkpoints/fit_start009.npz ]]; do
  sleep 30
done

/home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages final-eval trace-eval \
  --output results/corenflos_j100_eps025_pilot_10 \
  --starts 10 \
  --particles 100 \
  --epsilon 0.25 \
  --optimizer adam \
  --learning-rate 0.02 \
  --learning-rate-decay 0.002 \
  --maximum-acceptable-invalid-fraction 0.5 \
  --maximum-updates 5000 \
  --maximum-elapsed-seconds 800 \
  --output-selection-seconds 700 \
  --output-selection maximum-pseudo \
  --eval-particles 5000 \
  --eval-replicates 36 \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.plots \
  --data results/corenflos_j100_eps025_pilot_10
