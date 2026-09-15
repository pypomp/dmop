#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f results/preflight_adaptive_j250_eps025/checkpoints/fit_start000.npz ]]; do
  sleep 30
done

common=(
  --starts 1
  --epsilon 0.25
  --optimizer adam
  --learning-rate 0.02
  --learning-rate-decay 0.002
  --maximum-updates 5000
  --maximum-elapsed-seconds 800
  --output-selection-seconds 700
  --output-selection maximum-pseudo
  --eval-particles 5000
  --eval-replicates 36
  --trace-eval-particles 5000
  --trace-eval-replicates 1
)

/home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages final-eval trace-eval \
  --output results/preflight_adaptive_eps025 \
  --particles 100 \
  "${common[@]}"

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m corenflos.benchmark \
  --stages final-eval trace-eval \
  --output results/preflight_adaptive_j250_eps025 \
  --particles 250 \
  "${common[@]}"
