#!/usr/bin/env bash
set -euo pipefail

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.warm_start_evaluate \
  --starts-file ../ditlevsen/results/reference/ifad097_comparable_post_if2.npz \
  --output results/ifad097_post_if2_reference \
  --starts 100 \
  --particles 5000 \
  --replicates 36 \
  --if2-elapsed-seconds 92.16042757034302
