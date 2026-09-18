#!/usr/bin/env bash
set -euo pipefail

exec /home/kevin/anaconda3/envs/pypomp/bin/python -m ditlevsen.smc_benchmark \
  --stages final-eval trace-eval \
  --output results/block_smc_guided_j100_if2warm_ifad097_budget_final_100 \
  --starts 100 \
  --starts-file results/reference/ifad097_comparable_post_if2.npz \
  --elapsed-time-offset-seconds 244.44243359565735 \
  --nsteps 20 \
  --transition block \
  --proposal guided \
  --particles 100 \
  --iterations 175 \
  --learning-rate 0.10 \
  --learning-rate-decay-start 90 \
  --learning-rate-decay-exponent 0.3 \
  --burnin 30 \
  --gain-exponent 0.9 \
  --maximum-acceptable-invalid-fraction 0.8 \
  --likelihood-guard-particles 100 \
  --likelihood-guard-interval 10 \
  --maximum-guard-loglik-drop 5.0 \
  --maximum-elapsed-seconds 614.6859633922577 \
  --output-selection-seconds 859.128396987915 \
  --trace-every-update \
  --eval-particles 5000 \
  --eval-replicates 36 \
  --trace-eval-particles 5000 \
  --trace-eval-replicates 1
