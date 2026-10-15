# Batch continuation checks

`check.py` runs four short Dhaka chains and captures earlier checkpoints at
iterations 3, 2, 3 and 2. These form a simulated partial save during adaptive
warmup. After the reference process finishes, the continuation resumes each
chain at its own iteration, with the original batch shape and random states.
It compares every checkpoint array except elapsed times against the reference.

Run with the same Pypomp/JAX and NUTS dependencies as the sampler:

```sh
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda PYTHONPATH=/tmp/dmop-nuts-deps \
  python code/bayesian/corrected/results/batch-continuation-check-v1/check.py \
  --output /tmp/dmop-batch-continuation-gpu-v1
```

The four-chain GPU check uses J100, G20, a float64 target, a float32 force,
maximum depth 2, six warmup iterations and two retained iterations. It tests
execution and recovery, not posterior convergence. Read `verification.json`
for the result once the check has finished.

An earlier two-chain CPU check extended a completed four-iteration Dhaka run
by two iterations. All numerical arrays matched a fresh six-iteration run
exactly. Unit tests separately reject changed source, lane order, cached state
and missing terminal verification, and replay unequal lanes for NUTS and PMMH.
