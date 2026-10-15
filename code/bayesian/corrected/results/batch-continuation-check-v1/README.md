# Batch continuation checks

`check.py` runs four short Dhaka chains and captures earlier checkpoints at
iterations 3, 2, 3 and 2. These form a simulated partial save during adaptive
warmup. After the reference process finishes, the continuation resumes each
chain at its own iteration, with the original batch shape and random states.
It compares every checkpoint array except elapsed times against the reference.

Run with the same Pypomp/JAX and NUTS dependencies as the sampler:

```sh
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda PYTHONPATH=/tmp/dmop-nuts-deps \
  XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python code/bayesian/corrected/results/batch-continuation-check-v1/check.py \
  --output /tmp/dmop-batch-continuation-gpu-deterministic-v1
```

The four-chain GPU check uses J100, G20, a float64 target, a float32 force,
maximum depth 2, six warmup iterations and two retained iterations. It tests
execution and recovery, not posterior convergence. The full deterministic
replay has not yet run. The initial check without deterministic flags failed
bitwise comparison: random states and discrete decisions matched, with
parameter differences up to 1.03e-11. See `gpu-replay-differences.json`.

`repeat_force.py` reproduced small force differences at identical inputs
without any continuation. Deterministic XLA operations removed that variation
in all five evaluations; both results are saved as `force-repeatability*.json`.
The float64 likelihood was bitwise repeatable in both execution modes.

An earlier two-chain CPU check extended a completed four-iteration Dhaka run
by two iterations. All numerical arrays matched a fresh six-iteration run
exactly. Unit tests separately reject changed source, lane order, cached state
and missing terminal verification, and replay unequal lanes for NUTS and PMMH.
