# GPU precision and batching checks

These are fixed-point checks, not posterior samples. The four positions and
proposal matrices in `inputs.npz` came from saved CPU NUTS warmup checkpoints;
`protocol.json` records their hashes and iteration counts. All checks use
5,000 likelihood particles, 1,000 force particles, alpha .95, the common
proper prior and maximum NUTS depth 4. Both GPUs have unrelated shared work.

`../../device_check.py` generates each run. Its command, physical GPU, host
PID and process creation time are in the corresponding `*-launch.json`.
Each run saves its inputs, source and results; read `result.json` for completion.
The default production precision remains float64. These checks do not alter
existing sampler states or frozen source snapshots.

| Directory | Purpose |
|---|---|
| `float32`, `float64` | Initial 16 likelihood evaluations at each point; the float32 run also times scalar NUTS |
| `float32-reps128`, `float64-reps128` | Expanded likelihood comparison; includes the initial 16 seed values, so do not pool with the screen |
| `float32-batch2`, `float32-batch4` | Vectorized independent NUTS updates, testing GPU throughput |
| `float64-batch4` | Full-precision GPU throughput; check completion before interpreting |
| `mixed-force-batch4` | Float64 likelihood/acceptance with float32 proposal force; see `mixed-force-protocol.json` |
| `deterministic-mixed-batch4` | Same sizes with `--xla_gpu_deterministic_ops=true`; check completion before using timings |

Three postcompilation float32 timings give median wall times of 34.50 seconds
for one chain, 33.53 seconds for two chains together, and 39.11 seconds for
four chains together. Each update took 15 integration steps. This measures
kernel throughput under shared load, not effective samples per second.
The earlier float64 scalar check is in `../gpu-current-config-check-v1`.
Full float64 batch4 completed with median 85.94 seconds per four-chain update.
The mixed-force experiment retains the same likelihood precision; its first
likelihood evaluation at each of the four saved points exactly matches the
corresponding float64 result. Read its completion flag before using timings.

`batch-smoke-verification.json` records a short real-model check of the new
batch runner, including checkpoint dimensions, finite float64 targets and
step sizes frozen after warmup. Separate unit tests compare batched and scalar
NUTS/PMMH transitions, including RNG states, for a known noisy Gaussian target.
These checks verify execution; they provide no posterior convergence evidence.

Run `python summarize_precision.py` to reproduce `precision-comparison.json`.
It compares log averages of likelihood estimates, using 4,000 bootstrap
resamples of the 128 seed-index pairs at each point. Float32-minus-float64
differences are .050, .062, .332 and .108 log-likelihood units. The pointwise
95% interval at point 2 is [.071,.582]; the other intervals contain zero.
Thus this check does not establish identical numerical likelihoods, and it
does not establish posterior sensitivity across the parameter space.
