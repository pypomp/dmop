# Forward differentiation of the proposal force

This isolates the differentiation operator used for the MOP proposal force.
The running samplers and their model source are unchanged. The likelihood
target remains untempered and in float64; these evaluations concern only
the 1,000-particle float32 force used to construct NUTS proposals.

`check.py` compares the current reverse derivative with a forward derivative
at the four frozen production starts and fixed particle keys. Scalar values
must match exactly; the maximum force difference divided by one plus the
largest reference force entry must be at most 1e-4. Repeated evaluations
also record bitwise agreement. These are local numerical checks, not a
proof of equivalence throughout parameter space.

Two implementations were checked. The default `direct` candidate changes
the differentiation call in an isolated model copy. The `wrapped` candidate
instead projects the scalar value from the existing value-and-gradient
function and applies `jax.jacfwd` to that value. It leaves the model source
unchanged; the compiler can eliminate the unused reverse derivative. The
generated `forward_model.py` is unused for the wrapped candidate.

| Check | Reverse median seconds | Forward median seconds | Scaled max force difference |
| --- | ---: | ---: | ---: |
| GPU direct, four chains | 4.688 | 1.070 | 1.56e-6 |
| GPU wrapped, four chains | 4.595 | 1.008 | 1.56e-6 |

Each median uses three postcompilation evaluations. Both GPU checks used
GPU1 with the same optimized deterministic XLA flags as production. Scalar
values matched exactly and repeated evaluations were bitwise identical in
both modes. Two-chain CPU smoke checks at 20 force particles also passed.
GPU1 hosted concurrent PMMH work, so these timings are provisional force
costs, not a controlled method speedup or effective-sample comparison.

Both benchmark processes are terminal and their result files are complete.
The direct check used commit f92e8c0db; the wrapped extension used 4939f6d99.
Source and launch hashes are recorded with each run. Existing production
processes were verified still live after the checks.

Next, validate the wrapper inside complete NUTS updates, then test exact
interrupted-batch continuation with it. Integrate it as an explicit optional
force mode only after those checks. Any new sampling configuration must be
recorded; do not change a running chain's numerical force silently. The
existing likelihood/model source can remain identical across settings.
