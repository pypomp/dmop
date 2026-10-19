# Timing interpretation

## Current GPU comparison

The primary runs are `gpu-forward-nuts-v2/nuts` (NUTS on GPU0) and
`gpu-comparison-v2/pmrw-long` (J10000 PMMH on GPU1). NUTS is still running;
the primary PMMH, J5000 continuation and wider-prior comparison have
finished. GPU1 was shared during parts of these experiments. Their elapsed
times do not establish a controlled speed comparison.

`batch.py` records the wall time of one update of all four chains in each
chain's `iteration_seconds` array. Count that time once, not four times.
The first update includes transition compilation. The status clock also
includes initialization and intervening host work, so it is different from
the sum of update times. Neither includes preparation done by earlier jobs.

`batch_extend.py` copies the parent's timing arrays and adds its elapsed time
to the parent's status clock. Use the continuation's cumulative record;
adding the parent again counts it twice. Compilation of the continuation's
transition is included in its first new update.

Preparation includes the original initialization and score-based metric,
the PMMH pilot used for the shared proposal covariance, and, for the current
NUTS configuration, the retired NUTS warmups and the depth and covariance
screens. The latter covariance also uses the primary PMMH reference samples
identified in `nuts-metric-check-v1/protocol.json`. Account for those samples
as a preparation dependency when reporting NUTS cost. Shared work should be
identified explicitly rather than charged twice to a combined workflow.
The current runs' own warmup is additional. The retired forward-force run's
148 updates alone took 6744.85 seconds for the whole batch, as recorded in
`gpu-forward-nuts-v1/termination.json`; this is not its full setup cost.

A final comparison should report preparation, compilation, warmup and
retained sampling separately, with timing under comparable hardware load.
Short kernel or tuning-screen timings alone do not establish an effective
sample size per second advantage. The current comparison script makes no
speedup claim.

## Earlier CPU development runs

The concurrent candidate chains run on separate cores of a hybrid Intel
13900K. The NUTS candidates use performance cores. The random-walk
candidates started on efficiency cores; three were moved to released
performance cores during retained sampling (see their `affinity_change.json`). Each configuration records its CPU affinity.
Their elapsed wall times are therefore not a matched hardware speed comparison.

These runs assess posterior sampling and iteration counts. Any reported speed
comparison must additionally time the selected kernels serially on the same
core class, with the same target, proposal preparation accounted for, and
compilation identified separately. New checkpoints retain per-iteration times;
these do not remove the need for that controlled comparison. NUTS also records
its variable number of integration steps. The initial score-derived metric
and optimized starts are shared with random-walk MH, so that preparation is
not described as derivative-free.

The completed serial force-count check in `timing-force-counts.json` compares
short frozen MALA continuations on core8, excluding compilation. With5,000
likelihood particles, median transition times were9.383s for5,000 force
particles and3.437s for1,000 force particles. This measures kernel cost;
it does not establish a posterior ESS-per-second speedup. The continuations
copy their parent traces and cannot be counted as independent chains.

The stopped cheaper-force MALA development candidates started from saved iteration250 states
of the original full-force warmup runs, with new seeds and a fresh1,000-step
warmup. Original optimization and warmup cost3262.72s elapsed per chain and
remain part of preparation. The newer run clocks do not include that cost.
