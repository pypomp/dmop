# Timing interpretation

The concurrent candidate chains run on separate cores of a hybrid Intel
13900K. The NUTS candidates use performance cores; the random-walk
candidates use efficiency cores. Each configuration records its CPU affinity.
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
