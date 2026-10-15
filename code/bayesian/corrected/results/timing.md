# Timing interpretation

The concurrent candidate chains run on separate cores of a hybrid Intel
13900K. The MALA candidates currently use performance cores; the random-walk
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
