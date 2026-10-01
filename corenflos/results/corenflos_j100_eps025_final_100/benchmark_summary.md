# Dacca benchmark summary

This run contains 100 bounding-box starts of the Corenflos differentiable
particle filter with 100 fitting particles and an 800-second budget per start.
The selected checkpoint maximizes the differentiable-filter likelihood before
700 seconds. Selection does not use the Euler likelihood.

All reported likelihoods are ordinary Pypomp Euler-20 particle-filter
likelihoods. Final estimates use 5,000 particles and 36 replicates. The
elapsed-time trace evaluates every stored optimizer update with 5,000
particles and one replicate; its 27,163 rows exactly match the training trace.

| Method | Median | Best | At least -4300 | At least -4000 |
|---|---:|---:|---:|---:|
| Corenflos | -4554.67 | -3842.43 | 38/100 | 9/100 |
| Ditlevsen | -4100.34 | -3872.97 | 74/100 | 29/100 |
| IF2, comparable effort | -3755.41 | -3747.82 | 100/100 | 100/100 |
| IFAD-0, comparable effort | -3753.88 | -3748.61 | 100/100 | 100/100 |
| IFAD-0.97, comparable effort | -3745.77 | -3744.04 | 99/100 | 99/100 |
| IFAD-1, comparable effort | -3749.12 | -3745.08 | 96/100 | 96/100 |

Corenflos has a slightly better single best run than Ditlevsen, but its median
is 454 log-likelihood units lower and half as many starts clear -4300. It is
not competitive with IFAD or IF2 on this benchmark.

The completion audit passed with 100 checkpoints, 100 successful final
evaluations, contiguous every-update traces, and no PDF figures. Of the 100
fits, 66 reached the time budget, 32 exhausted the rollback limit, and two
never obtained a valid differentiable-filter evaluation.
