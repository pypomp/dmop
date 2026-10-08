# Dhaka optimizer A/B experiment

This experiment compares the existing score updates with numerical
generalized-SAEM on the same monthly Gaussian transition model. It contains
20 paired global starts and 20 paired IF2 warm starts: 80 fits in total.
Each fit uses 1,000 particles and an 850-second continuation budget. The
initial and both final estimates in every pair receive independent Euler-20
evaluations with 5,000 particles and 36 replicates.

Pairs are defined within each initialization regime. Global start `i` is
not necessarily the ancestor of IF2 starting estimate `i`; the two input sets
come from the existing experiments. The cost of obtaining the precomputed
IF2 estimates is excluded from both warm arms.

Read [protocol.json](protocol.json) for the frozen settings and source hashes.
[starts.npz](starts.npz) contains the exact input vectors, and [source/](source/)
preserves the Python sources used at launch. These snapshots are an audit
record; run the package from the repository root using the
[reproduction commands](../../README.md#paired-optimizer-comparison).

- `cold/startNNN/` and `warm/startNNN/` hold the two fits, parameter traces,
  independent evaluations and raw evaluation replicates for each pair.
- `score.json` and `saem.json` record termination, updates and timings.
  `*_parameters.npz` retains the selected estimate even when a fit fails.
- A pair's `status.json` becomes complete only after all three evaluations
  finish. Absence of that file means the pair is unfinished.
- `workerN.json` records CPU affinity and preliminary score-kernel compilation.
  `workerN_status.json` records completion of that worker's assigned pairs.
- `report/` is generated only once all 40 pairs are complete. It contains
  paired panels on linear axes, tables, Monte Carlo errors and provenance.

Four workers use CPU groups 0–3, 4–7, 8–11 and 12–15, beginning after the
SPX batch on each group finishes. Their logs are `/tmp/dmop-saem-ab-workerN.log`
on the launch machine. The score arm retains its archived settings, while
SAEM uses the pilot settings; the [protocol](protocol.json) records both.
These outputs require scientific review before incorporation into the SI.
