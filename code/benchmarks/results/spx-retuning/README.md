# SPX IFAD tuning, October 15, 2026

All 28 tuning fits are complete. The selected recipe is the original one:
alpha .97, initial Adam rate .001, 100 IF2 warm-start iterations, then
300 gradient updates with cosine decay to .0001. Every fit used 500 particles.

The six alpha/rate combinations used the same four independent tuning starts
(seed 2026101501). The winning combination was also tried with 300 warm-start
iterations. `protocol.json` records the plan, and `selection.json` records
every candidate. Selection uses failures, median final evaluation likelihood,
then median fitting time. Tuning evaluations use eight independent filters
with 2,000 particles; they are not publication estimates.

| Alpha | Initial rate | Warm start | Median log-likelihood | Median fitting seconds |
|---|---|---|---|---|
| .97 | .001 | 100 | 11839.017 | 186.0 |
| .97 | .01 | 100 | 11719.262 | 192.0 |
| .995 | .001 | 100 | 11837.960 | 198.8 |
| .995 | .01 | 100 | 11678.558 | 206.5 |
| 1 | .001 | 100 | 11838.020 | 190.7 |
| 1 | .01 | 100 | 11660.526 | 198.0 |
| .97 | .001 | 300 | 11837.870 | 309.5 |

These four-start results do not isolate gradient bias from variance or prove
that the selected recipe is optimal. Larger alpha and the longer warm start
did not improve the tuning median. The tenfold larger rate was markedly
worse. The original implementation already used learning-rate annealing.

## Final results and verification

The planned final refit was unnecessary because tuning selected the existing
recipe. We retain **all 20 original IFAD fits** in `../final-spx-{00,05,10,15}`,
alongside the original IF2, DS19 and CTDD21 fits. No estimate was selected
using final evaluation results. Their 24 independent evaluations with 5,000
particles, original timings and source provenance are unchanged.

`reuse_final.json` hashes the reused inputs. `../../audit_spx_retuning.py`
checks the selected settings, model, data, seeds, Pypomp revision and raw
evaluation summaries. One older runner version differs only through the
addition of Gaussian DS19 SAEM branches; the IFAD computation is unchanged.
The inspected difference is saved in `reviewed_runner_difference.patch`.
`audit.json` records checks of all 28 tuning fits and all 80 original final
method/start combinations. `status.json` explicitly marks the final fits
as reused. The final SPX medians remain 11838.633 (IFAD), 11841.959 (IF2),
11844.828 (DS19), and 11796.975 (CTDD21).

The automatic driver had begun redundant final repeats when the selection
finished. These were stopped; their partial files are retained under
`abandoned-repeat/`, including a brief accidental restart during the source
audit. Neither set contributes to any figure, table or selection. This
departure from the planned refit is recorded in `reuse_final.json`.

Recheck without fitting:

```sh
python code/benchmarks/audit_spx_retuning.py
python code/benchmarks/retune_spx.py
python code/benchmarks/report.py
```

The second command recognizes the completed reuse record, verifies the
report inputs and exits. The third regenerates the publication outputs.
