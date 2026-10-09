# Ditlevsen comparison for the Dhaka model

The manuscript uses **DS19** as the algorithm name for the Ditlevsen comparison.
Source directory and module names retain their original spelling.

This directory implements the transition-density comparison in SI Section S11.
The Dhaka model is called `Dacca` in the code and data filenames. The final
comparison uses a monthly Gaussian transition approximation, a particle filter
with observation-dependent proposals, backward sampling of state trajectories,
and numerical score ascent. The derivation and differences from the published
SAEM algorithm are in [method.tex](method.tex).

A separate [numerical generalized-SAEM pilot](ditlevsen/block_saem.py) tests
the optimizer modification. It retains the monthly transition model, guided
filter and backward sampler, but replaces averaged-score Adam updates with
numerical maximization of an averaged complete-path objective. The old paths
are reevaluated at each candidate parameter. This requires storing paths and
has increasing cost after burn-in; it does not establish the exponential-family
or convergence assumptions in DS19 for Dhaka. An accepted step increases this
averaged objective, which need not increase the independently evaluated Euler
likelihood. The pilot is excluded from the manuscript comparison.

From the repository root, using the shared environment:

```sh
PYTHONPATH=ditlevsen:../pypomp JAX_PLATFORMS=cpu python -m ditlevsen.block_saem \
  --output ditlevsen/results/block_saem_pilot_start0 --start 0
```

This uses the first saved IF2 starting estimate, 100 particles, eight updates,
three burn-in updates and at most 25 L-BFGS iterations per maximization step.
It evaluates the initial and final estimates with 24 independent Euler-20
filters of 5,000 particles. Seeds, settings, source snapshots, every parameter
update and M-step diagnostics are saved. This short pilot tests feasibility;
it is not a comparison at equal computation time. More particles or internal
integration steps do not remove the monthly Gaussian approximation.

## Find the manuscript results

### Paired optimizer comparison

[saem_ab.py](ditlevsen/saem_ab.py) compares the score implementation with
numerical generalized-SAEM from 20 global starts and 20 saved IF2 starts.
The two arms share each starting estimate, use 1,000 particles and receive
850 seconds of optimization time. The Gaussian transition approximation,
guided proposals and backward sampler are shared. The score arm retains its
archived settings, including learning rates .1/.0005 for global/IF2 starts
and burn-in 30. Numerical SAEM uses the pilot settings: burn-in 3, at most
80 updates, and at most 25 L-BFGS iterations per M-step. Thus this compares
the two optimizer packages, including their schedules.

Each numerical M-step divides the objective and gradient by
`max(1, max(abs(initial_gradient)))`, fixed throughout that M-step. This
preserves the objective's maximizer and its nondecreasing acceptance test.
The original unscaled run encountered false L-BFGS-B convergence at unchanged
parameters. Two saved fixed-path diagnostics reproduced this behavior and
showed substantial objective increases after scaling. The corrected study
is in `results/saem_ab_scaled`; `results/saem_ab` preserves the superseded
run. Compatible completed score fits are reused, including failed fits,
with hashes recorded in `score_reuse.json`. Starts, score settings, particle
counts, time budgets and score-fitting source hashes must match before reuse.

Both arms retain the last finite estimate recorded within the budget. A failed
fit contributes its last eligible estimate, or the input estimate if no update
completed. Computation used to detect a deadline overrun is recorded, but its
candidate is discarded. Per-fit compilation is excluded. The IF2 estimates
are precomputed inputs; their original fitting cost is excluded from both
warm arms, so this experiment does not compare total global-versus-warm cost.

All initial and final estimates receive independent Euler-20 evaluations with
5,000 particles and 36 replicates. Within each pair, evaluation random numbers
are shared to estimate the Monte Carlo error of the difference. These draws
never select a fitting output. For pairs fitted from scratch, the order of
the two fitting methods alternates across pairs within each CPU worker.
Reused score fits retain their recorded fitting times.

```sh
PYTHONPATH=ditlevsen:../pypomp JAX_PLATFORMS=cpu python -m ditlevsen.saem_ab \
  --prepare --output ditlevsen/results/saem_ab_scaled \
  --reuse-score-from ditlevsen/results/saem_ab
# Run worker indices 0, 1, 2, 3 on separate CPU cores, or use --workers 1.
PYTHONPATH=ditlevsen:../pypomp JAX_PLATFORMS=cpu python -m ditlevsen.saem_ab \
  --output ditlevsen/results/saem_ab_scaled --worker 0 --workers 4
```

`protocol.json` freezes settings, source hashes and starting points before
fitting. Workers verify source hashes, save each arm before evaluation, and
can resume completed arms and pairs. Once all pairs are complete,
[saem_ab_report.py](ditlevsen/saem_ab_report.py) creates linear-scale paired
panels, summary tables, Monte Carlo errors and provenance under `report/`.
These outputs require review before being incorporated into the SI.

### Existing S11 results

The SI figures use Ditlevsen with **1,000 fitting particles**, with and without
an IF2 warm start. Those runs are stored under
[corenflos/results/particle_increase_j1000_final_100/](../corenflos/results/particle_increase_j1000_final_100/README.md),
because a shared driver ran both methods. The SI tables also include the
100-particle experiments and the separately tuned 5,000-particle global search.
See [results/README.md](results/README.md) for the complete mapping.

The final paper plots and tables are in
[imgs/competitors/](../imgs/competitors/README.md). Their generators are
[corenflos.settings_plots](../corenflos/corenflos/settings_plots.py) and
[code/competitor_results.py](../code/competitor_results.py). The plotted values
are independently evaluated Euler-20 log-likelihoods, not the Gaussian
approximation's fitting objective. Each final estimate uses 5,000 evaluation
particles and 36 replicates.

## Read or run the code

| Location | Contents |
|---|---|
| [ditlevsen/](ditlevsen/README.md) | Python implementation and module guide |
| [results/](results/README.md) | Final runs, shared IFAD/IF2 inputs, validation, and pilot results |
| [scripts/](scripts/README.md) | Original launch argument lists and evaluation wrappers |
| [tests/](tests/) | Unit and regression tests for filtering, optimization, and result handling |
| [method.tex](method.tex) | Mathematical description of the implemented method |
| [invertibility_note.tex](invertibility_note.tex) | Covariance rank calculation for the Dhaka extension |
| [literature_notes.md](literature_notes.md) | Notes on the source papers and methodological choices |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Historical pilots, tuning, and superseded results |

Start with the [shared reproduction guide](../code/competitor_reproduction.md)
for the environment, sibling Pypomp data dependency, tests, fitting, and plots.
From this directory, after setting up that environment:

```sh
PYTHONPATH=.:../../pypomp python -m ditlevsen.smc_benchmark --help
JAX_PLATFORMS=cpu make test PYTHON="$(command -v python)"
```

`smc_benchmark` is the current fitting entry point. Its `--transition block`
and `--proposal guided` options select the method used in S11. The older
`benchmark`/`ekf` and local-transition `smc` workflows remain for development
history. The bridge kernels are separate diagnostics and are not part of
the reported monthly-block experiment.

The [reference directory](results/reference/README.md) contains the tracked
IF2 initialization and current manuscript reference exports. Re-extracting
those inputs from the original pickles requires the
[bulk-output archive](../artifacts/README.md); ordinary fitting can use the
already exported IF2 starting estimates.
