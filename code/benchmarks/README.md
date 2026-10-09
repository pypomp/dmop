# Comparisons across models

This directory is the working source for the requested cross-model comparison.
Experiments are under development; no results or claim of saturation have yet
been added to the manuscript. Existing Daphnia and Dhaka results remain in
[../daphnia/](../daphnia/README.md) and
[../../imgs/competitors/](../../imgs/competitors/README.md).

## Where to start

| File | Purpose |
|---|---|
| [models.py](models.py) | Gaussian examples, analytic likelihoods, and the original Pypomp SPX model |
| [ctdd.py](ctdd.py) | CTDD21 resampling around the existing Pypomp simulator and measurement functions |
| [smoothing.py](smoothing.py) | Small-model transition densities, particle smoothing, and DS19 score updates |
| [saem.py](saem.py) | Averaged Gaussian sufficient statistics and SAEM maximization steps |
| [daphnia.py](daphnia.py) | Original S10 Euler evaluation and the Gaussian block approximation for DS19 |
| [run.py](run.py) | Repeated oscillator, linear Gaussian, and SPX fits |
| [run_daphnia.py](run_daphnia.py) | MPIF, IFAD, DS19, and CTDD21 from common Daphnia starts |
| [report.py](report.py) | Combine completed final runs and the archived Dhaka results into panels and a table |
| [finish_report.py](finish_report.py) | Generate the complete report after declared final batches finish |
| [matched_particles.py](matched_particles.py) | Gaussian DS19, IFAD and IF2 runs with equal particle counts, at 100 and 500 |
| [tune_particles.py](tune_particles.py) | Select the shared Gaussian particle count on separate IFAD tuning starts |
| [export_dhaka.py](export_dhaka.py) | Recover MC errors for the exact archived IFAD/IF2 estimates used in the manuscript |
| [queue_final.py](queue_final.py) | Wait for tuning and free assigned cores, then run a declared final batch |
| [recover.py](recover.py) | Retain completed starts and recover an interrupted batch, preserving the original files |
| [tests/](tests/) | Independent checks of likelihoods, scores, model adapters, and report completeness |

The transition-density and transport implementations for the existing Dhaka
study remain in [../../ditlevsen/](../../ditlevsen/README.md) and
[../../corenflos/](../../corenflos/README.md). The original Daphnia model and
50-start experiment remain in [../daphnia/](../daphnia/README.md).

## Scope

Compare IFAD, IF2 (MPIF for the panel), DS19, and CTDD21 on:

1. The harmonic oscillator in Ditlevsen and Samson (2019), Section 6.1:
   1,000 position observations, interval 0.02, true (D, gamma, sigma) =
   (4, 0.5, 0.5). Simulate the continuous model exactly; fit the published
   order-1.5 Gaussian approximation and evaluate its likelihood by Kalman
   filtering. Condition on the first position. All particle methods use the
   conditional transition for the unobserved velocity, without adding
   observation noise. This requires transition-density information even for
   the IFAD and IF2 implementation of this example.
2. The two-dimensional linear Gaussian model in Corenflos et al. (2021),
   Section 5.1 and Supplement E.1: transition diag(theta), process covariance
   0.5 I, measurement covariance 0.1 I, 150 observations, true theta=(0.5,0.5).
   Specify the initial law and use it for simulation, fitting and evaluation.
   This is model-parameter estimation; their proposal-learning, music and
   robot-localization experiments optimize different targets and are outside
   this comparison.
3. **The existing `pypomp.models.spx()`**, with its bundled data, parameter
   transformation, covariate interpolation and boundary rule. Do not
   substitute another stochastic volatility model. Record the Pypomp revision
   and the data hash. The scripts in `quant/tests/spx` provide earlier IF2
   settings, not completed four-method results.
4. The 38-parameter Daphnia model in SI S10, retaining its original model,
   data, numerical boundaries, and starting-point distribution.
5. Dhaka, using the existing independently evaluated S11 experiments.

## Reporting protocol

- Use shared starts and separate tuning starts. Record every start and failed
  fit. Do not select or omit runs to obtain a particular ranking.
- Evaluate all fitted parameter vectors with the same likelihood within a
  model. Use analytic likelihoods for the two Gaussian examples; ordinary
  independent particle filters for SPX, Daphnia, and Dhaka. Report Monte Carlo
  uncertainty for the latter.
- Keep fitting objectives separate from evaluation likelihoods. Choose the
  final checkpoint by a prespecified rule, not the reported evaluation draws.
- Report convergence and final likelihood distributions in consistent panels;
  give best and median likelihood, spread, failures, and runtime in one table.
  Plot log-likelihood on ordinary linear axes, with a separate range for each
  model. Do not compare likelihood values between datasets. A reference line
  marks the analytic maximum or the best displayed final estimate.
  Progress curves show medians, with `fill_between` from the 10th to 100th
  percentiles at alpha 0.10, matching the existing Dhaka SI plots.
  `optimization.pdf` retains the full range of the median trajectories;
  `optimization_detail.pdf` shows the same curves near the final estimates,
  also on linear axes. `plot_ranges.json` records the limits and their rule.
  All final estimates, including outliers, appear in `final_likelihoods.pdf`.
- Measure synchronized wall time, separate compilation and evaluation, and
  record hardware and concurrency. Existing V100 times and new RTX 3090 times
  cannot establish a runtime ranking. Rerun a common timing experiment before
  drawing that conclusion.
- Describe DS19 extensions explicitly. SMC with a numerical score update is
  not the original paper's SAEM M-step. The Gaussian examples use SAEM;
  SPX, Daphnia, and Dhaka use the numerical-score extension. Daphnia and SPX require additional
  transition-density work: SPX includes the atom created by its variance
  floor; Daphnia uses the Gaussian block approximation described below.
- Whether the small benchmarks are saturated is an empirical question. The
  prose must follow the completed experiments, including exceptions.

## Environment

From the repository root, use the environment in
[../competitor_reproduction.md](../competitor_reproduction.md):

```sh
export PYTHONPATH="code/benchmarks:corenflos:ditlevsen:../pypomp"
JAX_PLATFORMS=cpu python -m pytest code/benchmarks/tests
```

The new runs use Python 3.13, JAX 0.9.0.1 with 64-bit arithmetic, NumPy 2.4.2,
SciPy 1.17, and Pypomp source revision
`d231517c4e84d2b270b7ba2a9461083cc0973ed3`. Daphnia also requires `xlrd`
(tested with 2.0.2) to read the original Excel data. On this machine it is
installed in `/tmp/dmop-benchmark-deps`; that path is an environment detail,
not a required location. Add it to PYTHONPATH if using that temporary install.

## Running and reading results

Each output directory contains a configuration with seeds, source hashes,
particle counts, transformations, and CPU affinity. Existing directories
are never overwritten. `status.json` distinguishes partial and complete runs;
`purpose` distinguishes pilots from final experiments. Parameter traces are
saved before independent evaluation. `checkpoints.csv` contains evaluation
likelihoods; `fitting_objectives.csv` contains optimization objectives. They
must not be used interchangeably. `evaluation_replicates.csv` retains the
individual likelihood estimates used for Monte Carlo errors.
Refitting a time-limited search can complete a different number of updates
on a different machine. Saved estimates and evaluations make the figures
reproducible without refitting; seeds alone do not reproduce a wall-time stop.

For an interrupted SPX or Daphnia batch, `recover.py --model MODEL --folder
PATH` retains all completely evaluated starts, including unsuccessful fits,
and reruns the unfinished suffix with the same seeds and fitting settings.
It copies the original directory to `results/interrupted/` before doing any
work. The suffix has its own configuration and source snapshot under
`results/recovery-*`; `recovery.json` in the combined directory records the
provenance of both parts. The final status is written only after checking
that every expected start and method has exactly one final evaluation.
Incomplete fits are archived even when the whole start must be rerun.
Run long experiments as persistent services; terminal sessions did not
survive the October 8 interruption. Never start a second writer to a batch.

For example, a short workflow check is:

```sh
JAX_PLATFORMS=cpu python code/benchmarks/run.py --model linear \
  --starts 2 --iterations 10 --warm 5 --ds-update saem --output /tmp/dmop-linear-check
```

`run.py --start-index` supports disjoint batches of a prespecified set of
starts. The starting vectors and random streams do not depend on the batch
boundary or on which other methods run. Run each method serially within a
batch. On the i9-13900K used here, affinity groups 0–3, 4–7, 8–11, and 12–15
each contain two performance cores with their hardware threads. Concurrent
batches use disjoint groups. These are shared-machine timings, not isolated
whole-machine benchmarks. The initial Daphnia GPU pilot was interrupted when another GPU workload
started; CPU tuning is in progress on cores 16–19. The older
Dhaka timing convention is retained and must not be compared directly with
these new per-fit measurements.

The reporting command is:

```sh
python code/benchmarks/report.py
```

It requires all 20 final starts for all four methods on each new model. It
rejects incomplete directories, duplicate starts, missing final evaluations,
and nonconverged analytic reference fits. `--models` and `--output` allow
checking a completed subset in a separate directory during development.

## Gaussian SAEM

For the oscillator, we use 80 SAEM iterations, a gain of one
for the first 30 iterations, then `(m-30)^(-0.9)`, following DS19's example.
Each iteration draws a smoothed state path and updates the sufficient
statistics of the Gaussian complete-data likelihood. Its maximization step
uses L-BFGS-B. For the linear Gaussian example the corresponding maximization
is explicit. The initial distributions and parameter bounds are the same as
for the other methods. The tests compare sufficient-statistic likelihoods and
derivatives against direct complete-path calculations.

DS19, IFAD and IF2 use equal particle counts in the manuscript's Gaussian
panels. `matched_particles.py` runs both 100 and 500 particles on the same
20 final starts (seed 631450) and CPU group. `tune_particles.py` selects one
common count using four separate IFAD starts per model (seed 2026100901):
fewest failed fits, then smallest sum of median deficits from the two analytic
maxima, then fitting time. The SI reports the selected common setting; the
other configuration remains in the repository. `report.py` reads this
selection by default and rejects unequal IFAD/IF2/DS19 particle counts.
CTDD21 retains its 25-particle configuration and original timing hardware.

The Gaussian DS19 timing includes the particle filter, backward simulation,
sufficient-statistic update and M-step. DS19 uses 80 iterations; IFAD uses
100 IF2 plus 300 gradient updates; IF2 uses 600 updates; CTDD21 uses 300
gradient updates. Iteration counts differ even at equal particle counts.
The linear Gaussian M-step is explicit;
the oscillator's three-parameter M-step works with sufficient statistics.
Compilation and final analytic likelihood evaluations are excluded for all
methods. These timings do not establish a general runtime ordering.

The first Gaussian runs used the numerical-score extension. Those traces and
the subsequent `final-*-saem` runs remain as implementation history. The main
report now uses `matched-*-jCOUNT-if` for IFAD/IF2 and `matched-*-jCOUNT-ds`
for DS19. Only CTDD21 comes from the original Gaussian `final-*` directories.

## Daphnia approximation

CTDD21 transports the eight biological states and resets the interval error
accumulator. Propagation, the day-4 inoculation, and negative-binomial
measurements use the S10 code. DS19 composes local strong-order-1.5 Gaussian
moments over 24 Euler-grid steps for the first observation interval and 20
thereafter. A relative diagonal floor of `1e-8` is applied after scaling the
states by `(3,1,3,1,1,1,16,25)`. Proposals condition on a Gaussian approximation
to the four observed counts, with their negative-binomial variances evaluated
at the predicted counts. Weights include the transition/proposal ratio and
the original measurement density. Out-of-range Gaussian proposals receive
zero weight. This boundary treatment differs from the original Euler model.

All Daphnia final estimates are therefore evaluated using the original Euler
particle filter. Likelihood estimates are averaged within each unit before
taking logs and summing across units. The reported MC error combines the
independent unit errors. DS19 and CTDD21 start at the same MPIF estimate as
IFAD and receive its measured continuation time; the cost of the warm start
is included for each method. A last update that exceeds the budget is timed
but does not replace the last eligible estimate. Pypomp stage times are
distributed uniformly across their saved iterates for the progress plots;
DS19 and CTDD21 updates are timed individually.

Final Daphnia estimates use 10 evaluation replicates with 2,000 particles per
unit. Progress plots use two replicates with 500 particles, every 50 updates
and at the final checkpoint; these noisier evaluations never select estimates.
Early tuning runs used the final-evaluation settings for intermediate iterates
as well. Each run records its actual settings and source snapshot.

Source papers: [DS19](https://arxiv.org/abs/1707.04235),
[CTDD21](https://proceedings.mlr.press/v139/corenflos21a.html).
