# Ditlevsen results

Paths in this table are relative to this directory. Each final run has 100
starting points. `J` is the fitting particle count; final evaluation always
uses 5,000 particles and 36 replicates of the Euler-20 filter.

| Experiment | Directory | Use in the SI |
|---|---|---|
| Global search, J=100 | [block_smc_guided_j100_final_100/](block_smc_guided_j100_final_100/) | Results table |
| IF2 warm start, J=100 | [block_smc_guided_j100_if2warm_ifad097_budget_final_100/](block_smc_guided_j100_if2warm_ifad097_budget_final_100/) | Results and paired-change tables |
| Global search, J=1,000 | [../../corenflos/results/particle_increase_j1000_final_100/ditlevsen_vanilla/](../../corenflos/results/particle_increase_j1000_final_100/ditlevsen_vanilla/) | Tables and all comparison figures |
| IF2 warm start, J=1,000 | [../../corenflos/results/particle_increase_j1000_final_100/ditlevsen_warm/](../../corenflos/results/particle_increase_j1000_final_100/ditlevsen_warm/) | Tables and all comparison figures |
| Global search, J=5,000 | [block_smc_guided_j5000_final_100/](block_smc_guided_j5000_final_100/) | Results table; separately tuned experiment |
| IFAD/IF2 reference inputs | [reference/](reference/README.md) | Shared initialization and manuscript reference distributions |

## Files within a final run

- `configuration.json`: fitting settings and original paths.
- `fit_summary.csv`: one row per start, including selected parameters, timing,
  and number of completed updates.
- `final_evaluations.csv`: independent evaluation of each selected estimate.
- `evaluation_configuration.json`, where present: separate evaluation settings.
- `trace_evaluations.csv`, `traces/`, and `checkpoints/`: iteration-level
  information and resumable state, generally in the bulk archive rather than Git.
- `figures/`: experiment-specific plots, which may predate the combined SI set.

The [combined SI output directory](../../imgs/competitors/README.md) is the
place to find the figures included by `si.tex`. The table generator reads
compact tracked CSVs; the full plotting and audit workflows also need
[archived traces](../../artifacts/README.md).

## Development results

`saem_ab/` is the paired optimizer comparison requested after the two pilots
below. It has 20 global and 20 IF2 starting points, two optimizer arms, and a
frozen protocol. Read its `protocol.json` and pair-level `status.json` files
before treating the experiment as complete. The
[reproduction guide](../README.md#paired-optimizer-comparison) describes the
budgets, settings and output-selection rules. This is separate from the
existing S11 results in the table above.

`block_saem_pilot_start0/` and `block_saem_pilot_start1/` test numerical
generalized-SAEM from two saved IF2 estimates, using the same monthly Gaussian
transition model as S11. Each uses eight updates and evaluates its initial and
final estimates with 5,000 Euler particles and 24 replicates. These are short
feasibility checks, excluded from the SI results above. The
[module guide](../README.md) describes the objective and reproduction command.
Their status files distinguish completed fits from completed evaluations.
Both pilots completed. Euler log-likelihood changed from -3769.754 to
-3769.846 for start 0 and from -3761.260 to -3760.267 for start 1. Monte Carlo
standard errors were .151/.141 and .158/.228, respectively. All eight M-steps
in each run retained or increased their fixed averaged complete-data objective;
four M-steps in each run reached the numerical convergence criterion. More
starts and iterations are needed to assess estimation performance. These
pilots leave the transition approximation unchanged.

Other directories preserve validation and tuning. `validation/` contains
numerical and oscillator checks; `particle_sweep*`, `optimizer_tuning*`, and
`j5000_*` contain tuning and timing. `block_smc_guided_100/` stopped at 50
starts, and `block_smc_10/`, `smc_benchmark*`, and `*preflight*` are pilots or
diagnostics. `withdrawn_qml/` contains superseded EKF/QML results.
They are not additional final 100-start comparisons. The
[historical notes](../DEVELOPMENT.md) explain their chronology.

The 100-particle global run used four concurrent workers. Its JSON retains
one worker index, so replaying that index alone reproduces only one partition.
The 5,000-particle run also changes the learning rate, SA schedule, and
likelihood-guard particle count; it is not a particle-count-only experiment.
