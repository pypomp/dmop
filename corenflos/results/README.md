# Corenflos and combined comparison results

All final experiments below have 100 starts. `J` denotes fitting particles;
final evaluation uses 5,000 particles and 36 replicates of the Euler-20 filter.

| Experiment | Directory | Use in the SI |
|---|---|---|
| Corenflos global search, J=100 | [corenflos_j100_eps025_final_100/](corenflos_j100_eps025_final_100/) | Tables and S6–S9 |
| Corenflos IF2 warm start, J=100 | [if2warm_ifad097_budget_j100_final_100/](if2warm_ifad097_budget_j100_final_100/) | Tables and S6–S9 |
| Both methods, global and IF2 starts, J=1,000 | [particle_increase_j1000_final_100/](particle_increase_j1000_final_100/README.md) | All four in tables; Ditlevsen variants in S6–S9 |
| Evaluation of shared IF2 starting estimates | [ifad097_post_if2_reference/](ifad097_post_if2_reference/) | Baseline distribution and paired changes |
| Combined J=100 exports | [if2warm_ifad097_comparison/](if2warm_ifad097_comparison/) | Paired diagnostic CSV used by the SI table generator; earlier figure set |

The J=100 and J=5,000 Ditlevsen fits live in
[ditlevsen/results/](../../ditlevsen/results/README.md).
The current mixed-particle figure set lives in
[imgs/competitors/](../../imgs/competitors/README.md), not in these older
experiment `figures/` folders.

Each fitting directory contains `configuration.json`, `fit_summary.csv`,
and `final_evaluations.csv`. The JSON records settings and original paths;
the CSVs record the selected estimates and independent evaluations.
Iteration-level traces and checkpoints require the
[bulk-output archive](../../artifacts/README.md). See the
[reproduction guide](../../code/competitor_reproduction.md) for the stages
that read or produce them.

Directories named `preflight_*`, `recovery_*`, `*pilot*`, and
`particle_increase_j1000_preflight/` are tuning or validation runs.
Their results are not substituted for the final runs in the SI. Earlier
comparison figures can use older reference exports and different display
cutoffs; use the current manuscript exports in
[ditlevsen/results/reference/](../../ditlevsen/results/reference/README.md).
