# Figures included in SI Section S11

This directory combines both initialization experiments for Corenflos at
100 fitting particles and Ditlevsen at 1,000. Each method has 100 searches;
IFAD-0.97 and the shared IF2 estimates provide reference distributions.

| SI figure | PDF included by `si.tex` |
|---|---|
| S6: final likelihoods | [likelihood_if2warm_comparison_r20.pdf](likelihood_if2warm_comparison_r20.pdf) |
| S7: optimization progress | [optimization_if2warm_elapsed_full_r20.pdf](optimization_if2warm_elapsed_full_r20.pdf) |
| S8: IF2 continuation | [optimization_if2warm_elapsed_r20.pdf](optimization_if2warm_elapsed_r20.pdf) |
| S9: parameter estimates | [parameter_if2warm_comparison_r20.pdf](parameter_if2warm_comparison_r20.pdf) |

Each PDF has a PNG counterpart. `likelihood_all_outputs_r20` and
`optimization_all_values_r20` show the complete likelihood range.
`objective_mismatch_if2warm_medians_r20` and
`objective_mismatch_if2warm_paired_r20` are diagnostic figures, not extra SI
figures. Overview axis limits do not remove runs before calculating summaries.

| Numerical export | Contents |
|---|---|
| [settings.csv](settings.csv) | Fitting counts, learning rates, budgets, summary values, and source directories |
| [settings.json](settings.json) | Full fitting configurations |
| [final_all_runs.csv](final_all_runs.csv) | Final evaluations for the four competitor variants |
| [optimization_summary.csv](optimization_summary.csv) | Plotted trajectory summaries |
| [objective_mismatch.csv](objective_mismatch.csv) | Paired fitting-objective and Euler-likelihood changes |

The generator is
[corenflos.settings_plots](../../../corenflos/corenflos/settings_plots.py).
It reads the original result directories, including archived trace CSVs;
these compact exports alone are not its complete input set. The
[parent guide](../README.md) gives settings, plotting commands, and archive
requirements. Table generation is separate and uses tracked CSVs only.
