# Original Corenflos launchers

These files preserve the argument lists and scheduling used for the
experiments. They assume the working directory is `dmop/corenflos`, the
`PYTHONPATH` in the [reproduction guide](../../code/competitor_reproduction.md),
and the original `/home/kevin/anaconda3/envs/pypomp/bin/python` interpreter.
For a new machine, use the module arguments with your active `python` and a
new output directory. Several wrappers wait for old jobs or result files;
they are records of the original workflow, not a general installation script.

| Scripts | Purpose |
|---|---|
| [run_final100.sh](run_final100.sh), [evaluate_final100.sh](evaluate_final100.sh) | J=100 global fits and independent evaluations |
| [run_if2warm_final100.sh](run_if2warm_final100.sh), [evaluate_if2warm_final100.sh](evaluate_if2warm_final100.sh) | J=100 IF2-initialized fits and evaluations |
| [evaluate_if2warm_starts.sh](evaluate_if2warm_starts.sh) | Evaluate the shared IF2 starting estimates |
| [evaluate_if2warm_pipeline.sh](evaluate_if2warm_pipeline.sh) | Wait for Ditlevsen completion, evaluate both methods, and draw/audit the earlier J=100 comparison |
| [run_if2warm_corrected_final_pipeline.sh](run_if2warm_corrected_final_pipeline.sh), [run_if2warm_corrected_pilot10.sh](run_if2warm_corrected_pilot10.sh) | Original rerun scheduling after correcting the IF2 input export |
| [run_pilot10_after_tuning.sh](run_pilot10_after_tuning.sh), [evaluate_pilot10.sh](evaluate_pilot10.sh) | Ten-start pilot |
| [run_j250_after_eps025.sh](run_j250_after_eps025.sh), [evaluate_tuning_after_j250.sh](evaluate_tuning_after_j250.sh), [run_recovery_start4.sh](run_recovery_start4.sh) | Tuning and recovery diagnostics |

The current four-method J=1,000 pipeline is
[corenflos.particle_experiment](../corenflos/particle_experiment.py).
The current SI plot generator is
[corenflos.settings_plots](../corenflos/settings_plots.py).
See the [result index](../results/README.md) for final output directories.
