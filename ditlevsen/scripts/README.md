# Original Ditlevsen launchers

Run context for these files is `dmop/ditlevsen`, with
`PYTHONPATH=.:../../pypomp`. They retain the original local Python interpreter
and result paths. For a new machine, use the `python -m ditlevsen.smc_benchmark`
arguments with your active interpreter and a new output directory, following
the [reproduction guide](../../code/competitor_reproduction.md).

| Script | Purpose |
|---|---|
| [run_if2warm_final100.sh](run_if2warm_final100.sh) | J=100 fit from the shared IF2 estimates, with the 400-update/850.657-second continuation budget |
| [evaluate_if2warm_final100.sh](evaluate_if2warm_final100.sh) | Independent final and per-update evaluation; `IF2WARM_FINAL_ONLY=1` selects final evaluation only |
| [run_if2warm_after_corenflos.sh](run_if2warm_after_corenflos.sh) | Original scheduling wrapper that waits for Corenflos before using the GPU |

The global-search settings are recorded in
[results/block_smc_guided_j100_final_100/configuration.json](../results/block_smc_guided_j100_final_100/configuration.json)
and
[results/block_smc_guided_j5000_final_100/configuration.json](../results/block_smc_guided_j5000_final_100/configuration.json).
There is no standalone global-search launcher in this directory. Use
`ditlevsen.smc_benchmark --help` to map those settings to command-line options.
The J=100 experiment used four worker indices (0–3); its saved JSON records
one of them. The J=5,000 experiment used a single worker and separate tuning.

The J=1,000 experiments are run by
[corenflos.particle_experiment](../../corenflos/corenflos/particle_experiment.py)
and stored in the [combined result directory](../../corenflos/results/particle_increase_j1000_final_100/README.md).
