# Shorter NUTS trajectories

This experiment compares maximum depths 2 and 3 with a fresh PMMH control.
All three arms run sequentially on GPU0, with four chains per arm. The model,
prior, 10,000 likelihood particles and four starting parameter vectors are
identical. NUTS retains its main-run covariance and chain-specific final step
sizes; PMMH retains its own main-run covariance and final step sizes. No
adaptation is repeated. Starts are the final parameter states of the completed
primary NUTS run; particle-filter randomness is freshly initialized.

Each arm runs 1,100 iterations. The first 100 are discarded for comparison,
leaving 1,000 draws per chain. The worker reports all 18 bulk/tail ESS values,
ESS per second, R-hat, divergences and trajectory lengths. Compilation and
initial discarded iterations are timed separately from retained sampling.
Previous pilot preparation is excluded. Between-chain differences and low
ESS can make this pilot inconclusive; faster updates alone are not success.
The existing depth-5 results are historical context, not a contemporaneous
control in this three-arm experiment.

`protocol.json` fixes the settings, inputs, source hashes and reporting rules.
`launch.py` prepares, launches and reports the experiment. The preparation
inputs and frozen source are archived under `artifacts/packed`; restore with:

```sh
python artifacts/restore_results.py --prefix code/bayesian/results/short-trajectories-v1
```

With the documented Python environment, `python .../launch.py` launches the
worker and refuses an existing launch. `--report` rebuilds the report after
all arms complete. The worker waits on subprocess exits; it has no polling,
retries or automatic extensions. Logs, checkpoints and generated reports are
ignored. Report results in chat/code only; no automatic manuscript/SI edits.
