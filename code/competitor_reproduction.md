# Reproducing the Ditlevsen and Corenflos comparisons

The scientific methods and experimental design are in SI Section S11.
The [Ditlevsen guide](../ditlevsen/README.md) and
[Corenflos guide](../corenflos/README.md) map the implementation files.
This guide covers the shared environment and the steps between fitting and
manuscript output. Commands below use `python` from your active environment.

## What a fresh clone can do

| Task | Entry point | Inputs beyond the dmop clone |
|---|---|---|
| Rebuild SI tables | `python code/competitor_results.py` from dmop | NumPy and pandas |
| Compile existing figures into SI | `si.tex` | A working project LaTeX toolchain |
| Refit comparisons | Method benchmark modules or the four-method pipeline below | Python environment, sibling Pypomp checkout, GPU for the recorded time budgets |
| Regenerate all S6–S9 plots | `corenflos.settings_plots` | Environment below and restored per-iteration trace CSVs |
| Audit all stored iterations | `corenflos.warm_audit` or pipeline audit | Restored checkpoints and evaluations |
| Re-extract the IF2 initialization | `ditlevsen.warm_starts` | Archived main-experiment pickles; the exported initialization is already tracked |

Final CSVs, configuration files, IF2 starting estimates, and manuscript PDFs
are tracked. Full checkpoints and traces are described in
[artifacts/README.md](../artifacts/README.md). That archive currently requires
access from the maintainer; a public archive for the revision remains to be
published. It is unnecessary for rebuilding the SI tables or using the saved
figure PDFs.

## Environment and directory layout

The data loader expects a Pypomp source checkout beside dmop:

```text
parent/
  dmop/
    ditlevsen/
    corenflos/
  pypomp/
    pypomp/data/dacca/
```

Installing a Pypomp wheel alone does not supply this expected sibling path.
The methods share `ditlevsen.data` and `ditlevsen.model`; Corenflos imports
these for the data, parameter transformations, and starting points. Ordinary
Euler-20 evaluations use Pypomp. Fig. 1A is self-contained and does not use
this checkout.

The October 7 documentation checks used Python 3.13.12 and the clean Pypomp
checkout `d231517c4e84d2b270b7ba2a9461083cc0973ed3`, with these installed versions:

| Package | Version |
|---|---|
| JAX / jaxlib | 0.9.0.1 / 0.9.0.1 |
| NumPy | 2.4.2 |
| pandas | 3.0.0 |
| SciPy | 1.17.0 |
| Matplotlib | 3.10.8 |
| plotnine | 0.15.3 |
| pytest | 9.0.2 |

This records the environment checked for these instructions, not a recovered
lockfile for every historical run. The sibling source revision identifies the
Pypomp code; the editable installation's package-version metadata may be stale.
The separate `global_search/requirements.txt` describes the main Dhaka search
environment and is not the environment specification for these comparisons.

For a new CPU environment, clone Pypomp into the sibling location, check out
the revision above, and from dmop install:

```sh
python -m pip install -e ../pypomp
python -m pip install 'jax==0.9.0.1' 'jaxlib==0.9.0.1' \
  'numpy==2.4.2' 'pandas==3.0.0' 'scipy==1.17.0' \
  'matplotlib==3.10.8' 'plotnine==0.15.3' 'pytest==9.0.2'
```

GPU fitting additionally requires a JAX GPU installation compatible with the
host driver. The recorded experiments used an RTX 3090; CPU runs are useful
for checks but do not reproduce the wall-clock comparison. Full numerical
results can vary with hardware and JAX compilation.

From `dmop/corenflos`, set:

```sh
export PYTHONPATH=.:../ditlevsen:../../pypomp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
python -m corenflos.benchmark --help
python -m ditlevsen.smc_benchmark --help
```

From `dmop/ditlevsen`, use `PYTHONPATH=.:../../pypomp` instead.
For CPU-only checks, set `JAX_PLATFORMS=cpu`. If a locally installed CUDA
plugin probes unavailable hardware even for those checks, set
`JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1` as well.

## Tests

From dmop, use an explicit interpreter to override the Makefiles' original
machine-specific default:

```sh
JAX_PLATFORMS=cpu make -C ditlevsen test PYTHON="$(command -v python)"
JAX_PLATFORMS=cpu make -C corenflos test PYTHON="$(command -v python)"
```

The tests cover the Gaussian approximation and oscillator check, filtering,
transport derivatives, warm-start inputs, optimizer behavior, and the plotting
and experiment drivers. They do not rerun the complete 100-start experiments.

## Fitting and evaluation

The [result indexes](../corenflos/results/README.md) identify the exact
configurations used for the paper. Each `configuration.json` records fitting
settings; `evaluation_configuration.json`, where present, records separate
evaluation settings. Some recorded paths are absolute paths on the original
machine. Keep those files as provenance and use a new output directory for
new runs. Changing an existing experiment's settings is not a supported resume.

For the 1,000-particle comparison, the driver reads all four tracked
100-particle configurations, replaces the particle count and output paths,
and runs the standalone and IF2-initialized variants of both methods.
From `dmop/corenflos`, with the environment above and a working CUDA backend:

```sh
python -m corenflos.particle_experiment --particles 1000 \
  --output results/reproduction_j1000
```

This is a full GPU experiment, not a redraw. It runs ten serial partitions
of ten starts, evaluates each selected estimate with 5,000 particles and
36 replicates, then evaluates the saved iterates with 5,000 particles and
one replicate. It resumes completed work and audits coverage at the end.
The driver explicitly requests CUDA. Its plots are the per-particle-count
comparison; the mixed settings used in S6–S9 are drawn separately below.

For 100-particle fits, the original argument lists are in the
[Corenflos launchers](../corenflos/scripts/README.md) and
[Ditlevsen launchers](../ditlevsen/scripts/README.md).
Those shell files retain their original local interpreter and output paths.
Use their `python -m ...` argument lists with your active interpreter and a
new output directory. The Ditlevsen standalone 100-particle configuration
records four workers; reproducing its scheduling requires worker indices
0, 1, 2, and 3, not just the single worker index saved in the JSON.
The separate 5,000-particle Ditlevsen experiment has its own configuration,
including different tuning and a 5,000-particle likelihood guard.

Both benchmark modules expose `fit`, `final-eval`, and `trace-eval` stages.
Use the same fitting arguments and output directory when requesting the
later stages. Final evaluations use `--eval-particles 5000 --eval-replicates 36`;
traces use `--trace-eval-particles 5000 --trace-eval-replicates 1`.
Final Euler likelihoods are independent evaluations, not the objectives used
to choose the parameter estimates during fitting.

## Tables and figures

From dmop, regenerate Tables S4–S5 using only tracked CSVs:

```sh
python code/competitor_results.py
```

For the exact S6–S9 figure set, restore the trace inputs for the recorded
result directories before running, from `dmop/corenflos`:

```sh
JAX_PLATFORMS=cpu python -m corenflos.settings_plots \
  --output /tmp/dmop-comparison-figures
```

The defaults select Corenflos with 100 particles and Ditlevsen with 1,000,
for both initialization schemes. Omit `--output` to replace the repository's
figure set. The generator reads the recorded result directories, not a new
`reproduction_j1000` directory. See the
[figure guide](../imgs/competitors/README.md) for the source mapping, filenames,
axis limits, and tracked numerical exports. Use `settings_plots` for the SI;
older `plots` and `warm_plots` command-line defaults reproduce historical views.
