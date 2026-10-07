# Corenflos comparison for the Dhaka model

This directory implements the differentiable particle filter of Corenflos
et al. (2021) for SI Section S11. It uses entropy-regularized optimal transport
for resampling and Adam for parameter optimization. The code calls the Dhaka
model `Dacca`. [implementation_notes.md](implementation_notes.md) explains the
correspondence with the published FilterFlow implementation.

## Find the manuscript results

The SI figures use Corenflos with **100 fitting particles**, with and without
an IF2 warm start. The SI tables also include the 1,000-particle experiments.
The [result index](results/README.md) identifies all four runs and the
Ditlevsen runs stored here by the shared 1,000-particle driver.

The final S6–S9 PDFs are in
[imgs/competitors/corenflos100_ditlevsen1000/](../imgs/competitors/corenflos100_ditlevsen1000/README.md).
Use `corenflos.settings_plots` to redraw this set; `corenflos.plots` and the
command-line defaults of `corenflos.warm_plots` produce earlier comparisons.
The [combined output guide](../imgs/competitors/README.md) maps the plot files
to the SI and explains their inputs.

Fitting uses the differentiable filter's objective. Final parameter estimates
are evaluated independently with the ordinary Euler-20 particle filter,
using 5,000 particles and 36 replicates. The global searches select the best
valid fitting objective before approximately 700 seconds. Searches initialized
by IF2 use the last valid estimate before the time limit. The independently
evaluated likelihood does not select the fitted parameter estimate.

## Read or run the code

| Location | Contents |
|---|---|
| [corenflos/](corenflos/README.md) | Transport, particle filter, optimization, experiment drivers, and plotting |
| [results/](results/README.md) | Final runs, IF2 baseline evaluations, combined comparisons, and pilots |
| [scripts/](scripts/README.md) | Original fitting and evaluation argument lists |
| [tests/](tests/) | Transport derivatives, filtering, optimizer, pipeline, and plotting tests |
| [implementation_notes.md](implementation_notes.md) | Algorithm and FilterFlow correspondence |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Earlier comparisons and experiment notes |

The [shared reproduction guide](../code/competitor_reproduction.md) gives
setup instructions, test commands, and the complete fitting-to-figure workflow.
Corenflos imports the shared data, model, and initialization helpers from the
sibling `ditlevsen/` directory, and ordinary evaluation uses the sibling
Pypomp source checkout. From this directory, after environment setup:

```sh
PYTHONPATH=.:../ditlevsen:../../pypomp python -m corenflos.benchmark --help
JAX_PLATFORMS=cpu make test PYTHON="$(command -v python)"
```

The four-method `corenflos.particle_experiment` driver runs both methods at
1,000 particles. It uses the tracked 100-particle settings and IF2 initial
estimates, fits and evaluates in serial GPU partitions, and checks all 100
starts. More particles change the number of updates possible within a fixed
time; the original 100-particle Ditlevsen global searches also used different
GPU scheduling. The SI reports update counts alongside likelihoods.

References: [Corenflos et al. (2021)](https://proceedings.mlr.press/v139/corenflos21a.html)
and [FilterFlow](https://github.com/JTT94/filterflow).
