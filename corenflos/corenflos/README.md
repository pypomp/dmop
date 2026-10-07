# Corenflos Python modules

Run modules with `python -m corenflos.<module>` from the parent directory,
using the [shared environment](../../code/competitor_reproduction.md).
This inner directory is the Python package; experiments and documentation
live in the outer `corenflos/` directory.

| Module | Role |
|---|---|
| [transport.py](transport.py) | Sinkhorn transport resampling and its derivative |
| [dpf.py](dpf.py) | Differentiable particle filter using the Dhaka Euler simulator |
| [fit.py](fit.py) | Adam optimization, numerical checks, and restarts |
| [benchmark.py](benchmark.py) | Resumable fit/final-eval/trace-eval driver |
| [particle_experiment.py](particle_experiment.py) | Four-method 1,000-particle fitting, evaluation, and completion audit |
| [settings_plots.py](settings_plots.py) | Current S6–S9 figure generator, mixing Corenflos J=100 and Ditlevsen J=1,000 |
| [warm_plots.py](warm_plots.py) | Shared plot helpers; CLI produces the earlier same-particle-count warm-start comparisons |
| [plots.py](plots.py) | Earlier global-search comparisons and shared plot helpers |
| [audit.py](audit.py), [warm_audit.py](warm_audit.py) | Original and warm-start result checks |
| [particle_profile.py](particle_profile.py) | Particle-count timing/profile experiment |

The package reuses data and parameter helpers from
[ditlevsen/ditlevsen/](../../ditlevsen/ditlevsen/README.md).
The [result index](../results/README.md) distinguishes final runs from pilots.
Final SI files are in [imgs/competitors/](../../imgs/competitors/README.md).
