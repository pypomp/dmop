# Prior sensitivity

This comparison doubles the standard deviations of the common normal prior
in the 18 estimation coordinates. Prior centers and the coordinate
transformation remain fixed. The likelihood is untempered and uses 10,000
particles, as in `../gpu-comparison-v2`. No IF2 parameter cloud defines
either prior.

The four PMMH chains use the primary comparison's starts, fixed covariance,
500 warmup updates and 10,000 retained draws, with fresh seeds. The protocol
and source/input hashes are saved before launch. To avoid adding another
concurrent GPU job, launch only after the existing 5,000-particle PMMH
continuation finishes:

```sh
/home/kevin/anaconda3/envs/pypomp/bin/python code/bayesian/corrected/results/prior-sensitivity-gpu-v1/launch.py --check-only
/home/kevin/anaconda3/envs/pypomp/bin/python code/bayesian/corrected/results/prior-sensitivity-gpu-v1/launch.py
```

Run these commands from the repository root with host process visibility.
The launcher checks completion, process identity, source and inputs, and
refuses to overwrite an existing run. No background scheduler is installed.

After both settings have converged, compare all parameter marginals, means,
standard deviations and 95% intervals, accounting for Monte Carlo error.
The long lower tails of omega5 and omega6 in the initial PMMH review deserve
particular attention. Report any material changes. A narrow sampled
distribution relative to one prior does not by itself establish that the
prior has little effect, and one scale comparison cannot establish
insensitivity to every prior choice.
