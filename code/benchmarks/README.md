# Comparisons across models

This directory is the working source for the requested cross-model comparison.
Experiments are under development; no results or claim of saturation have yet
been added to the manuscript. Existing Daphnia and Dhaka results remain in
[../daphnia/](../daphnia/README.md) and
[../../imgs/competitors/](../../imgs/competitors/README.md).

## Scope

Compare IFAD, IF2 (MPIF for the panel), DS19, and CTDD21 on:

1. The harmonic oscillator in Ditlevsen and Samson (2019), Section 6.1:
   1,000 position observations, interval 0.02, true (D, gamma, sigma) =
   (4, 0.5, 0.5). Simulate the continuous model exactly; fit the published
   order-1.5 Gaussian approximation and evaluate its likelihood by Kalman
   filtering. Condition on the first position. All particle methods use the
   conditional transition for the unobserved velocity, without adding
   observation noise. This requires transition-density information even for
   the IFAD and IF2 implementation of this example.
2. The two-dimensional linear Gaussian model in Corenflos et al. (2021),
   Section 5.1 and Supplement E.1: transition diag(theta), process covariance
   0.5 I, measurement covariance 0.1 I, 150 observations, true theta=(0.5,0.5).
   Specify the initial law and use it for simulation, fitting and evaluation.
   This is model-parameter estimation; their proposal-learning, music and
   robot-localization experiments optimize different targets and are outside
   this comparison.
3. **The existing `pypomp.models.spx()`**, with its bundled data, parameter
   transformation, covariate interpolation and boundary rule. Do not
   substitute another stochastic volatility model. Record the Pypomp revision
   and the data hash. The scripts in `quant/tests/spx` provide earlier IF2
   settings, not completed four-method results.
4. The 38-parameter Daphnia model in SI S10, retaining its original model,
   data, numerical boundaries, and starting-point distribution.
5. Dhaka, using the existing independently evaluated S11 experiments.

## Reporting protocol

- Use shared starts and separate tuning starts. Record every start and failed
  fit. Do not select or omit runs to obtain a particular ranking.
- Evaluate all fitted parameter vectors with the same likelihood within a
  model. Use analytic likelihoods for the two Gaussian examples; ordinary
  independent particle filters for SPX, Daphnia, and Dhaka. Report Monte Carlo
  uncertainty for the latter.
- Keep fitting objectives separate from evaluation likelihoods. Choose the
  final checkpoint by a prespecified rule, not the reported evaluation draws.
- Report convergence and final likelihood distributions in consistent panels;
  give best and median likelihood, spread, failures, and runtime in one table.
  Do not compare raw likelihoods between datasets. Show deficits from the
  analytic maximum or a declared model-specific reference.
- Measure synchronized wall time, separate compilation and evaluation, and
  record hardware and concurrency. Existing V100 times and new RTX 3090 times
  cannot establish a runtime ranking. Rerun a common timing experiment before
  drawing that conclusion.
- Describe DS19 extensions explicitly. SMC with a numerical score update is
  not the original paper's SAEM M-step. Daphnia and SPX require additional
  transition-density work and validation before a DS19 result is reported.
- Whether the small benchmarks are saturated is an empirical question. The
  prose must follow the completed experiments, including exceptions.

## Environment

From the repository root, use the environment in
[../competitor_reproduction.md](../competitor_reproduction.md):

```sh
export PYTHONPATH="code/benchmarks:corenflos:ditlevsen:../pypomp"
JAX_PLATFORMS=cpu python -m pytest code/benchmarks/tests
```

Source papers: [DS19](https://arxiv.org/abs/1707.04235),
[CTDD21](https://proceedings.mlr.press/v139/corenflos21a.html).
