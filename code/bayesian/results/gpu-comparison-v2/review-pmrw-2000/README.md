# PMMH review after 2,000 retained draws per chain

This report uses iterations 501--2500 of the four primary PMMH chains.
The length matches the previously scheduled NUTS endpoint. These are the
first draws of the completed 30,000-draw reference, not a new run or a
selected interval. No samples or sampler settings were changed.

All 18 traces and marginal distributions and the trend panels were
visually inspected. The trend distributions overlap, but autocorrelation
remains substantial at lag 100. Seasonal coefficients show persistent
dependence and differences between chains. Chain 3 spends more time in
the upper tails of omega1 and omega3 and the lower tails of omega2 and
omega6. The omega5 and omega6 traces have long excursions. The figures
show slow exploration without the large trend drift in the archived run.

The trend mean is -0.4290, standard deviation 0.1047, and central 95%
interval [-0.6356, -0.2358]. Its rank R-hat is 1.0250, bulk ESS 72.7,
tail ESS 164.9, and MCSE 0.0124. Across all parameters, the largest R-hat
is 1.3122 and smallest bulk ESS is 10.3, both for beta3. The smallest
tail ESS is 26.4 for beta1. Acceptance rates are 0.1965, 0.2555, 0.2035
and 0.1890. These chains have not established joint convergence.

`numerical_review.json` records exact agreement with the saved checkpoint
prefixes and the first 2,000 draws in `../review-pmrw-30000`. Sampled states
and diagnostics are finite, and retained step sizes are fixed. The report's
`complete` flag is false because this report intentionally uses a prefix
of the completed 30,000-draw run.

Equal iteration counts do not imply equal computation: NUTS uses multiple
particle-filter evaluations per iteration, and the preparation costs differ.
Use the longer PMMH reference for checking marginal agreement and retain
its results alongside this shorter comparison. The NUTS endpoint review
is still pending.

From the repository root, reproduce into a fresh directory with:

```bash
python code/bayesian/report.py \
  code/bayesian/results/gpu-comparison-v2/pmrw-long/chain{0,1,2,3} \
  --max-retained 2000 --output NEW_REPORT_DIRECTORY
```
