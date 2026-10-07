# Input data for Fig. 1A

These CSVs are copied from the exact diffPomp revision identified in the
[parent README](../README.md). They are the inputs used by
[generate.py](../generate.py); they do not come from the newer sibling Pypomp
checkout used for the SI benchmarks.

| File | Contents |
|---|---|
| [dacca.csv](dacca.csv) | Monthly observation times, cholera deaths, and covariates; the generator reads `cholera.deaths` |
| [covars.csv](covars.csv) | Trend, population derivative, population, and six seasonal spline covariates |
| [covart.csv](covart.csv) | Time coordinates for the rows of `covars.csv` |

The first column in each file is the original CSV row index. The generator
pairs `covars.csv` with `covart.csv` and interpolates onto the original
20-steps-per-month grid. Keeping the original inputs here makes the Fig. 1A
calculation independent of later changes to model data and interpolation.
