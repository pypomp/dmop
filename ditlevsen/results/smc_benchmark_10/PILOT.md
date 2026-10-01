# Ten-start Dacca pilot

This is a gate before the 100-start experiment. Ten starting values were drawn
from the manuscript bounding box and reused for fits with 5, 10, and 20 process
substeps per month. Each fit used 100 particles and at most 1,000 seconds.
Final parameter vectors were evaluated independently under the manuscript's
Euler-20 model using 5,000 particles and 36 replications.

The fitted method is a regularized Ditlevsen-style Gaussian pseudo-score
method, not a literal implementation of Ditlevsen and Samson Algorithm 2. The
first-bracket Dacca covariance has rank at most two in six dimensions, so the
implementation adds a relative covariance floor of \(10^{-7}\). SMC draws a
complete trajectory under that regularized model, and automatic
differentiation gives the complete-data pseudo-score for a Robbins--Monro
update. The reported Euler-20 likelihood is not the training objective.

| substeps/month | 10th percentile | median | maximum | at least -3780 |
|---:|---:|---:|---:|---:|
| 5  | -9201.26 | -3847.94 | -3786.13 | 0/10 |
| 10 | -5043.64 | -3821.47 | -3776.58 | 1/10 |
| 20 | -4046.67 | -3846.53 | -3761.41 | 1/10 |

The comparable-effort manuscript medians are -3755.41 for IF2, -3753.88 for
IFAD-0, -3749.12 for IFAD-1, and -3745.77 for IFAD-0.97. On this pilot, IFAD
and IF2 therefore outperform the regularized Ditlevsen-style method. Even its
best result, -3761.41 at 20 substeps, is below the IF2 median and 15.65 log
units below the IFAD-0.97 median.

Increasing the number of substeps has two different effects. Robustness
improves: zero-update failures fall from 3/10 at five substeps to 1/10 at ten
and 0/10 at twenty. Computational progress slows: the median numbers of
accepted updates are 487.5, 1391.5, and 794.0, respectively, with the
five-substep median lowered by early failures. Conditional on surviving,
twenty-substep fits perform fewer updates per second than ten-substep fits.
Statistical performance is not monotone: the median is best at ten substeps,
whereas the best individual result occurs at twenty.

Figure 3 analogues omit likelihoods below -3780, exactly as in the manuscript.
Consequently no five-substep Ditlevsen point appears in that panel, and only
one appears for ten and twenty substeps. The elapsed-time plots relax the
manuscript's lower display limit from -3800 to -3900 so that the Ditlevsen
medians are visible without letting catastrophic starts determine the scale.
Full results remain in final_evaluations.csv.

No 100-start run has been launched.
