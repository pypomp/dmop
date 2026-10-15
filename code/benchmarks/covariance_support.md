# Local covariance support in the implemented comparisons

Run `covariance_support.py` using the environment in this directory's README.
The saved output is `results/covariance-support/diagnostics.json`. The script
uses the implemented **unregularized** local moment covariance. Normalized
factor columns determine its range; no diagonal floor enters these checks.
The analysis concerns the implemented Gaussian approximation, not a proof of
global hypoellipticity of either SDE.

## Dhaka

In coordinates `(S,I,R1,R2,R3,M)`, write the diffusion vector as `g=q e`,
where `e=(-1,1,0,0,0,0)` and `q=sigma*S*I`. All diffusion-derivative terms
in the order-1.5 covariance are parallel to `e`; the drift derivative adds
`(Db)g`. Modulo a multiple of `e`, `(Db)e` equals

```
v = (0, -(gamma+m), gamma, 0, 0, m).
```

Thus, at an interior state with positive rates and nonzero noise, the local
covariance range is `span(e,v)`, of dimension two. It depends on the ratio
`m/gamma`. Changing both rates by the same positive factor leaves this plane
unchanged, as do changes to the other parameters at the fixed state.
Changing either rate alone changes the plane. The diagnostic checks each
estimation coordinate at a fixed state and finds plane changes for `gamma`
and `m`, with numerical residuals below 1e-10 when projecting the covariance
onto the asserted range.

Two Gaussian laws on distinct two-dimensional affine planes are mutually
singular. Even when covariance ranges agree, changes in the mean can move
the affine support. Consequently an ordinary common full-state density for
the unregularized local Gaussian model is not justified by merely imputing
more time points. A noise-coordinate representation is possible, but would
require different augmentation and parameter updates; this is not a claim
that inference or a density on a suitable support is impossible.

The implemented method composes local moments over 20 internal steps and
adds the covariance floor **once at the monthly endpoint**. It does not
regularize every Euler step. Its regularized Gaussian transitions are full
rank, so the finite-step singular-support statement above does not apply
to the transition actually fitted.

## Daphnia

The fitted model fixes noise in the two susceptible-adult coordinates to
zero. Six independent noises act on the other coordinates. At a strictly
positive state with positive fitted noise coefficients, the local covariance
has full rank eight. This can be checked without a numerical rank threshold:

- The squared-noise (Milstein) diagonal factor has six nonzero entries and
  spans the six directly forced coordinates.
- In the integrated-noise factor `(Db)G-L0G`, the juvenile-noise columns
  project onto the two susceptible coordinates with coefficients
  `lambda_J*sigma_Jn*Jn` and `lambda_J*sigma_Ji*Ji`. Both are nonzero and
  independent; the other components can be removed using the six directions
  already spanned.

The range is therefore all of R^8 throughout this positive interior and does
not rotate with the parameters there. Daphnia is not covered by DS19's
one-smooth-coordinate setup, but that does **not** imply its order-1.5
covariance is rank deficient everywhere. At the specified initial state,
several populations are zero and the local covariance instead has rank three.
The checks include all 38 parameter perturbations at the interior and initial
states and 20 additional positive state/parameter combinations. All tested
interior cases have rank eight. Boundary rank loss and poor conditioning
remain reasons to treat the approximation carefully.

## SPX and fine time grids

For `pypomp.models.spx`, the conditioned latent variance is scalar, with
transition variance `xi^2*v*(1-rho^2)>0` in the interior. Its floor creates
an atom, explicitly included in the transition law. It has neither Dhaka's
rank-two obstruction nor a missing eight-dimensional density.

Karppinen, Singh and Vihola (2024), *Conditional particle filters with bridge
backward sampling*, [doi:10.1080/10618600.2023.2231514](https://doi.org/10.1080/10618600.2023.2231514),
address weakly informative observations and slowly mixing dynamics, including
fine time discretizations, which make resampling/backward sampling inefficient.
This is distinct from the finite-step rank and parameter-dependent support
calculation above. Their result should not be cited as a demonstration of
mutual singularity between parameter-indexed Dhaka or Daphnia path laws.
The current monthly/observation-interval implementation has no intervening
latent grid in the smoother and does not test that fine-grid mechanism.
