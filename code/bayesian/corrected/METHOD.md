# Target and acceptance calculations

Write z for the 18 standardized estimation coordinates in `model.py`, and u
for a particle-filter random seed. Let m denote the seed's base law and let
Lhat(z,u) be the ordinary bootstrap-filter likelihood estimate. The intended
extended density is

    pi_ext(z,u) proportional to Lhat(z,u) p(z) m(u).

For a nonnegative unbiased likelihood estimator, integrating u gives the
ordinary posterior. This statement refers to the specified Euler model,
not the continuous-time diffusion limit. Numerical PRNG and floating-point
approximations are those of Pypomp/JAX. There is no division by observation
count, and no estimated KDE in p(z). The prior is Normal(0, prior_scale^2 I)
in z coordinates. `CENTER` and `SCALE` convert it to estimation coordinates;
the affine Jacobian is constant. Positive rates are lognormal on the natural
scale. Trend is transformed as 100 beta_trend. Fixed parameters are excluded.

The force G(z,u) is the Pypomp MOP score plus the prior gradient. It may be
biased or differ from the derivative of the numerical PF log likelihood.
Nonfinite entries are set to zero and its norm is capped at 10,000. These
operations define a deterministic proposal force, not an approximation to
use in the acceptance target.

## Joint particle MALA (`pmmala`)

Let C = B B^T be the fixed proposal covariance shape and h the step size.
Propose independently

    u' ~ m,
    z' ~ q(. | z,u) = Normal(z + h^2 C G(z,u)/2, h^2 C).

The log acceptance ratio is

    log Lhat(z',u') - log Lhat(z,u)
    + log p(z') - log p(z)
    + log q(z | z',u') - log q(z' | z,u).

Both proposal densities are evaluated, including the candidate's force at
u'. The m terms cancel. On rejection, retain z, u, the likelihood estimate
and the force together. Thus score bias affects efficiency but does not
remove the Metropolis correction. This is the particle MALA construction
of [Nemeth, Sherlock and Fearnhead (2016)](https://arxiv.org/abs/1412.7299),
with a MOP force in place of their score estimate. We make no claim that
this replacement meets the assumptions of their scaling theorems.

## Other pilot kernels

`pmrw` is the same joint parameter/randomness update with a symmetric random
walk and no force. It supplies a comparison with the same likelihood and
prior. `mala` alternates an independent MH update of u at fixed z, followed
by MALA at fixed u, using both conditional proposal densities.

`hmc` alternates the same u update with a fixed-length leapfrog trajectory
at fixed u. Write p ~ Normal(0,I). A kick adds h B^T G(z,u)/2 to p, and a
drift adds h Bp to z. Each kick and drift is a shear of determinant one,
even for a nonconservative force. Their symmetric composition, followed
by a momentum flip, is an involution. The endpoint correction is therefore

    log Lhat(z',u) - log Lhat(z,u) + log p(z') - log p(z)
    + (||p||^2 - ||p'||^2)/2.

The force is recomputed deterministically at each endpoint. This argument
covers fixed-length proposals; it is not a justification of an arbitrary
NUTS stopping rule.

## Checks and adaptation

`test_kernels.py` verifies the leapfrog involution and Jacobian determinant
with a deliberately nonconservative force. It also checks all four kernels
against an analytically known extended Gaussian target, using a deliberately
inaccurate force. The auxiliary likelihood factor is

    exp(a z u - a^2 z^2/2),  u ~ Normal(0,1),  a=.8,

whose expectation is one. With likelihood exp(-z^2/2) and prior
exp(-z^2/2), the marginal variance is 1/2, Cov(z,u)=a/2, and
Var(u)=1+a^2/2. The checks include these auxiliary moments, not just z.
A near-half-period HMC tuning mixed poorly in the first test; the retained
check uses a nonresonant trajectory and 240,000 transitions per kernel.

Step size and covariance shape are adapted only during warmup. Retained
sampling freezes both. Every rejection is retained. Diagnostic gates apply
to all 18 estimated coordinates and are followed by visual inspection and
prior/particle sensitivity checks. A numerical pass is evidence of mixing,
not a proof that every posterior mode has been visited.
