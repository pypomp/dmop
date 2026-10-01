# Source audit for the Ditlevsen–Karppinen comparison

This note records the computational-statistics details used to design the
benchmark.  Page and equation numbers refer to the cited paper versions in
`../../dmop_private/refs/`.

## [Ditlevsen and Samson](https://arxiv.org/abs/1707.04235) (arXiv:1707.04235v2; JRSSB 2019)

- Their definition of hypoellipticity explicitly combines a rank-deficient
  instantaneous diffusion with a smooth finite-time transition density
  (arXiv pp. 1–3).
- Their model class is narrower than an arbitrary hypoelliptic SDE.  Equations
  (2)–(4) have one smooth coordinate `V`, a `p`-dimensional rough block `U`, a
  `p`-dimensional Brownian motion, and diagonal `Gamma` with every diagonal
  entry positive.  Thus a state of dimension `d=p+1` has diffusion rank
  `p=d-1`.
- Condition 1 propagates one of those directly forced rough directions to the
  single smooth coordinate through the first drift bracket.
- The order-1.5 scheme is given in equations (30)–(33).  Its conditional mean
  is the second-order generator expansion in equation (10).  The leading
  smooth variance is order `Delta^3`, the smooth/rough covariance is order
  `Delta^2`, and the rough variance is order `Delta`.
- Section 5 and Algorithms 1–2 treat the partially observed case by SMC path
  imputation followed by SAEM.  The observed coordinate `V` is observed
  directly, whereas all Dacca states are latent behind a noisy death-count
  measurement.  Their SAEM convergence construction also assumes the complete
  pseudo-likelihood is a curved exponential family; this is verified for their
  three examples, not for Dacca's 23-parameter model.
- Their harmonic-oscillator simulation uses `Delta=0.02` with 1,000 observed
  points and 100 particles per SAEM iteration (Section 6.1).  The scheme grid
  is the observation grid in the reported examples.  The formulas can be
  evaluated at other positive step sizes, but that alone gives neither
  large-step stability nor a full-rank transition outside their bracket-depth
  assumptions.

The local private copy and the URL supplied by the user are byte-for-byte the
same PDF (SHA-256
`f96facb1ab3c0bb69780cd024f729984e47acf7ab8998cf91308b9157c9a9373`).

Using the observation grid in their examples is not a stability theorem for an
arbitrary observation interval.  At the Dacca point, the three-stage immunity
rate is `3 epsilon = 57.3/year`.  The DS second-order mean applied to a decay
mode has stability polynomial `1-z+z^2/2`, stable on the negative-real test
equation only for `0 <= z <= 2`.  Monthly and half-monthly Dacca steps give
`z=4.78` and `2.39`; five substeps give `z=0.96`.  This independently predicts
the observed failure at one and two substeps.

## Why Dacca is still hypoelliptic but is not their depth-one model

The active Dacca representation has six coordinates
`(S, I, R1, R2, R3, Mn)` and one scalar Brownian driver.  The instantaneous
diffusion rank is therefore one, as expected for a hypoelliptic model.  At the
published parameter point, normalized columns from the successive bracket
sequence

`g, [b^S,g], [b^S,[b^S,g]], ...`, where
`b^S = b - (Dg)g/2` is the Stratonovich drift,

have numerical ranks `1, 2, 3, 4, 5, 6`.  This supports a smooth
full-dimensional finite-time law, but only after five drift-bracket levels.
The first-bracket leading Gaussian covariance used by the order-1.5 local
generalization has rank at most two.  This is a statement about the approximate
transition, not a demand that the instantaneous diffusion be full rank.

The state dependence `sigma*S*I` does not add stochastic directions at this
order.  Dacca has `g(x)=q(x)e` for the constant vector
`e=(-1,1,0,0,0,0)`, so `(Dg)w` is collinear with `e` for every `w`.
Milstein/diffusion-derivative terms remain on the forced line, while the first
drift-propagation term adds only `J_b g`.

Diffusion rank is invariant under invertible state transformations.  A
six-state/rank-one system therefore cannot be transformed into the
six-state/rank-five structure assumed in equations (2)–(4).

## Reviewer 1's proposal

Section 3.1 of `../../dmop_private/jrssb-r2/response.tex` argues that the DS
scheme can be evaluated at `Delta=1/20` month and would give a tractable
transition.  The report separately acknowledges that ordinary backward
sampling degrades under fine path discretization and points to Karppinen et al.
as a possible repair.

The first claim correctly distinguishes the order-1.5 scheme from a singular
Euler step, but it does not address the mismatch between DS's diffusion rank
`d-1`/first-bracket model and Dacca's diffusion rank one/longer bracket chain.
The second claim motivates a separate mixing experiment rather than changing
the target likelihood.

## Karppinen, Singh, and Vihola (2024)

- Their CPF-BBS targets fine-grid smoothing degeneracy: ordinary backward
  weights contain a concentrated one-step transition density and tend to
  recover the existing ancestor.
- Assumption 7 requires evaluable multi-step endpoint densities and simulable,
  evaluable bridge conditionals for each block.  Linear-Gaussian proposals meet
  this condition.
- Their Feynman–Kac formulation permits the tractable proposal to differ from
  the statistical transition, provided the transition/proposal density ratio
  is included in the potentials (Section 2, equation (2), and Section 5).
- Algorithms 7–8 run a forward conditional particle filter and then bridge CPF
  updates from the final block backward.  A fixed-month endpoint-weight ESS is
  only a diagnostic of this mechanism; it is not the full CPF-BBS Markov
  kernel.

Consequently, Karppinen is relevant only once extra within-month latent points
are introduced.  It does not fix a coarse monthly Taylor instability, supply
missing bracket directions, or create Dacca's numerical M-step.

## DMOP benchmark target

The authoritative settings are in `../ms.tex`,
`../code/global_search/prep.py`, and `../imgs/precise_table.tex`:

- 20 Euler substeps per monthly interval;
- 23 estimated parameters, with the remaining Dacca parameters fixed;
- 100 searches initialized uniformly from a specified wide physical-scale box;
- selected parameter vectors reevaluated with 5,000 particles and 36
  replications;
- comparison by both best and median final Euler-model log likelihood;
- comparable-effort IFAD-0.97 best/median: `-3744.17 / -3745.77`;
- extended-effort IFAD-0.97 best/median: `-3744.19 / -3744.22`.

Therefore `r`, the number of Ditlevsen inference substeps per month, must vary
without changing the evaluation model.  Every candidate parameter vector is
scored under Euler-20.  A one-step Euler likelihood answers a different
question and is excluded from the headline comparison.
