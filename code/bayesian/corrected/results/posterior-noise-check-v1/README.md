# Particle likelihood noise at sampled states

The early GPU PMMH report showed a long rejection plateau in chain0 and lower
acceptance than in the other chains. `prepare.py` freezes that plateau's state
and three comparison states, using only the first 1,400 iterations.
`protocol.json` records the exact selection and source checkpoint hashes.

`../../noise_check.py` evaluates 32 independent particle filters at each state,
with 5,000 and 10,000 particles. It saves every likelihood estimate and seed.
It compares the retained 5,000-particle estimate with fresh estimates at the
same parameter value, including the estimated acceptance of a likelihood-only
randomness refresh. This helps assess whether likelihood noise contributes
to the rejection plateau, without changing the ongoing sampler.

These are local diagnostics. They do not establish convergence or particle
adequacy over the full posterior. Repeated seeds across particle counts do
not imply identical simulated paths or particle genealogies.
