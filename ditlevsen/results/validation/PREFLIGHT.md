# Dacca SMC-score preflight (not the final experiment)

The corresponding visualizations are:

- `preflight_likelihood_comparison.png`: both panels of Fig. 3, with the three
  20-update Ditlevsen fits in panel A and the single 80-update fit in panel B.
- `preflight_parameter_comparison.png`: Fig. 4 density facets for the original
  methods with the 80-update DS estimate shown as a dashed black line.
- `preflight_optimization_elapsed.png`: Euler-20 likelihood against elapsed
  seconds beside the manuscript's 100-search trace summaries.
- `preflight_substep_diagnostics.png`: collapse and filtering health as process
  substeps per month increase.
- `figure_captions.tex`: captions kept separate from the PNG files.

These checks use independent bounding-box starts from `make_starts(3, 631409)`.
Euler-20 values use fresh Pypomp particle-filter evaluations with 1,000
particles and four replicates. They are deliberately smaller than the final
5,000-particle, 36-replicate evaluation and must not be used as final estimates.

`preflight_optimizer.csv` checks 20 updates at five inference substeps and a
learning rate of 0.05. All three starts move strongly in the correct Euler-20
direction, but only two complete all requested updates. The first terminates
after its health-preserving backtracking cannot find a proposal below the 50%
invalid-particle threshold. Thus the fitting signal works, but the optimizer is
not yet robust enough for an unattended 100-start run.

`preflight_convergence.csv` is a separate 60-update stress test from start zero,
using learning rate 0.1 before health-preserving backtracking was added. It
improves the independently estimated Euler-20 likelihood from about -6209 to
-4015 in 82 seconds, but is non-monotone and places the measurement-noise
parameter `tau` at its upper bound. It demonstrates a real SMC fitting signal
while also exposing parameter bias/instability that must be resolved or
reported.

An equal-update grid smoke test at learning rate 0.1 gave, after ten requested
updates from start zero: five substeps collapsed after update five; ten
substeps completed in 29.69 seconds at Euler-20 -4388.82; twenty substeps
completed in 55.06 seconds at Euler-20 -4291.95. Twenty substeps were slightly
better per update but substantially worse per elapsed second (34.8 versus 61.2
log-likelihood units gained per second for ten substeps). This is computational
degradation, not evidence of the global CPF mixing effect studied by KSV.

The optional fixed-endpoint bridge kernel changed many within-month interior
states but had mixed effects on five-update parameter progress and never
changed the one-ancestor forward genealogy. It is therefore excluded from the
planned headline fit. It is not full KSV CPF-BBS.

`preflight_paper_schedule.csv` is the decisive paper-length check: 80 updates,
128 particles, learning rate 0.05, and the DS gain schedule with 30 unaveraged
iterations followed by exponent 0.9. It completed without backtracking in
106.79 seconds and improved Euler-20 from -6208.21 to -4020.13. This remains
about 274 log units below the comparable-effort IFAD-0.97 median near -3745.8.
Several parameters reached the search-box boundary (`gamma=10`, `m=0.6`, and
early `tau=0.5`). Thus the method is computationally operative but has not
passed the good-parameter-estimation gate for a final 100-start claim.
