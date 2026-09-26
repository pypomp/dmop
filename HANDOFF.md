# Handoff

## September 26, 2026: corrected warm-start experiment complete

- The corrected 100-start IF2-warm-start experiment, final Euler-20
  evaluations, every-update evaluations, figures, and audit are complete.
- The original `dmop-if2warm-corrected-final100-v2.service` was OOM-killed
  after persisting 12,162 Corenflos trace evaluations. The resumable v3 service
  skipped completed work, finished successfully, and exited with status 0.
- Both methods have 100 checkpoints and 100 successful final evaluations.
  The consolidated traces contain 21,238 Corenflos rows and 24,848 Ditlevsen
  rows. All four comparison PNGs render cleanly and were visually inspected.
- `python -m corenflos.warm_audit` passed. At the common 942.817-second
  endpoint, final median Euler-20 log likelihoods are -3765.914 for Corenflos
  and -3768.725 for Ditlevsen, versus -3766.639 at the IF2 checkpoint.
- Paired with each start's IF2 checkpoint, Corenflos has median change -0.103
  and mean change -0.019, with 45/100 starts improving. Ditlevsen has median
  change -1.777 and mean change -1.977, with 18/100 starts improving. This
  strengthens the pilot evidence of Ditlevsen surrogate-target mismatch.

## September 24, 2026: remaining review corrections and explicit logs

- Applied the user's approved fixes for points 5, 6, 8 (both propagation-law
  errors), 9, 11, and 12. Point 4 was already fixed in the preceding commit.
  Point 7 was overstated: an independent audit found no active proof equating
  the two empirical random measures or invoking off-parameter unbiasedness
  incorrectly. Their notation is unchanged.
- Corrected selected measurement ratios and whole-factor product scope in
  Lemma S2 and its repeated DMOP-0 derivation; corrected the potential's
  prediction index and the parentwise conditional propagation law.
- Defined the existing corrected weights and normalized resampling
  probabilities, showed the exact conditional-expectation identity, and
  corrected the telescoping initial denominator to the filtered weights.
- Replaced the density-based mixing argument with a past/future sigma-field
  argument for the cloud. Accounted for endpoint resampling in the window
  gaps and corresponding covariance calculation and main proof outline.
- Clarified current-state versus simulator-history spaces, the base-simulator
  and particle-filter expectation laws, and the separate derivative bounds.
- Retained logarithmic J factors in active variance/L2 bounds, particle MSEs,
  theorem copies, summaries, and tuning regimes. Weak-bias squared terms and
  the independent alpha=1 bound remain unchanged. All inactive iffalse blocks
  were preserved. The old main Theorem 5 and its proposed replacement still
  coexist; this pass does not reconcile their pre-existing differences.
- Independent reviewers approved the changed formulas and log propagation.
  Numerical checks covered all 27 J=3 resampling outcomes, DMOP-0 score
  finite differences, all events of a finite Markov mixing example, and
  98 tuning-bound cases. Both PDFs rebuild with Tectonic. SI has no overfull
  boxes or undefined references; MS retains its pre-existing 12.65704pt
  overfull box in an unchanged proof. Equation (3) was split to prevent the
  new logs from colliding with its number. git diff --check passed.
- The user requested an image diff of every fix. The rendered before/after
  gallery is at ../output/pdf/dmop-review-diffs/index.html; its PNGs and
  manifest are in the same directory. Baseline: 5c4576d0. The gallery includes
  all substantive rendered changes, with line shading; equation renumbering
  and pagination-only differences are omitted.
- Keep edits close to the original presentation and avoid unnecessary new
  notation. Experiment state below is retained and was not rechecked.

## September 24, 2026: Lemma S1 ancestry proof

- Replaced the incorrect prediction-weight derivation in `si.tex` with the
  alpha=1 filtered-weight recursion and its measurement-ratio product along
  the filtered ancestry. Retained the original proof's sequence: telescoping,
  weight product, log derivative, path score, then evaluation at theta=phi.
- User preference: keep proofs close to their original structure and scale,
  show essential intermediate steps, and avoid unnecessary auxiliary
  notation. Use existing A,F ancestry notation; do not reintroduce separate
  weight-sum, ancestor-map, or measurement-log symbols for this proof.
- Fixed the terminal history from 1:n to 1:N and qualified Lemma S1's
  comparison with Poyiadjis/Scibior as the same score target, not particlewise
  equality. Differentiate with the reference run fixed before theta=phi.
- The adjacent off-parameter formula now uses normalized terminal filtered
  weights; unnormalized weight growth alone does not establish its variance
  rate. This also addresses point 4 locally.
- Verification: an independent subagent approved the final compact proof,
  including the implicit terminal-time convention and parent-product step.
  Independent numerical checks covered 18 on/off-parameter cases (N=1,2,5;
  J=1,3,8); maximum finite-difference discrepancy was 3.8e-10. Full SI rebuilt
  with Tectonic, no undefined references or overfull boxes; revised pages
  S-2 through S-4 visually checked. `git diff --check` passed.
- The remaining review work is recorded in the newer entry above.
- The experiment state below is retained from the prior handoff and was not
  rechecked during this manuscript-only change.

## Current state

The corrected IF2-warm-start 100-run experiment and its audit are complete.
No experiment service remains active.

## Corrected design

- The warm starts are the 100 inputs to the monitored IFAD training stage,
  checked against MIF iteration 175. The earlier extraction incorrectly used
  the inputs to MIF and is invalid.
- Supplied checkpoints are no longer clipped back to the initial search box.
  The audit checks that optimizer iteration 0 equals the supplied checkpoint.
- Current manuscript timing is 92.16042757034302 seconds for 175 IF2 updates,
  followed by at most 400 continuation updates or 850.6569547653198 seconds.
  The common endpoint is 942.8173823356628 seconds.
- Fits are sequential on one GPU. Fitting uses 100 particles. Final Euler-20
  evaluation uses 5,000 particles and 36 replicates; every stored update uses
  5,000 particles and one replicate.
- Warm-start learning rates are 0.0002 for Corenflos and 0.0005 for Ditlevsen.

## Pilot evidence

The reusable lower-rate pilot used starts 0, 10, ..., 90. Relative to their
independently evaluated IF2 checkpoints, median changes were -0.35 for
Corenflos and -2.80 for Ditlevsen. Corenflos was effectively stable.
Ditlevsen improved its surrogate while all 10 Euler-20 targets worsened,
supporting surrogate-target mismatch. Lowering Ditlevsen's rate from 0.002 to
0.0005 reduced its median loss from -8.2 to -2.8; further tuning toward zero
would increasingly turn the continuation into a no-op.

## Verification

- `52 passed` across `ditlevsen/tests` and `corenflos/tests` after the
  checkpoint extraction and no-clipping fixes.
- Corrected start 0 independently evaluated at -3769.86. Both methods'
  iteration-0 Euler-20 evaluations were about -3771 after removing clipping.
- The 100 corrected baseline evaluations are complete.
- The completed output has 100 checkpoints and 101-line final-evaluation CSVs
  for each method, plus 21,239-line Corenflos and 24,849-line Ditlevsen trace
  CSVs including headers.
- The automatic service audit and an independent rerun both passed.

## Outputs and next steps

- Corenflos fit: `corenflos/results/if2warm_ifad097_budget_j100_final_100`
- Ditlevsen fit:
  `ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100`
- Figures: `corenflos/results/if2warm_ifad097_comparison/figures`
- Invalid/earlier pilots were moved under `/tmp/dmop-if2warm-*`; the committed
  history also preserves earlier results.

The completed results and figures are ready for manuscript use.
