# Handoff

## October 1, 2026: visualize logged-objective / Euler disagreement

- Added two diagnostic figures for the corrected 100-start IF2-warm-start
  experiment: `objective_mismatch_if2warm_medians_r20.png` compares median
  logged-training and independently evaluated Euler-20 changes;
  `objective_mismatch_if2warm_paired_r20.png` plots each start's two changes,
  with a separate panel per method, zero reference lines, and marginal-median
  diamond markers. All 100 starts per method, including outliers, remain in
  the scatter. No titles, captions, or prose annotations were added.
- Changes are paired by start identity. Logged-training change uses the
  pseudo log likelihood at `output_selected_iteration` minus iteration 0,
  not an unselected terminal update. Euler change uses the 5,000-particle,
  36-replicate final evaluation minus the matching IF2 checkpoint evaluation.
  Exported all 200 paired rows to
  `corenflos/results/if2warm_ifad097_comparison/objective_mismatch_if2warm_r20.csv`.
- Verified medians match the previous diagnosis exactly: Corenflos +5.564
  logged-training / -0.103 Euler; Ditlevsen +17.244 / -1.777. Opposite signs
  (logged training improves, Euler worsens) occur for 33/100 Corenflos starts
  and 80/100 Ditlevsen starts. These are observed disagreements; the noisy,
  single-filter training values do not prove objective bias by themselves.
- Added tests for shuffled-row identity pairing, selected-update selection,
  missing inputs, paired medians, full scatter inclusion, and no plot prose.
  All 59 Corenflos/Ditlevsen tests and the expanded six-figure completion
  audit pass. Both new PNGs were visually checked; the four prior PNGs and
  all experimental results are unchanged. `git diff --check` is clean.

## October 1, 2026: remove plot prose and show the Corenflos median

- User preference: no explanatory captions or prose annotations on these
  figures. Keep axis labels and method legends; do not replace a missing
  curve with text about it.
- Removed the overview caption and in-panel off-scale annotation. Extended
  the overview's lower limit to -4800 so the standalone Corenflos median
  near -4560 is visible alongside the other five methods. The separate
  continuation zoom's limits are unchanged. Initial cold-start values below -4800
  remain outside this overview's displayed range.
- Updated regression tests to check the visible Corenflos median, six methods,
  no caption, and no text/label annotation layers.
- Display labels now read "Ditlevsen + IF2 warm start" and "Corenflos + IF2
  warm start" throughout the optimization, likelihood, and parameter figures.
  Internal result keys and experiment data are unchanged.
- Regenerated and visually checked all four PNGs. All 55 Corenflos/Ditlevsen
  tests and the completion audit pass; `git diff --check` is clean.

## October 1, 2026: include standalone runs in the overview

- The full optimization figure now includes standalone Corenflos and
  Ditlevsen alongside their IF2-warm-start versions, IF2, and IFAD-0.97.
  Its displayed limits are -4300 through -3735. The preceding plotting
  implementation omitted both standalone trace sources; this was an oversight.
- Retained the separate -3820 through -3735 continuation zoom. Made the
  standalone maximum-envelope boundaries more visible and labeled the bands.
  Corenflos's standalone median is below -4300 throughout, ending near -4560;
  the overview explicitly labels that fact rather than clamping the median.
  Ditlevsen's standalone median ends near -4099. These are one-replicate
  trajectory summaries, not the 36-replicate selected-final estimates.
- Visually checked both optimization PNGs. Only the overview PNG changes;
  the focused optimization, likelihood, and parameter PNGs are unchanged.
  Added regression tests for all six trace sources, warm origins, separate
  limits, and preservation of the off-scale median. All 54 Corenflos/Ditlevsen
  tests and the completion audit pass; `git diff --check` is clean.
- Investigated the apparent warm-start plateau using existing fit logs and
  optimizer code; no training runs or algorithm settings were changed.
  Median completed updates are 213 for Corenflos and 252 for Ditlevsen, and
  parameters move in both. Logged training-objective median changes at the
  selected updates are +5.564 and +17.244, respectively, while paired final
  Euler-20 median changes are -0.103 and -1.777. Training values are noisy,
  single-filter estimates, so these differences alone are not a controlled
  demonstration of objective bias.
- Ditlevsen uses an approximate Gaussian block-transition objective and a
  sampled-path score; Corenflos uses finite-particle entropic transport with
  clipped transport adjoints. Both fit with 100 particles and conservative
  learning rates. Every recorded Corenflos gradient norm exceeds its global
  clipping threshold of 100; 35/100 fits restart and reduce their rate.
  Ditlevsen clips an averaged score, not the recorded raw score norm.
  Objective mismatch/noisy directions and conservative steps are the leading
  explanations, not a frozen optimizer. Distinguishing these quantitatively
  would require a controlled particle-count/gradient-alignment experiment.

## October 1, 2026: make continuation traces readable

- Corrected the main warm-start optimization figure's log-likelihood limits
  from a -4300 lower bound to -3820 through -3735, with ticks every 10 units.
  The old scale compressed the continuation traces into a thin strip; the
  initial September 26 visual inspection missed this readability problem.
- Used a dashed Corenflos median and removed its extra thick overlaid line,
  so the nearby Ditlevsen median remains visible. Regenerated both optimization
  PNGs; the full-range figure retains its overview scale.
- Checked the regenerated main PNG visually and confirmed that all median,
  10th-percentile, and maximum values after the IF2 checkpoint fit inside the
  focused limits. The completion audit still passes; experimental results,
  likelihood comparison, and parameter figures are unchanged.

## September 28, 2026: Daphnia references in the main text

- Added the approved Daphnia sentences to the Introduction and Discussion,
  including the existing `yang25daphnia` citation and supplement Section S10.
- Rebuilt `ms.pdf` with latexmk; checked both changed pages and confirmed the
  Yang et al. (2025) bibliography entry. No undefined citations or references.
  The existing overfull box in the unchanged targeting proof remains.

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
