# Handoff

## October 9, 2026: Gaussian timing hardware stated in SI

- Added the recorded CPU distinction to the Gaussian timing paragraph:
  IFAD, IF2 and DS19 used four efficiency cores of the i9-13900K; CTDD21
  used two performance cores with four hardware threads. Verified all
  retained configurations and the local CPU/core mapping before editing.
  Concurrent batches used disjoint groups. This supplements the particle
  and iteration-count caveats requested for the runtime comparison.
- Native compiler attempt returned `No handler registered`; local
  `make dmop/si.pdf` succeeded in place. No errors, undefined references or
  overfull boxes were found. Visually inspected page S76; still 78 pages.
  Build log `/tmp/dmop-si-hardware-build.log` and page image
  `/tmp/dmop-si-hardware-76.png` retain verification output.
- All eight final fitting subprocesses remain live. Daphnia's first full
  MPIF fits are complete and its IFAD stages are running. Corrected A/B
  remains at 16 audited cold pairs with workers fitting starts 16--19;
  no warm pair is complete. Final reports and SI results remain pending.

## October 9, 2026: sixteen corrected cold pairs audited; Daphnia fitting

- Corrected cold starts 0--15 pass the A/B audit. New score/SAEM Euler
  likelihoods: start 13 (-4113.783415, -4436.841451), start 14
  (-5074.391802, -6269.649828), start 15 (-4042.373562, -4258.714804).
  Scores for these starts were newly fitted, not reused. Both arms used
  their declared time budgets and retained the last eligible estimates.
- Workers are on the final four cold starts, 16--19. Worker 0 has completed
  the score arm of start 16 and is fitting SAEM; the other workers have
  advanced to their next score fits. No warm pair has completed. Do not
  use this partial cold comparison as the full cold/warm result.
- All four final Daphnia batches completed compilation and the first warm
  MPIF stage. They have now also completed their first full MPIF searches,
  at starts 0, 5, 10 and 15, in 647.35, 653.36, 660.15 and 639.78 s.
  IFAD and competitor stages/evaluations remain. No final Daphnia batch is
  complete. Actual subprocesses 1515337, 1515335, 1515336 and 1515829 remain
  live with increasing CPU time; the same four corrected A/B children are
  live as well. No restart or fitting-setting change occurred.
- Final five-model panels/table, full A/B report, results prose and SI
  integration remain pending. This continuation audited and preserved
  three complete pairs after a verified computational wait.

## October 9, 2026: thirteenth corrected pair complete

- Cold start 12 passes the A/B audit, bringing the total to 13 complete
  cold pairs and no warm pairs. Its score fit stopped with
  `non-finite-path-or-score` after three updates (37.88 s); the last valid
  estimate is retained. SAEM completed 31 eligible outer iterations by
  841.09 s. Independent Euler log-likelihoods: initial -7044.278648,
  score -6282.268162 (MCSE .825878), SAEM -5836.647378 (.367649).
  Preserve the failed-fit marker and include this pair in all summaries.
- Worker 0 has finished cold start 12; the other workers are finishing
  numerical SAEM on starts 13--15, with score fits still needed. All four
  final Daphnia subprocesses continue accumulating CPU time while compiling.
  The same eight actual process handles were repeatedly confirmed live;
  none was restarted. Goal remains active, awaiting full experiments and
  the final plots/table/SI integration.

## October 9, 2026: twelve corrected cold pairs complete

- Corrected starts 0--11 pass the A/B audit. New score/SAEM Euler
  likelihoods: start 9 (-4046.263869, -5628.450932), start 10
  (-4148.924998, -4949.562755), and start 11 (-4059.624215, -4719.293243).
  All completed compatible old score fits have now been reused and audited;
  remaining starts require both fits. All paired evaluations remain
  independent of fitting and are retained even when a fit stops early.
- Workers have advanced to cold starts 12--15. No warm pair is complete.
  The full 20-cold/20-warm experiment and its complete report remain pending.
- Confirmed all eight actual fitting subprocesses live with accumulating CPU
  time. Final Daphnia children 1515337, 1515335, 1515336 and 1515829 are
  compiling; A/B children 1474435, 1479209, 1479396 and 1478936 are fitting.
  No process was restarted on account of quiet compilation logs.
- The current continuation made progress by auditing and preserving three
  more complete pairs. Full plots/tables, results prose and SI integration
  still depend on the remaining declared jobs.

## October 9, 2026: selected settings reflected in SI and documentation

- Updated the Gaussian runtime paragraph in `si.tex` to state the selected
  500 particles for DS19, IFAD and IF2. Removed the provisional sentence
  about selecting between counts; the repository retains the tuning record.
  The benchmark README now links the completed selection and gives .001
  for both Daphnia competitor learning rates.
- Native `compile_latex_document` now responds, but failed because its
  standalone compiler could not load the existing project `macros.tex`.
  Do not replace the document or inline its dependencies to work around this
  service limitation. `make dmop/si.pdf` succeeded in place (78 pages);
  visually checked updated page S76. Existing theoremstyle/PDF-string
  warnings remain; no new source error or overfull box was reported.
  Build log: `/tmp/dmop-si-selected-particles-build.log`.
- All four final Daphnia child processes are live, initially compiling:
  1515337, 1515335, 1515336, 1515829. All four corrected A/B workers are
  also live. Nine cold pairs pass the audit; starts 9--11 are evaluating
  and worker 0 is fitting start 12. No warm pair or final Daphnia batch
  is complete yet. Complete plots/tables and SI results remain pending.

## October 9, 2026: tuning finished; all final Daphnia batches launched

- Separate Gaussian tuning selected J=500 for IFAD, IF2 and DS19 in both
  examples. Both counts had no failed tuning fits; summed median analytic
  likelihood deficits were .674126 (J=100) and .048004 (J=500). Selection
  and all 16 tuning fits are saved under `code/benchmarks/results`.
  The completed batches pass validation; no final starts selected J.
- Regenerated the completed four-model preview using the default selected
  count: `/tmp/dmop-four-model-selected`. Visually inspected final
  distributions, full-range progress and the separate linear detail view.
  All points and bands are visible in their appropriate full-range plots;
  detail views are explicitly labeled. Daphnia is still absent, so these
  artifacts remain previews outside `imgs/benchmarks` and the SI.
- Lower-rate Daphnia recovery completed with 16 fits and no failures. All
  16 raw final evaluations pass independent likelihood/MCSE recalculation.
  DS19 median is -870.817540 and CTDD21 median is -866.542933; both select
  learning rate .001 under the saved tuning rule. Recovery provenance and
  the interrupted original archive are preserved.
- All final Daphnia queues have launched: `final-daphnia-00`, `05`, `10`
  and `15`, five starts each, on E-core groups 16--19, 20--23, 24--27 and
  28--31. Initial child PIDs 1515337, 1515335 and 1515336 correspond to
  the first three batches. The fourth launched after particle tuning freed
  its cores. Source/settings unchanged; jobs begin with compilation.
- Corrected cold start 8 is now complete and passes the nine-pair A/B
  audit: score -4050.744, SAEM -4409.795. The A/B workers continue;
  no warm pair is complete. Final cross-model report and full A/B report,
  SI results prose, inclusion, compilation and visual review remain pending.

## October 9, 2026: matched Gaussian runs and higher-rate Daphnia tuning complete

- All eight equal-particle Gaussian batches are complete. The final
  oscillator J=500 batch has 20 fits each for IFAD, IF2 and DS19, all with
  successful status. Median likelihood/time (s): IFAD 5403.6663/70.0061,
  IF2 5401.9489/89.4258, DS19 5403.6880/9.3939. CTDD21 retains its original
  J=25 run (5397.1785/117.3200); its core type differs, as documented.
- Ran the batch audit and report validation, then visually inspected all
  three oscillator J=500 plots at `/tmp/dmop-oscillator-j500-complete`.
  All use linear axes; progress has 10th--100th percentile shading at .10.
  The complete cross-model publication report still awaits particle-count
  selection and Daphnia final fits. Do not promote this explicit-J preview
  before the separately declared tuning selects the common count.
- `dmop-matched-particles` is inactive after successful completion. The
  existing `dmop-tune-particles` service (PID 1437187) has started
  `tuning-linear-particles100` on the freed E cores. It will also evaluate
  oscillator and J=500, then choose the shared count by the existing rule.
- Higher-rate Daphnia tuning recovery is complete, and all 16 final
  evaluations pass raw-replicate likelihood/MCSE recalculation. All four
  DS19 fits failed on nonfinite updates and remain included. Median
  likelihoods: MPIF -864.5831, IFAD -861.5425, DS19 -937.4442,
  CTDD21 -867.5931. Immutable original archive and recovery provenance are
  retained. The lower-rate recovery is still finishing its last start;
  final Daphnia queues still wait for both tuning batches.
- Corrected Dhaka A/B workers remain active with eight completed cold
  pairs. Full cold/warm report and SI integration remain outstanding.

## October 9, 2026: eight corrected cold pairs audited

- Corrected cold starts 0--7 now pass `audit_results.py --dhaka-ab`, including
  reused-source hashes, selected estimates, M-step Q acceptance and raw
  likelihood/MCSE recalculation. Newly completed score/SAEM likelihoods:
  start 5 (-3980.858, -10675.681), start 6 (-3967.135, -3963.489), and
  start 7 (-3972.757, -6549.959). Numerical SAEM is higher for one of these
  eight starts; no warm pair is complete. Avoid full-study conclusions.
- All four corrected workers have advanced to cold starts 8--11. The
  oscillator J=500 IFAD/IF2 batch is finishing start 20; the final DS19
  batch and separate particle selection are still downstream. Both
  Daphnia tuning processes remain live, finishing their final start.
- This continuation made progress by auditing and saving three new complete
  pairs, after monitoring the actual child PIDs and increasing CPU time.
  Full report generation, visual review and SI integration remain pending.

## October 9, 2026: fifth corrected pair complete

- Corrected cold start 4 completed and passed the five-pair audit. Its score
  fit stopped after ten updates with `non-finite-path-or-score`; retain the
  last eligible estimate and mark it failed. Euler likelihoods are initial
  -8887.596107, score -5831.464297 and SAEM -8862.452111. SAEM completed 16
  eligible iterations; nine increased Q by more than 1e-7. Its final MCSE
  is approximately 1, with the likelihood mean dominated by a replicate;
  do not interpret this delta-method MCSE as reliable confidence coverage.
- Worker 0 has advanced to cold start 8. The other three corrected workers
  are finishing cold starts 5--7. All seven fitting subprocesses were
  confirmed live with increasing CPU time during this continuation.
- The oscillator J=500 IFAD/IF2 batch has completed 18 of 20 starts; DS19
  J=500 and separate particle tuning remain queued. Both Daphnia recovery
  tuning processes are finishing their last start. No final Daphnia batch
  or warm A/B pair is complete yet. Prepared publication fragments remain
  unincluded until complete reports can be reviewed and compiled.

## October 9, 2026: four corrected pairs audited and previewed

- Corrected cold starts 0--3 are complete; `audit_results.py --dhaka-ab`
  passes all four, including raw likelihood replicates, reused score hashes,
  selected checkpoints and M-step acceptance. Score/SAEM likelihoods are
  (-4027.31, -5785.06), (-4010.90, -4352.70), (-4040.37, -4598.35), and
  (-3983.29, -6614.14). No warm pair is complete. Do not infer the full
  comparison from these four starts.
- At the user's request, generated and visually inspected a partial two-panel
  preview: `/tmp/dmop-saem-ab-preview/comparison.png` and `.pdf`; generating
  script, paired CSV and audit JSON are in the same directory. The plot uses
  linear axes, connected matched starts and paired differences with twice
  Monte Carlo SE. It explicitly says 4/20 global pairs and warm starts pending.
  Shown in chat only; no partial figure was inserted into the SI.
- Confirmed all corrected worker service handles live (PIDs 1474287,
  1474290, 1474327, 1474357), now fitting starts 4--7. Matched-particle
  oscillator J=500 is fitting start 14; six of eight batches are complete.
  Both Daphnia tuning recovery services are live and finishing their last
  recovery start. Final Daphnia, particle tuning and reporting queues are
  still waiting for their declared dependencies. No run was restarted.
- The previous user-request turn made progress by producing and validating
  the requested partial A/B figure. Final benchmark panels/table, complete
  cold/warm A/B report and SI integration remain outstanding; goal active.

## October 9, 2026: first corrected Dhaka pair complete and audited

- Corrected `saem_ab_scaled/cold/start000` is complete. Euler log-likelihoods
  are initial -6207.322898 (MCSE .159354), score -4027.308296 (.882202), and
  numerical SAEM -5785.058460 (.469738). SAEM completed 14 eligible outer
  iterations; all 14 increased their fixed Q by more than 1e-7. Its selected
  estimate is from 765.85 s; the call ended at 853.19 s after discarding an
  incomplete over-budget M-step. This single corrected pair still favors
  score updates; it is not a result for the whole cold or warm comparison.
- `audit_results.py --dhaka-ab` passes for this pair, including original and
  reused score hashes, checkpoint eligibility, M-step acceptance and raw
  Euler likelihood/MCSE recalculation. Audit output is temporarily saved at
  `/tmp/dmop-corrected-pair-audit.json`. Use the same audit as more pairs finish.
- All seven fitting services remained live through 05:35 UTC. Worker 0 is
  advancing to the next corrected pair. Oscillator J=500 had completed 11
  of 20 IFAD/IF2 starts. Daphnia's higher-rate tuning recovery had completed
  start index 2 (DS19 again stopped on a nonfinite update); the lower-rate
  run was still finishing that start. Full report/SI work remains pending.

## October 9, 2026: A/B caption fragment prepared during verified wait

- Added `imgs/benchmarks/saem_ab_figures.tex` for the corrected Dhaka paired
  figure and table. Captions define cold/warm starts, the common particle
  and time budgets, paired Monte Carlo error bars, retained failed fits,
  iteration counts and exclusion of precomputed IF2 cost. The fragment is
  not yet included in `si.tex`; include and compile it only after all 40
  corrected pairs and the generated report have been reviewed.
- Repeatedly verified the same live service handles and advancing fitting
  logs. At 05:31 UTC corrected worker 0 had reached 765.85 seconds in its
  first SAEM fit, with no completed corrected pair yet. The other corrected
  workers were progressing through their first fits. The oscillator J=500
  IFAD/IF2 batch had completed 10 of 20 starts; Daphnia tuning was fitting
  its next start. No restart or change of fitting settings occurred.
- This continuation is a verified computational wait plus preparation of
  the remaining SI caption fragment. The requested final figures/table and
  SI integration still require the unfinished experiments. Goal is active.

## October 9, 2026: all corrected workers have taken over

- At 05:23 UTC all four `dmop-saem-scaled-wN` services were active and had
  launched their corrected fitting subprocesses. All old `dmop-saem-ab-wN`
  services were inactive with MainPID=0. The queue preserved the in-flight
  score fits for cold starts 8--11 before stopping the old processes.
  There are 12 complete original score fits eligible for reuse, eight
  complete original cold pairs, and no original warm pairs. Added an archive
  README clarifying the incomplete original pairs and the corrected study.
- `audit_results.py` now verifies both copied and original score files
  against every `score_reuse.json` hash. The actual copied cold-start-0
  files pass this check. The sibling Pypomp worktree is clean at the recorded
  revision. No frozen fitting source changed in this continuation.
- The corrected worker 0's first six M-steps all increased their fixed Q
  objectives. This establishes progress of the corrected implementation,
  not an observed-likelihood or final-performance conclusion. Continue to
  evaluate completed pairs independently and inspect all final results.
- Gaussian J=500 oscillator fits and both Daphnia tuning recoveries remain
  active. The full final report and SI integration still await the declared
  experiments and Gaussian tuning choice. Goal remains active.

## October 9, 2026: corrected SAEM services launched

- The corrected experiment is now frozen in `ditlevsen/results/saem_ab_scaled`
  (source revision c24b11e9e, frozen in commit 1a227456e). Use this directory
  for the final A/B report and SI. The original unscaled results remain an
  implementation diagnostic; eight complete cold pairs are preserved.
  Do not modify the new frozen `ditlevsen/ditlevsen/*.py` while these workers
  are active or may need resume, unless another concrete defect warrants a
  separately documented experiment amendment.
- Persistent services `dmop-saem-scaled-w0` through `w3` are active, parent
  PIDs 1474287, 1474290, 1474327, 1474357 at 05:18 UTC. Logs are
  `/tmp/dmop-saem-scaled-wN.log`. They run `queue_saem_scaled.py`, which waits
  until the corresponding old worker has saved the current cold score fit
  (indices 8--11), stops only that superseded worker and runs the corrected
  worker on the same four-core group. Worker 0 has transitioned and begun
  the corrected run. Workers 1--3 still wait for their old score checkpoints;
  this preserves in-flight work. If an old worker terminates before saving,
  the new worker proceeds and fits any missing score arm normally.
- New preparation verified exact starts, score source hashes/settings,
  particle counts, budget and Pypomp revision for reuse. New workers verify
  the corrected source hashes at launch. All completed compatible score
  fits, including failures, are reused with provenance; no old-SAEM outputs
  are reused. Remaining score arms are fitted normally. The full target is
  still 20 cold and 20 warm pairs, not the eight completed old pairs.
- Local SI build succeeds with no overfull boxes or undefined/multiple
  references; visually reviewed the revised page S-76. Native compilation
  was attempted and returned `No handler registered`. Existing editor and
  document remain open. Five block-SAEM and four A/B tests pass.
- Other experiments continue unchanged. At 05:18 UTC oscillator J=500
  IFAD/IF2 had completed five starts; both Daphnia tuning recoveries had
  finished all four methods for start index 1 and begun index 2. All final
  report, Daphnia and Gaussian-selection queues retain their dependencies.
  Goal remains active; final outputs and their SI integration are pending.

## October 9, 2026: numerical SAEM defect confirmed and corrected

- IMPORTANT: the unscaled Dhaka A/B study is superseded by a corrected
  numerical M-step. Fixed-path replays of cold starts 2 and 5 reproduced
  L-BFGS-B convergence with exactly zero movement. Dividing Q and its
  gradient by the initial gradient infinity norm produced Q increases
  13968.79 and 74520.02 under the same 25-iteration limit. Saved diagnostics
  are `code/benchmarks/results/saem-scaling-{002,005}`; the replay script is
  `diagnose_saem.py`. These are fixed-path diagnostics, not fitted results.
- `maximize_objective` now fixes the divisor to max(1, initial gradient
  infinity norm) within an M-step, while acceptance uses the original Q.
  A regression test reproduces the original false convergence after an
  undefined line-search trial and checks the corrected optimum. The five
  block-SAEM tests pass. Four A/B tests pass, including reuse of a failed
  score fit without refitting or dropping it. No score kernel changed.
- Prepare the new study in `ditlevsen/results/saem_ab_scaled` with
  `--reuse-score-from ditlevsen/results/saem_ab`. Preparation verifies score
  settings, seeds, budget, particles, model source hashes, Pypomp revision
  and exact starts. Each reused score fit gets file hashes and source path.
  Old score/SAEM outputs remain in place; never pool the old SAEM arm with
  corrected results. The audit defaults to the corrected directory now;
  `--dhaka-ab-root` allows explicit inspection of the old study.
- At 05:15 UTC the old worker 0 was stopped after saving cold start 8's
  score fit (its subsequent partial old-SAEM trace is preserved). Old
  workers 1--3 still run their score arms for cold starts 9--11; stop them
  once their respective score.json exists, preserving those complete fits.
  Do not restart old workers against the modified source: they intentionally
  fail the frozen-source check. Existing processes still hold the old code.
  The corrected study is not yet prepared/launched at this checkpoint.
- The report now labels outer iteration counts explicitly as iterations,
  including iterations without parameter movement; it no longer implies
  they count accepted nonzero updates. This resolves the pending metadata
  interpretation issue noted in older entries below.
- SI S12.1 now gives the fixed scaling in the numerical-SAEM M-step.
  Compilation/visual review of this latest small edit is in progress.
  Eight original cold pairs are complete and audited. Full goal remains
  active, including corrected cold and warm comparisons, Gaussian selection,
  Daphnia, final panels/table and SI integration.

## October 9, 2026: J=500 linear comparison complete

- Verified the same live service/process handles throughout this continuation;
  no job was restarted. Both `matched-linear-j500-{if,ds}` batches completed
  all 20 starts without failed fits. The serial Gaussian service has advanced
  to `matched-oscillator-j500-if`. Six of eight matched batches are complete;
  the independent particle-count tuning still waits for the final two.
- Generated and visually inspected all three plots in
  `/tmp/dmop-linear-j500-complete`. Shared-start and completeness checks pass.
  At J=500, linear-model median log-likelihoods/times are IFAD -359.960533/
  16.085 s, IF2 -362.144909/13.261 s, and DS19 -359.927143/2.096 s. Retained
  CTDD21 J=25 gives -360.306505/51.631 s on its original core assignment.
  These are the completed J=500 results, not the tuning selection.
- Checked that observation arrays are exactly identical across all existing
  original and matched batches for each Gaussian model (five linear and
  seven oscillator batches at the time of the check). Starting vectors also
  pass the report's comparison. Recorded completed-batch checks in the
  temporary `/tmp/dmop-latest-evaluation-audit.json`; the source audit command
  remains in the benchmark README.
- Daphnia tuning start 1 has reached the competitor stage. DS19 at rate .01
  stopped with `nonfinite_update` after 101 updates, retaining its last valid
  estimate; at .001 it completed 525 updates within its roughly 305-second
  continuation budget. Do not select a rate until all tuning starts finish.
  The live Dhaka A/B experiment still has five complete cold pairs and no
  completed warm pairs. Full figures/table and SI integration remain pending.

## October 9, 2026: SI figure captions prepared; fifth A/B pair audited

- Added `imgs/benchmarks/figures.tex` with the final-distribution figure,
  detail and full-range progress figures, and central results table. Captions
  define boxes, failure markers, linear scales, interpolation, percentile
  bands, reference lines, Monte Carlo errors and timing boundaries. This
  fragment is deliberately not yet included in `si.tex`: generated final
  five-model artifacts are still pending. Compile and visually inspect it
  in the existing SI after all outputs are ready. Added the directory README
  with the regeneration command and artifact map. No SI source changed.
- Audited the fifth completed Dhaka A/B pair, cold start 4. Score stopped
  early with `non-finite-path-or-score`, retaining iteration 10. Its final
  Euler log-likelihood is -5831.464, versus numerical SAEM -8863.604 and
  initial -8887.596. Keep the failure status and last eligible estimate in
  the final report. All five completed pairs pass selection, M-step and
  raw-evaluation checks. There are still no completed warm-start pairs.
- At approximately 04:59 UTC, all seven fitting service handles remained
  active: matched Gaussian, both Daphnia tuning recoveries and four Dhaka
  A/B workers. Gaussian J=500 linear IFAD/IF2 is progressing through its
  20 starts; Daphnia tuning is fitting start 1. Queued final/report services
  remain dependent on these runs. This is a verified computational wait,
  not a failed or blocked experiment. The full goal remains unfinished.

## October 9, 2026: completed J=100 Gaussian runs and report checks

- Previous goal turn made progress: committed the per-model SAEM equations
  and completed SPX recovery. This continuation verified the persistent
  service handles and completed further reporting work. Goal remains active.
- All four `matched-{linear,oscillator}-j100-{if,ds}` batches are complete:
  20 starts for each of IFAD, IF2 and DS19 in each model, no failed fits.
  The serial service has moved on to `matched-linear-j500-if`. Do not choose
  the manuscript count until the separate tuning service writes its selection.
- `report.py` now verifies identical starting vectors wherever batches
  overlap, with matching parameter columns and a recorded vector for every
  evaluated start. It handles the older start files whose row order supplies
  the identifier. Provenance now hashes starting vectors, complete statuses
  and analytic references as well as evaluation/configuration files.
  Eight report and three recovery tests pass. Actual J=100 Gaussian and SPX
  data pass the common-start checks. No fitting code or protocol changed.
- Generated `/tmp/dmop-four-model-j100` from completed oscillator, linear,
  SPX and archived Dhaka data. Visually inspected all three panels and checked
  that every progress median/band is finite and ordered. All scales are
  linear. These are development previews: Daphnia is absent and the Gaussian
  count is not selected yet. Do not insert these previews as the final report.
- The audit now reports Dhaka numerical-SAEM iteration counts, Q increases
  above `1e-7`, solver convergence counts and total M-step evaluations. It
  also verifies the saved initial parameters and trace length. Among the
  first four completed cold pairs, start 2 had zero Q increases in 35
  iterations and retained its initial estimate despite solver convergence
  flags. Treat this as an implementation diagnostic, not proof about SAEM
  generally. Other workers are still fitting their second cold pairs.
- README now describes the actual CPU/core assignments, replacing outdated
  GPU timing wording, and gives the saved-evaluation audit command. The
  remaining full-goal deliverables and frozen-source restrictions below
  still apply. Daphnia tuning/final queues and Dhaka A/B remain live.

## October 9, 2026: SAEM equations and completed SPX runs

- Added S12.1 in `si.tex`: the SAEM objective recursion and M-step, the
  averaged-score recursion, and the distinction between reevaluating retained
  paths at a candidate parameter and retaining gradients at old parameters.
  Each example has its actual complete-path objective: explicit Gaussian
  regression update; oscillator sufficient-statistic maximization; SPX
  transition including its variance-floor atom; Daphnia block-Gaussian score;
  and Dhaka score versus numerical generalized SAEM. Included deterministic
  initial-state dependence, gain schedules, and acceptance rules. S11 now
  points to this explanation rather than implying that SAEM needs a
  closed-form M-step. No fitting code changed (only a sampler docstring).
- `make dmop/si.pdf` succeeds (78 pages), with no overfull boxes, undefined
  references or multiply defined labels in the build log. Visually reviewed
  pages S-73 through S-76 after fixing equation-reference wording and a long
  sufficient-statistic line. The built-in compiler stalled again and its
  call was cancelled. The existing source editor remains unchanged.
- SPX recovery completed at 04:43 UTC. All 80 final fits are complete, with
  20 starts per method. `audit_results.py` independently recalculates all
  final likelihoods and Monte Carlo errors from saved replicates; all pass.
  The script also checks completed Dhaka A/B selections, time eligibility,
  and nondecreasing numerical M-steps (four pairs passed at 04:50 UTC).
  README links the audit script. Inspect newly completed pairs again later.
- SPX median log-likelihoods: IFAD 11838.633, IF2 11841.959, DS19 11844.828,
  CTDD21 11796.975. No failed fits. Generated and visually inspected the
  three linear-scale SPX panels in `/tmp/dmop-spx-complete`; these are preview
  artifacts pending the combined report. The results do not support saying
  that all methods perform equally well on SPX. Preserve actual differences.
  The recovered start 4 used E-cores; all methods at that start used the same
  core group, recorded in execution provenance. Earlier SPX starts used
  P-cores. Do not generalize these timings beyond the recorded setting.
- Persistent services remain active. At 04:50 UTC, matched oscillator J=100
  IFAD/IF2 had reached start 17; both Daphnia tuning recoveries had completed
  warm compilation and started fitting start 1. Dhaka workers were fitting
  their second cold pairs. Daphnia final queues, Gaussian tuning queue, and
  final report queue still wait for their dependencies. Do not duplicate or
  restart live jobs. Full goal remains unfinished: final matched Gaussian
  selection, Daphnia runs, all 40 Dhaka A/B pairs, combined figures/table,
  final SI results prose and visual/compilation checks remain outstanding.
  The score update-count reporting correction below also remains pending.

## October 9, 2026: equal-particle Gaussian tuning and progress bands

- Latest user steering supersedes the earlier unequal-count Gaussian setup:
  DS19, IFAD and IF2 must use the same particle count in panels A/B. Compare
  100 and 500 as ordinary tuning, report the chosen common setting in the SI,
  and do not add a separate sensitivity section or a disclosure discussion.
  CTDD21 was not included in this new equal-count request and retains J=25.
- `matched_particles.py` runs both counts on all 20 original final starts,
  with all three methods on E-core group 28–31. IFAD/IF2 and DS19 are separate
  batches so their original 300/600 and 80 iteration schedules are retained.
  Service `dmop-matched-particles` is active, log `/tmp/dmop-matched-particles.log`.
  By 04:36 UTC both `matched-linear-j100-{if,ds}` batches were complete and
  `matched-oscillator-j100-if` was running. Remaining jobs are serial.
  These files are not named `final-*` and must not be pooled with old runs.
- `tune_particles.py` waits for the matched runs, then uses four separate
  IFAD starts per Gaussian model/count (seed 2026100901, versus final 631450).
  It selects a shared count by fewer failures, then the smallest sum of median
  deficits from the analytic maxima, then time. Service `dmop-tune-particles`
  is waiting on the same E-core group; log `/tmp/dmop-tune-particles.log`.
  Its output will be `results/matched-particle-tuning/selection.json` and a
  complete status. `dmop-final-daphnia-3` now waits for that complete status
  as well as both Daphnia tuning batches and SPX; other Daphnia queues unchanged.
  Confirm service handles before restarting; these are persistent jobs.
- `report.py` now uses the tuning selection by default for Gaussian IFAD,
  IF2 and DS19, verifies equal counts, and takes only CTDD21 from the original
  Gaussian runs. `--gaussian-particles 100|500` allows explicit inspection.
  The automatically queued complete report will therefore use the selected
  common count once the Daphnia batches complete. Eleven recovery/report
  tests pass, including rejection of unequal Gaussian particle counts.
- User requested the Dhaka SI's shading. Verified it is 10th–100th
  percentiles at alpha .10. Added `Axes.fill_between` accordingly in full
  and detail progress plots, with median curves. The detail window now
  starts below the smallest method final 10th percentile so the bands are
  legible; it stays linear, as does the full-range companion. Inspected the
  regenerated historical-subset preview in `/tmp/dmop-benchmark-report-preview`.
  Those previews still use the old unequal-count data for layout inspection;
  they are not manuscript results and must be replaced with completed matched
  results. Inspect every final image and its compiled SI page again.
- Updated S12's timing paragraph to equal counts selected on separate tuning
  starts, retained the different iteration schedules and cheap-M-step caveats,
  and added the median/band definition. Local `make dmop/si.pdf` succeeds,
  with no overfull boxes or undefined references. Visually inspected S-73.
  A second native compile call also stalled and was cancelled; no source
  editor/tab was replaced. Current source is 75 PDF pages, still awaiting
  final benchmark figures/table and Dhaka A/B results.
- First four global Dhaka A/B pairs are complete and workers continue to the
  next starts. Score/SAEM Euler log-likelihoods: start0 -4027.308/-5747.392;
  start1 -4010.903/-4326.347; start2 -4040.366/-8353.122; start3
  -3983.291/-7196.461. This early numerical-SAEM result is worse on all four;
  do not extrapolate to the unrun warm-start pairs. Starting-point and some
  poor-fit evaluations have MCSE near 1 from highly concentrated likelihood
  averages; inspect raw replicates before interpreting MC uncertainty.
  The metadata update-count reporting correction described below is still
  needed. Frozen fitting sources remain unchanged.

## October 9, 2026: recovery pipeline, figure scales, Gaussian timing caveats

- Active goal is to finish every benchmark and the Dhaka A/B experiment,
  generate the requested panels and central table, insert/review them in the
  existing `si.tex`, and compile it. This goal is unfinished. User additionally
  requested reasonable scales, visual inspection, and explicit caveats about
  the fast Gaussian DS19 timings and its smaller particle count.
- All four Dhaka A/B services from the entry below remain active. By 04:27 UTC
  each had finished its first score arm and was fitting numerical SAEM. The
  time budget includes filtering and backward simulation. Do not modify the
  frozen `ditlevsen/ditlevsen/*.py` sources while these workers may need resume.
  Reporting issue to fix after completion: score `updates` currently stores
  the selected trace index, while `updates_attempted` actually stores the
  fitter's total accepted updates. For the selected parameter vector, count
  positive `accepted_step_size_trace[:selected_index]` in the saved NPZ.
  The parameter selection and elapsed-time eligibility are unaffected.
  Preserve raw metadata and document the reporting correction before using
  update counts in the SI table.
- Added/tested `code/benchmarks/recover.py`. It archives the complete original
  partial batch under `results/interrupted/`, retains all fully evaluated
  starting points (including failed fits), and reruns only the unfinished
  suffix using its original seeds/settings. After completion it merges the
  suffix, checks all expected method/start pairs, and writes recovery provenance
  before marking the original batch complete. Interrupted individual fits
  within an unfinished start are retained in the archive, not selected by score.
- Persistent recovery services: `dmop-recover-final-spx-00` (cores 24–27,
  reruns start 4), `dmop-recover-tuning-daphnia-cpu` (16–19, starts 1–3), and
  `dmop-recover-tuning-daphnia-cpu-lr0001` (20–23, starts 1–3). Logs are
  `/tmp/dmop-recover-NAME.log`; suffix outputs are `results/recovery-NAME`.
  SPX's recovered IF2 fit was evaluated by 04:27 UTC. Its four methods share
  the E-core assignment, unlike the original P-core SPX batches; timing
  heterogeneity is recorded in `results/execution.json` and recovery configs.
- Persistent `dmop-final-daphnia-0` through `-3` queue the 20 final starts in
  five-start batches on E-core groups 16–19/20–23/24–27/28–31. All wait for
  both tuning batches and SPX batch 00 to finish. Their commands use
  `queue_final.py`; tuning chooses each competitor's rate separately, by
  failures then median independent likelihood. Logs `/tmp/dmop-final-daphnia-N.log`.
  No final batch has started yet. Do not duplicate these queues.
- `dmop-benchmark-report` runs `finish_report.py`, waiting for SPX and all four
  Daphnia batches, then generating the complete five-model report under
  `imgs/benchmarks`. Log `/tmp/dmop-benchmark-report.log`. It does not insert
  anything into the manuscript. Dhaka A/B workers generate their own report
  when all 40 pairs finish. Both reports still require scientific review.
- Report checks now reject invalid MC errors and times, and verify recovery
  configuration hashes using checkout-relative paths. Ten recovery/report
  checks pass, covering failed-fit retention, missing/duplicate data and
  incomplete batches. Plot layout is sized for manuscript width. All axes
  are linear; final distributions retain every point. `optimization.pdf`
  shows full median trajectories; `optimization_detail.pdf` uses a separate
  labeled window spanning method final medians through the reference, with
  padding. `plot_ranges.json` records the rule and limits. Visually inspected
  all three completed-subset preview images in `/tmp/dmop-benchmark-report-preview`.
  Inspect every final figure again after full generation, including A/B.
- Added SI S12 with Gaussian timing caveats: DS19 100 particles/80 SAEM steps;
  IFAD 500 particles/100 IF2 plus 300 gradient steps; IF2 500/600; CTDD21 25/300.
  No particle-count sensitivity study was done. DS19's filter and backward
  pass are timed; its M-steps exploit sufficient statistics (explicit linear,
  three-parameter oscillator). All methods exclude compilation/evaluation.
  These are configuration-specific timings, not a general speed ranking.
  Built the existing SI in place (75 pages) and visually inspected page S-73.
  Native compile was called but stalled for several minutes and was cancelled;
  repository `make dmop/si.pdf` succeeded. The source editor remains open.
- Next: monitor live service PIDs and logs; finish/review results; add model
  settings and actual results to S12, with final panels/table and the Dhaka
  A/B subsection. Follow the MS voice and completed data, without assuming
  saturation. Build and inspect the final SI pages, save all compact source
  records/results, and commit/push. Do not mark the active goal complete until
  all requested results and manuscript figures/tables are present and checked.

## October 9, 2026: interrupted workers and persistent Dhaka restart

- Status audit found all previous Dhaka, SPX and Daphnia tool-session
  workers had exited. The original sessions no longer exist; logs contain
  no traceback. The interruption coincided with the previous turn ending,
  but its cause has not been established. No production Dhaka A/B arm or
  pair finished. The October 8 launch status below is historical.
- Restarted all four Dhaka workers at 00:05 EDT (04:05 UTC), using persistent
  systemd user services `dmop-saem-ab-w0.service` through `w3.service`.
  Verified all four active/running in a separate call. They use the unchanged
  frozen protocol, ten pairs per worker, CPU groups 0–3/4–7/8–11/12–15,
  and append to `/tmp/dmop-saem-ab-workerN.log`. No SPX dependency is needed
  now. Check `systemctl --user status dmop-saem-ab-w{0,1,2,3}` on the host;
  do not launch duplicates. Completed arms and pairs resume automatically.
- Oscillator and linear-Gaussian runs remain complete. SPX has 76/80 final
  fits evaluated: batches 05/10/15 complete, batch 00 has four of five starts
  complete and the fifth IF2 parameter trace saved. Preserve these partial
  outputs when implementing recovery; the small-model runner currently
  refuses an existing output directory and has no resume option.
- Both four-start CPU Daphnia tuning runs stopped after one fully evaluated
  start, with additional partial parameter traces saved. No final Daphnia
  batch ran. Those tuning runs and dependent queues still need recovery;
  they are not running. The combined five-model panels and central SI table
  remain unfinished. No manuscript results were changed in this status audit.

## October 8, 2026: paired Dhaka A/B experiment

- User requested score versus numerical SAEM both with and without an IF2
  warm start. Added `ditlevsen.saem_ab` and `saem_ab_report`: 20 shared starts
  in each regime, 1,000 particles, 850 seconds per arm, independent Euler-20
  evaluation with 5,000 particles and 36 paired replicates. Initial values
  are evaluated too. Fits retain the last finite estimate within the budget;
  failures remain in the comparison. Method order alternates within workers.
- Score settings are the archived global/warm settings, including burn-in 30
  and learning rates .1/.0005. Numerical SAEM uses pilot settings: burn-in 3,
  at most 80 updates and 25 L-BFGS steps per update. This compares optimizer
  packages, including schedules, with the Gaussian transition fixed. IF2
  inputs are precomputed; their original cost is excluded from both warm arms.
- Numerical SAEM now differentiates each path before summing, avoiding an
  outer reverse-mode tape over all retained paths. Tests confirm the same
  weighted objective and derivative. Timed score fits explicitly compile the
  newly created derivative closure before starting their timer; the optional
  flag leaves historical fitting defaults unchanged. Deadlines discard an
  incomplete SAEM M-step. All completed fits are saved before evaluation.
- Thirteen fitting/selection checks pass. A complete low-particle workflow
  check with a deliberately tiny budget passed in `/tmp/dmop-saem-ab-smoke-v2`,
  including both initialization regimes, all evaluations, and report generation.
  Additional paired-MC-error check verifies cancellation with shared draws.
- Production output is `ditlevsen/results/saem_ab`, prepared from commit
  `16a4fa043`. Four workers were launched at 21:50 UTC. Workers 0/2/3 are
  fitting on P-core groups 0–3, 8–11 and 12–15; worker 1 waits for SPX batch 00
  before using cores 4–7. Tool sessions are 45921/75821/71296/94968 for workers
  0/1/2/3, and logs are `/tmp/dmop-saem-ab-workerN.log`. Host PIDs at launch
  were 1156468/1156464/1156469/1156457. No production A/B pair is complete yet.
  Protocol preparation snapshots sources and inputs; workers reject changed
  sources before fitting. Completed pairs/arms can be resumed. The last worker
  automatically produces linear-scale panels and tables under `report/`, with
  a file lock preventing concurrent report writes. These still need scientific
  review before updating the SI.
- SPX batches 05/10/15 are complete (60 fits); batch 00 continues. Daphnia
  tuning and its four dependent final batches continue unchanged. The new
  five-model panels and central SI table remain pending those complete runs.

## October 8, 2026: numerical generalized-SAEM pilot

- Added `ditlevsen.block_saem`: weighted complete-path objective averaging,
  reevaluated at every candidate parameter, with a numerical L-BFGS M-step
  accepted only when finite and nondecreasing in that objective. The monthly
  Gaussian transition, guided proposal and backward sampler are unchanged.
  This is a generalized-SAEM extension, not a claim of DS19 convergence for
  Dhaka. Archived score-based fits remain the current SI results.
- Three tests pass: known Gaussian M-step, averaging-weight reset, and the
  Dhaka objective/derivative compared with separately evaluated paths. The
  latter uses a 1e-8 covariance floor to isolate averaging from ill-conditioning.
  A diagnostic at the unchanged 1e-12 pilot floor found about 1.6e-6 relative
  gradient differences between equivalent compiled evaluation orders on a
  three-month example; at 1e-8 this was about 1e-10. This is a conditioning
  diagnostic, not evidence that changing the floor improves estimation.
- Eight-update, 100-particle pilots at IF2 starts 0 and 1 are complete in
  `ditlevsen/results/block_saem_pilot_start{0,1}`. They used CPU cores 24–27
  and 28–31, respectively, and at most 25 L-BFGS steps per M-step. All stored
  parameters are finite and all M-steps retain or improve the fixed Q objective;
  four of eight M-steps reported convergence in each run. Initial/final Euler
  evaluations use 5,000 particles and 24 reps: start 0 changes -3769.754 to
  -3769.846 (MCSE .151/.141), and start 1 changes -3761.260 to -3760.267
  (MCSE .158/.228). Fitting times are 349/292 s, excluding preliminary
  compilation and evaluation. These two short pilots establish feasibility,
  not convergence, a runtime ranking, or resolution of the Dhaka comparison.
  Source snapshots, parameter traces and diagnostics are saved. Current S11
  results are unchanged; all new five-model reporting must distinguish the
  SAEM and numerical-score implementations. Follow-up needs more starts and
  iterations, followed by a separate study of transition approximation error.
- Validated cheaper Daphnia trace evaluations with a small all-method run:
  final and trace replicate counts match their separate requested settings.
  Final production evaluations remain 2,000 particles x 10 reps; traces use
  500 x 2 every 50 updates. Existing tuning processes retain their older code.
- Completed Gaussian SAEM results and the last oscillator batch are saved.
  SPX and Daphnia computations continue; the five-model SI panels/table remain
  pending complete results and scientific review.

## October 8, 2026: linear axes and Figure S-6 repair

- User rejected the hybrid linear/log scale. Current `code/benchmarks/report.py`
  plots raw log-likelihood on ordinary linear axes for both final distributions
  and progress. Rebuilt and visually checked the completed Gaussian/Dhaka
  preview in `/tmp/dmop-benchmark-report-gaussian`; no new panels inserted yet.
  Historical run source snapshots remain unchanged.
- Fixed Figure S-6: equal-area violin scaling flattened the broad distributions,
  and the method-axis limit cut off the top IFAD violin. Each violin now has the
  same peak height, with room above the top method. Regenerated the cropped and
  full-range PDF/PNG companions; caption states the density scaling. All 100
  values per method still contribute. Twelve plotting checks pass. SI builds
  in place with `make dmop/si.pdf`; native compiler still lacks external macros.tex.
- User asked whether the Dhaka DS19 adaptation can be corrected. Existing
  implementation averages scores at changing parameters and applies Adam; it
  is not SAEM. Plan a separate pilot averaging complete-path objective functions
  and numerically maximizing each averaged objective. This isolates optimization
  from the monthly Gaussian approximation, which remains a separate limitation.
  Do not present poor Dhaka performance as a general failure of DS19.
- All Gaussian final fits, including the replacement SAEM fits, are complete.
  Four SPX final batches and two Daphnia tuning runs continue; final Daphnia
  batches wait on tuning. Their source snapshots and outputs are separate.

## October 8, 2026: README for the Dhaka global search

- Added `code/global_search/README.md`. It separates the files behind the
  main-text Dhaka table and figures from historical ones (`report.qmd`,
  `report.sbat`, and their Makefile targets), notes the two unused
  comparable-effort plots, and records the environment (Pypomp `17f8798`,
  code identical to v1.0.5; JAX 0.11.2). Linked from `code/README.md`.


## October 8, 2026: final batches queued; Gaussian DS19 uses SAEM

- Final linear runs and oscillator batches 00/05/10 are complete; oscillator
  batch 15 and SPX batches 05/10/15 are running. CPU-only Daphnia tuning at
  competitor learning rates .01 and .001 is running on disjoint efficiency
  cores. Its full-shape preliminary fits are expensive (roughly 20 minutes);
  final Daphnia fits have not started. Measured CPU baseline times are about
  185 s for the MPIF warm start, 640–680 s for full MPIF, and 278–296 s
  for IFAD continuation. Each competitor receives that continuation budget.
  Expect several hours for tuning and final fits. See `results/final_plan.json`.
- `queue_final.py` waits for declared predecessor directories and separate
  four-start tuning runs. It chooses fewer failed fits, then higher median
  independent likelihood, writes the choice, and launches final five-start
  batches on disjoint cores. Four Daphnia batches and SPX batch 00 are queued.
  Queue logs are `/tmp/dmop-queue-*.log`; final runs preserve source snapshots.
- Scientific correction before reporting: Gaussian DS19 comparisons now use
  an actual SAEM M-step, rather than the generic numerical-score extension.
  `saem.py` averages Gaussian sufficient statistics and maximizes the expected
  complete log likelihood (explicitly for linear Gaussian; L-BFGS-B for HO).
  Uses the paper's 80 iterations / 100 particles / 30-step burn-in. Two tests
  verify the likelihood, derivatives and maximization against direct paths.
  Four oscillator tuning starts finish within .1 log unit of the exact
  likelihood reference. Original score-update runs remain archived.
- Twenty-start SAEM runs are queued on cores 4–7 after oscillator batch 15,
  followed by SPX batch 00. Combined reporting explicitly requires the SAEM
  files and excludes Gaussian DS19 score rows; other methods are unchanged.
  SPX/Daphnia/Dhaka still use the disclosed numerical-score extension.
- Eight separate deterministic optimizations recover the same Kalman
  reference maximum for each Gaussian model. Numerical test total is now 11;
  the updated Daphnia batch/checkpoint workflow also passed a small smoke run,
  including retention of a failed DS19 fit's last finite estimate.
- No new cross-model SI section, combined publication plots, or central
  table has been inserted yet. Those require complete final runs, review,
  and compilation. Theorem/legend/documentation work from earlier entries
  is complete. Preserve the existing editor and compile SI in place later.


## October 8, 2026: cross-model runs resumed

- Completed one-start oscillator/SPX workflow checks and four-start linear
  tuning. These are pilots, not manuscript results. The linear tuning puts
  IFAD, DS19 and CTDD21 near the analytic maximum; IF2 is more variable.
- Final linear run completed all 80 fits with no nonfinite failures in
  `code/benchmarks/results/final-linear`. Oscillator tuning also completed;
  final five-start batches 00, 05, 10 are running on cores 4–7, 0–3, 12–15.
  Batch 15 remains to launch. Four-start SPX tuning uses cores 8–11. Seeds/settings/hashes and incremental status are in each
  output directory. Logs are `/tmp/dmop-{final-linear,tuning-oscillator,tuning-spx}.log`.
  GPU linear tuning was interrupted explicitly; its partial folder is marked.
- Added Daphnia adapters and runner. CTDD21 and all independent evaluations
  reuse original S10 Euler components. DS19 uses composed strong-1.5 Gaussian
  moments, NB-guided proposals, backward paths and numerical score ascent;
  its boundary treatment differs and its fitting objective is approximate.
  The original SAEM M-step is not claimed. A small all-method pilot completed.
- GPU Daphnia tuning was interrupted after another GPU workload began. Its
  partial directory is marked and no final result was selected. Four-start
  CPU tuning now runs on efficiency cores 16–19, log
  `/tmp/dmop-tuning-daphnia-cpu.log`. IFAD, DS19, CTDD21 share the same MPIF
  warm estimate; competitors receive the measured IFAD continuation budget.
  Existing S10/S11 studies are preserved. Runtime comparisons must stay within
  the new experiment; older V100 times are a different hardware setting.
- Excel reader `xlrd` 2.0.2 is installed only in `/tmp/dmop-benchmark-deps`;
  Daphnia commands add that to PYTHONPATH. Nine adapter/likelihood tests cover
  Gaussian likelihoods, smoothing scores, SPX clipping, original Daphnia drift
  and measurements, block covariance, and original-PF evaluation agreement.
- Added the combined report generator. It refuses incomplete/duplicate final
  starts and unresolved likelihood evaluations. Checked its Dhaka-only output
  against the original manuscript exports; recovered baseline MC errors from
  the exact archived result objects, and retained the two CTDD21 searches
  that reached the restart limit. Nine numerical checks pass. A disjoint-batch
  runner smoke check also passes. Source snapshots preserve final-run hashes.
- SPX tuning at learning rate .01 made IFAD deteriorate; .001 improved the
  all four separate tuning starts substantially. The full-method .01 pilot
  is still running before SPX settings are frozen. No saturation claim is warranted yet.
- New cross-model panels, central table and SI discussion remain pending
  complete final runs. Do not claim saturation from pilots or omit failures.


## October 8, 2026: simplify Theorem 5 and reorganize S6–S7

- Applied the chat draft after Kevin approved it. Main text now has a short
  Theorem 5, a compact bound table, interpretation for each alpha case before
  the proof-method discussion, and a brief explanation immediately after A6-J.
  Explicit mixing constants and finite-sum refinements remain in S7.
- The table distinguishes A6 and A6-J: the DMOP-0 particle MSE for J >= N
  has two logarithmic factors under A6 and one under A6-J. S6's theorem now
  states the two variance bounds already established by its proof.
- Retitled S6 as the DMOP-0 variance warmup. Removed the duplicate old
  Theorem S4 and the inactive superseded S6 extension. S7 is now titled as
  the proof of main Theorem 5; its retained general theorem renumbers from
  S5 to S4. Removed/redirected references to the deleted result. Section
  numbers S6/S7 and downstream experimental sections are unchanged.
- Validation: repository builds succeed (MS 23 pages, SI 74), with no
  undefined references/citations or new overfull boxes. The main manuscript
  retains the existing Theorem 3 proof overfull box. Inspected the rendered
  bound-table/discussion page and S6 opening. `git diff --check` passes.
  Native SI compilation was attempted but cannot load external macros.tex;
  the existing editor was retained. No replacement document was created.
- Cross-model benchmarking below is still unfinished; this edit adds no new
  experiment results or figures.

## October 8, 2026: cross-model benchmarks in progress; theory draft history

- Kevin expanded the requested oscillator plots to a comparison of IFAD,
  IF2/MPIF, DS19 and CTDD21 across the DS19 harmonic oscillator, the Pypomp
  SPX implementation, the CTDD21 linear Gaussian estimation example,
  Daphnia, and existing Dhaka results. Requested consistent panels and a
  central table. The proposed saturation of simple examples is a hypothesis,
  not a completed finding. Daphnia must include DS19 and CTDD21.
- New `code/benchmarks/` holds a reporting protocol, Gaussian model adapters,
  a CTDD21 filter that reuses actual Pypomp components, backward simulation
  and numerical complete-data scores, and a serial repeated-fit driver.
  SPX is `pypomp.models.spx()` with its bundled data, not a replacement model.
  The oscillator uses its exact position observations and a conditional
  velocity proposal. Analytic evaluation targets its strong-1.5 approximation.
- Six CPU tests pass: independent Gaussian likelihood checks, conditional
  oscillator PF agreement, CTDD21 finite gradients, smoothing-score agreement
  with Kalman, and the SPX clipping atom. A two-start, ten-update linear
  pilot completed all four methods in `/tmp/dmop-linear-pilot-1008`.
  These are validation/pilot outputs, not manuscript comparison results.
  No new SI plots, table, conclusions, or manuscript edits have been made.
- Remaining: review and tune the fit driver on separate starts, validate
  full-series SPX and oscillator fitting, implement/validate Daphnia extensions,
  perform repeated final fits and independent evaluations, then make panels
  and table and update SI. The numerical-score DS19 extension is explicitly
  distinguished from the original paper's SAEM M-step. Measure timings on
  uncontended hardware; do not rank V100 and RTX3090 timings as comparable.
- Earlier request (now implemented above): draft a shorter Theorem 5 in chat BEFORE editing. Include
  an informal bound table, interpretation for each alpha case before proof
  discussion, and concise method/difficulties rather than a proof sketch.
  Explain A6 versus A6-J briefly immediately after the assumptions and in the
  proof-method discussion. S7 states common particle rates under either;
  S6 additionally proves the sharper DMOP-0 variance bound with one factor
  `(1+log J)` under A6-J versus its square under A6. Preserve that distinction
  in the proposed table. Constants and proof route differ. A6 uses an age decomposition with
  geometric sensitivity decay; A6-J treats the gradient as a bounded current
  function of the joint state/tangent process. Neither is needed for the
  population discount-bias argument. Kevin subsequently approved this draft;
  see the completed edit above.
- Kevin also proposes removing the currently numbered Theorem S4
  (`thm:dmop-alpha-truncation-mse`), whose general bounds repeat S7, and
  reframing S6 as a DMOP-0 warmup. This structure was drafted in chat and
  implemented after approval, as recorded above.

## October 8, 2026: use DS19 and CTDD21 algorithm names

- Defined DS19 and CTDD21 alongside the author citations in `ms.tex` and
  S11. Changed algorithm mentions, paragraph headings, captions, and both SI
  tables to the abbreviations. Author citations and bibliography are intact.
- Added display-name mappings to the current SI table and plot generators,
  preserving historical result keys and source paths. Regenerated all eight
  PDF/PNG pairs in `imgs/competitors/corenflos100_ditlevsen1000/`, including
  the four SI figures and the uncropped/diagnostic companions. Updated the
  exported method labels and current figure/navigation documentation.
- Verification: numeric CSV columns and fitting configurations are unchanged;
  all eight PDFs contain DS19/CTDD21 and no old algorithm labels. Reviewed
  likelihood and parameter-density plots. The 12 existing settings/warm-plot
  tests pass, and `git diff --check` passes. Repository builds succeed
  (MS 24 pages, SI 75); SI has no undefined references/citations or overfull
  boxes. MS retains its pre-existing proof overfull box. Native compilation
  still cannot load external `macros.tex`; existing SI editor retained.
- Merged concurrent manuscript/proof/citation edits through `95b054e55`,
  preserving both handoff entries. Rebuilt main and SI after the merge;
  the naming diff against that remote revision passes `git diff --check`.
- Historical experiment folders/plots and internal data identifiers retain
  their original names. Current manuscript output is generated by
  `corenflos.settings_plots` and `code/competitor_results.py`.

## October 8, 2026: align the Del Moral--Jasra citation with the SPA article

- Checked the journal and preprint PDFs in `dmop_private/refs`. Updated
  the bibliography to Del Moral and Jasra (2018), including the Part I
  title, and replaced `chanDelMoral18` with `delmoral18` in main and SI.
- Mapped preprint Theorems 3.3 and 3.5 to journal Theorems 3.1 and 3.3,
  including the SI's subsequent theorem calls without repeated citations.
  Updated the inactive historical proof's Equation (3.9) to (3.10);
  the backward representation remains Equation (3.3) in both versions.
- Main and SI compile successfully (24 and 75 pages), with no undefined
  references or citations. Existing layout warnings are unchanged.
  Checked both generated bibliography entries and `git diff --check`.

## October 7, 2026: incorporate the revised Particle Marginals lemma

- Updated Lemma S6 and its proof on top of `828e6669d`, after comparing
  Kevin's latest manuscript changes with the October 6 review snapshot.
  The targeting section was unchanged; the SI additions were in S11.
- The lemma now states the conditional sampling law given the input
  particles and weights and requires weights uniformly bounded above and
  away from zero. The proof controls the centered propagation error by the
  conditional fourth-moment argument of Lemma S5.
- Added the approved remark explaining the standard particle-filter/MOP
  sampling condition and the weaker bound on normalized weights. All new
  text is black. Existing lemma and equation labels are preserved; the
  standalone note's references were converted to manuscript cross-references.
- Source before S6 and from Lemma S7 onward is byte-for-byte unchanged.
  Main text, S11, figures, code, and experimental results are unchanged.
- Validation: full SI builds to 75 pages, with no undefined references or
  citations and no overfull boxes. Existing theorem-style and PDF-string
  warnings match the baseline. All existing label numbers match the baseline;
  visually checked pages S15--S17. `git diff --check` passes.

## October 7, 2026: organize Fig. 1A and comparison documentation

- Reworked root and `code/README.md` as reader-facing navigation, retaining
  the old code README at the bottom as requested. Daphnia/global-search code
  is linked, not reorganized. No algorithm, script, result, or plot file moved.
- Replaced the long Ditlevsen/Corenflos landing pages with current SI maps;
  preserved the earlier versions in clearly marked `DEVELOPMENT.md` files.
  Added package, result, launch-script, IF2-reference, Fig. 1A data, and SI
  figure directory READMEs. Current/final runs are distinguished from pilots,
  tuning, abandoned QML, and earlier plot sets.
- Added `code/competitor_reproduction.md`: checked environment versions and
  sibling Pypomp revision/layout, portable interpreter commands, stage/input
  requirements, worker scheduling, mixed settings, and fresh-clone versus
  archived-data workflows. Original launchers still retain local interpreter
  paths; guides explain how to reuse their arguments. Bulk trace access is
  currently local, so publication of the revision archive remains open.
- Fig. 1A guide now maps data -> generator -> saved curves -> plot and gives
  preview/recomputation destinations. `imgs/095/README.md` identifies its
  manuscript asset; Fig. 1B is explicitly separate from this generator.
- Configuration audit found the separately tuned Ditlevsen J=5,000 run uses
  a 5,000-particle likelihood guard. Corrected the SI's claim that every
  particle count uses a 100-particle guard. Also corrected Corenflos notes
  to distinguish global versus IF2-warm-start output selection.
- Verification: all 204 local documentation links resolve to tracked/new
  paths, shell snippets pass `bash -n`, and four documented CLI entry points
  import and show help on CPU. In an isolated tracked-file export, Fig. 1A
  and both SI tables/CSV summaries reproduce byte-for-byte. The comparison
  plot command, using local bulk traces and a temporary output, reproduced
  all eight PNGs and five numerical/provenance exports byte-for-byte.
  No fitting experiments were rerun. `git diff --check` passes. Repository
  SI build succeeds (75 pages), with no undefined refs/citations or overfull
  boxes. Native compiler still cannot load external `macros.tex`; same editor
  retained. New package installation and GPU reruns were not attempted.
- Next: publish the existing bulk-output archive with the revision release
  so external readers can run full trace plotting/audits without contacting
  the maintainer. Ed plans to arrange Zenodo; no DOI, deposit, email, or other
  external publication was created. Other manuscript code remains with its
  existing owners for the broader documentation review.

## October 7, 2026: rewrite all of S11 after prose feedback

- User found the proposal/filtering description awkward and requested a
  review of the whole section. Rewrote S11 throughout against the main
  manuscript's methods and Dhaka application, including the introduction,
  both methods, experimental settings, results, and captions. The title is
  now “Comparison with Other Methods for the Dhaka Model.”
- Separated the Gaussian approximation, observation-conditioned proposal,
  and score updates into distinct explanations. Checked the proposal against
  `ditlevsen/ditlevsen/block_smc.py` and `ditlevsen/method.tex`: measurement
  variance is calculated using predicted deaths, not set equal to deaths.
  Explained the Corenflos differentiation and restart steps as actions rather
  than lists of implementation terms. Results now introduce the plotted
  settings before discussing the other particle counts.
- Tables, numerical results, figure files, and reference labels are unchanged;
  edits are confined to S11 prose. `git diff --check` passes. Repository SI
  build succeeds (75 pages), with no undefined references/citations, overfull
  boxes, or oversized floats. Reviewed rendered S67--S70. Native compilation
  still fails because it cannot load external `macros.tex`; editor retained.
- Next: author review of the revised section.

## October 7, 2026: match SI comparison prose to the manuscript

- Replaced “Ditlevsen-style” with “Ditlevsen” throughout S11, its tables,
  and the table generator/CSV exports. The methods describe the block
  approximation and numerical score ascent; captions no longer repeat
  qualifications about the method name.
- Read the introduction, Dhaka application, computational efficiency, and
  discussion in `ms.tex` as the writing sample. Revised S11 toward the same
  direct experimental prose, shorter figure captions, and consistent
  “log-likelihood” terminology. Reduced lab-note wording and repetitive
  qualifications. The reproduction paragraph points to the repository
  documentation for detailed scripts and asset paths.
- Existing plot assets, experiment settings, and all numerical table/CSV
  results are unchanged. Ran `code/competitor_results.py` and checked that
  the four generated outputs differ only by the requested method label.
- Verification: `make dmop/si.pdf` succeeds (75 pages), with no undefined
  references/citations, overfull boxes, or oversized floats. Reviewed the
  rendered results, table labels, and figure captions. `git diff --check`
  passes. Native compilation still cannot load external `macros.tex`;
  the same SI editor remains open and the repository build works.
- Next: editorial review of S11. No new fits or figures are needed.

## October 7, 2026: use status agent's mixed-setting plots in the SI

- Applied the user's request to update SI plots from the “Check Ditlevsen
  status” chat. S6--S9 now directly include that agent's four existing PDFs
  from `imgs/competitors/corenflos100_ditlevsen1000/` (commit `51e4a6e95`).
  Both Corenflos variants use J=100 and both Ditlevsen variants J=1,000;
  IFAD uses J=5,000. No plotting assets or fitting results were changed.
- Updated captions and reproduction notes for the settings, coordinate
  zooms using all 100 starts, and omission of the historical reference line.
  The results tables retain all tested configurations. Updated the figures
  README to identify the set now included in the SI.
- Verification: `make dmop/si.pdf` succeeds (75 pages), with no undefined
  references/citations, overfull boxes, or oversized floats. Visually checked
  S6--S9 on pages S70--S72. `git diff --check` passes. The native editor
  compiler still cannot resolve external `macros.tex`; the existing editor
  remains open and the repository build provides the compiled SI.
- Next: editorial review. The prior entry's “SI still uses its prior sets”
  note is superseded by this integration.

## October 7, 2026: mixed-setting comparison figures

- User requested a new set with J=100 for both Corenflos variants and J=1,000
  for both Ditlevsen variants. Standalone and IF2 warm start remain separate
  designs; these particle settings are fixed within each family, not chosen
  separately by initialization. No fits, evaluations, SI, or existing source
  figure files were changed.
- Added `corenflos.settings_plots`, using the established `warm_plots.py`
  helpers with optional settings labels/PDF output. Legacy defaults remain.
  Eight PNG/PDF pairs are in `imgs/competitors/corenflos100_ditlevsen1000`:
  final likelihood overview/uncropped, optimization overview/continuation
  zoom/uncropped, parameter densities, and two objective-mismatch views.
- Every method's label specifies its fitting particle count. All 100 starts
  contribute; final evaluation is 5,000 x 36, trajectories 5,000 x 1.
  New raincloud overviews use coordinate zooms, not the old pre-summary
  -4300 data filter. Uncropped views retain final tails down to about -24636
  and initial trace envelopes near -15307. No in-plot explanatory prose.
  The obsolete -3744.17 reference line is omitted only from the new set.
- Exported settings/configuration provenance, all four final-output CSVs
  combined, compact trace summaries, and the paired warm-start diagnostic.
  Median final likelihoods: Corenflos -4554.674, Corenflos + IF2 -3765.914,
  Ditlevsen -3998.931, Ditlevsen + IF2 -3769.038. Rates respectively .02,
  .0002, .1, .0005. Existing selection rules and wall-time budgets preserved.
- Verification: generator validates complete unique start identities,
  evaluation status/effort, fitting counts, and finite final values. All
  eight PNGs visually reviewed, including settings labels, visible standalone
  Corenflos median, and separated warm-start traces. Both package test suites
  pass (86 tests); `git diff --check` passes. See `imgs/competitors/README.md`
  for filenames, reproduction, and the necessary local archived trace inputs.
- Next: use/review this separate figure set as requested. SI still uses its
  prior J=100/J=1,000 sets; replacing those requires a separate user request.

## October 7, 2026: restore existing SI figures and document Fig. 1A

- User correction: use the previous Ditlevsen/Corenflos agent's figures;
  do not invent replacement plots for the SI. Removed the new four-panel
  plots and their plotting code. S6--S9 now include the existing likelihood,
  full optimization, focused continuation, and parameter-density PNGs for
  J=100 and J=1,000 directly from the two comparison result directories.
  All eight source PNGs are unchanged. S7 is now optimization progress.
  Captions disclose the original plot cutoffs, including the absent J=1,000
  standalone Corenflos median. Tables still summarize all 100 runs.
- `code/competitor_results.py` now generates only tables and CSV summaries.
  The new S11 methods/results text and tables are retained. The earlier
  entry's claim of new SI figures is superseded by this correction.
- Fig. 1A now says both "similar to Poyiadjis, 2011" and "similar to
  Naesseth, 2018", per the user's follow-up. `code/fig1a/` contains the
  recovered model/filter logic, original input CSVs, style, provenance,
  full-generation script, and cached numeric curves. `python code/fig1a/plot.py`
  redraws `imgs/095/mop.png` without JAX or the old checkout. README links it.
- A complete 10,000-particle GPU generation produced the committed curves.
  A second uncontrolled GPU run differed despite fixed seeds; the generator
  now fixes float32, non-partitionable Threefry, and deterministic XLA settings.
  A replay matched the first three grid points exactly before being stopped
  as redundant; a complete replay with those flags has not been checked.
  Cached-CSV redraw is byte-identical in a clean export. No bitwise numerical
  agreement with the old notebook across JAX/device versions is claimed.
- Merged and preserved remote caption/title edits through `67c3e578a`.
  Initial S11 work and the merge were pushed as `344fc3732` and `1c30c461f`.
- Validation: table generator checks pass, Python sources compile, and
  `git diff --check` is clean. MS builds to 23 pages and SI to 76 with the
  repository TinyTeX toolchain. Final reused-figure pages S70--S73 were visually
  inspected. SI has no undefined references/citations, overfull boxes, or
  oversized floats. MS retains its pre-existing proof overfull box.
  The native editor stays on `si.tex`; its single-file compiler cannot access
  `macros.tex`, so its preview remains unsupported for this multipart project.
- Next: editorial review of S11; no fitting experiments are running for this
  task. Future SI figure changes should use the established plotting workflow.

## October 7, 2026: add the Dhaka competitors to the SI

- Added Section S11 to `si.tex`, with a pointer in `ms.tex`: methods and
  qualifications for the Ditlevsen-style block-SMC adaptation and Corenflos
  transport filter; standalone and exact-IF2-warm-start designs; completed
  100/1,000-particle results and the separate 5,000-particle Ditlevsen run.
- Added two tables and two figures, generated by `code/competitor_results.py`
  solely from tracked compact CSVs. All 100 starts and early-stop outputs
  contribute; paired changes use matching start identities. No plot prose
  was added beyond axis labels and panel letters. Source paths are exported
  in `imgs/competitors/summary.csv`; see that directory's README.
- Used the current IFAD-0.97 reference (median -3744.40, maximum -3743.68),
  which agrees with `imgs/precise_table.tex`. Older experiment README prose
  quotes superseded IFAD values; do not copy those into the manuscript.
  The exported historical IF2 comparator also differs from the current main
  table, so S11 uses only the exact intermediate IF2 checkpoint and IFAD-0.97.
- Disclosed objective differences, noisy fitting values, changed update counts,
  early-stop/selection rules, and the standalone Ditlevsen scheduling caveat.
  These are results for the implemented adaptations, not universal comparisons.
- Generator validates complete unique starts, successful independent 5,000 x 36
  evaluations, finite outputs, and agreement of paired changes with the saved
  mismatch CSVs. Python compilation and `git diff --check` pass. Both documents
  build with TinyTeX/latexmk (MS 23 pages, SI 74 pages); new pages and Fig. 1
  were visually inspected. The native single-file compiler cannot resolve
  `macros.tex`; keep using the repository build for these multipart documents.
- Fig. 1A source was recovered from `hetankevin/diffPomp`, revision
  `ab31c911ae223aa91eb4e216d7d33710103050c3`, notebook `cholera_mop.ipynb`.
  Its embedded PNG matches the old manuscript PNG byte-for-byte. Extraction,
  rerun, and requested legend edit are in progress in `code/fig1a/` and will
  be recorded in the next entry. No experiments were refit for S11.

## October 6, 2026: backed-up result cleanup

- The J=1000 pipeline completed at 2026-10-06 03:27:20 UTC. All four
  arms have 100 fits/checkpoints and 100 successful final evaluations.
  There are 35,043 successful per-update evaluations; the completion audit
  and six-figure generation finished. Visual review of the new figures remains.
- Bulk outputs were archived outside the repository at
  `/home/kevin/storage-audit/dmop-cleanup-20261006T1855Z`.
  All 131,071 archived files were extracted and SHA-256 verified; original
  local result files/paths were preserved. The Git bundle preserves all refs.
- Removed bulk checkpoints, per-update evaluations/traces, global-search
  pickles, and seven compiled PDF/HTML documents from tracking only.
  Retained 399 compact result/configuration/figure files, including J=1000
  summaries. No blanket PDF/CSV ignore rules; manuscript figure PDFs remain.
- See `artifacts/README.md` and `artifacts/results-archive-20261006.json`
  for exact hashes, provenance, and restoration. Bulk-data audits/plots in a
  fresh clone need restoration; existing local scripts retain their paths.
- The cleaned tracked tree has 580 files. A clean index export builds both
  manuscript (23 pages) and SI (68 pages) with TinyTeX/latexmk, without
  undefined references/citations or inputs from the working result tree.
  Existing duplicate PDF destinations and one manuscript overfull box remain.
  All archived source files were rehashed after cleanup and were unchanged.
- This cleanup uses existing local storage and normal Git history. No new
  repository, external upload, history rewrite, or Overleaf relinking.
- The October 1 running-state entries below are historical. Remaining work:
  visually review J=1000 figures and interpret the standalone optimizer stops.

## October 1, 2026: higher-particle experiment running

- User requested higher-particle standalone and IF2-warm-started Corenflos
  and Ditlevsen. Launched all four variants with 1,000 fitting particles,
  100 starts each, under `corenflos/results/particle_increase_j1000_final_100`.
  This is an equal-wall-clock follow-up, not a matched-update experiment.
  Rates, guards, seeds, update caps, time offsets, budgets, and selection rules
  are cloned from each corresponding completed 100-particle configuration.
  Ditlevsen's auxiliary safety guard remains at 100 particles.
- Active user service: `dmop-particle-increase-j1000-20261001.service`.
  Main process 3607812; first Corenflos warm-start kernel compiled and the
  first fit is active on the RTX 3090. At the startup check the service was
  active/running, GPU utilization was 100%, and there had been no restarts.
  The other variants are queued, not concurrently running.
- Pipeline: `python -m corenflos.particle_experiment --particles 1000
  --starts 100 --partitions 10`. Ten serial partitions interleave ten starts
  per variant, so all four expose results before the whole experiment ends.
  One fitting/evaluation subprocess owns the GPU at a time. Each partition's
  fits are followed by independent final Euler-20 evaluation (5,000 particles,
  36 replicates); every-update evaluations (5,000 x 1) follow all fits.
  It then audits configuration, identical starts, trace coverage, and common
  evaluation effort; writes paired particle-count comparison CSVs; and
  regenerates comparison/mismatch figures with no prose annotations.
- `manifest.json` freezes all four planned configurations; `status.json`
  records the current variant, partition, and stage. Native checkpoints make
  the job resumable. An output lock prevents duplicate launches. The service
  uses GPU-only JAX, no GPU preallocation, a /tmp compilation cache, and a
  48-GiB host-memory limit. It retries failures after 120 seconds, at most
  three starts within an hour. Inspect failures before manually restarting.
- All four two-update GPU preflights passed, with finite logged likelihoods
  and gradients/scores and exact initial-parameter agreement with the old
  runs. Outputs are in `corenflos/results/particle_increase_j1000_preflight`.
  Timed runs: Corenflos warm 120.8 s, Ditlevsen warm 16.9 s, Corenflos
  standalone 32.0 s, Ditlevsen standalone 15.0 s. Warm Corenflos is about
  40 seconds per likelihood/gradient evaluation at J=1,000, so expect fewer
  updates within the original 850.657-second continuation budget.
- All 81 Corenflos/Ditlevsen tests pass, including new configuration,
  CLI-round-trip, serial-assignment, resume/completion, and audit tests.
  `git diff --check` is clean. Existing completed experiment outputs and
  figures are untouched. Do not edit fitting kernels while this job runs.
- Fitting should take roughly four days, plus evaluation/compilation time.
  The old standalone Ditlevsen baseline used four concurrent GPU workers;
  its old/new difference is also affected by scheduling. Report this caveat
  and update counts; do not claim that comparison isolates particle count.
- Next: inspect progress with `systemctl --user status
  dmop-particle-increase-j1000-20261001.service`, its journal, and per-variant
  checkpoint/evaluation counts. On completion, visually inspect all new
  figures (especially axis bounds and standalone medians), review paired
  results and objective mismatch, and commit/push the completed outputs.
  The long experiment has started; results are not yet complete.

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

The corrected 100-particle IF2-warm-start experiment and its audit are complete.
The four-variant 1,000-particle follow-up completed on October 6; see the
latest entry for the archived results and tracking cleanup.

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
