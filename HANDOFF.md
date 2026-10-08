# Handoff

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
