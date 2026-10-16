# Handoff

## October 16: live sampling revalidated after proposal explanation

- Host checks confirmed all four original PID/create-time/command identities live and advancing across a 50-second observation: NUTS330/500 warmup, primary J10000 PMMH14725total, J5000 PMMH18350total, doubled-prior PMMH7325total. Read-only observation session26041 completed0. No sampler restarted or changed; this turn is a verified wait after answering the proposal question.
- Next substantive reviews remain doubled-prior10500total, NUTS1000total (500retained), and reference PMMH30500total. The new runs already replace the empirical KDE with a proper Gaussian reference prior; independence of posterior conclusions from its width remains unverified. No manuscript changes or push; publication hold and active goal persist.

## October 16: historical-proposal control complete and reviewed

- Monitor61842 ended143 afterpoll34. Direct host recheck confirmed all four original PID/create-time/command identities still live; NUTS322warm,primary14525total,J500018150total,widerprior7125total. No live monitor remains. Monitor termination did not stop samplers; do not restart them. Recent turns are verified waits, unchanged settings/next review thresholds/publication hold.
- Read-only monitor61842 nowLIVE, same four verified sampler identities and40poll/50s window; stops at any completion/terminal sampler or NUTS1000total. Poll1:NUTS281warm,primary13500total,J500017025total,widerprior6075total. This supersedes the no-live-monitor note below. Settings unchanged; verified wait.
- Monitor71642 completed0 at its40-poll observation limit. All four sampler identities remained live and advancing atpoll39: NUTS279warm, primary13450total, J500016975total, widerprior6025total. No live monitor remains; the samplers were not stopped. Continuation turns after the250warm health check were verified waits. No configuration or manuscript changes; next review thresholds and publication hold unchanged.
- Halfway NUTS numerical-health check saved in gpu-forward-nuts-v2/warmup-health-250: frozen first250 trace/info/step arrays and input hashes. All states/force norms finite, zero divergences; updates201--250 trajectoryacceptance.591--.644, randomnessacceptance.66--.74, all four chains move every update. Forceddepthlimit98--100%; retain settings based on this health check and earlier depth screen, no convergence/efficiency claim. No new plot/report or retained NUTS draws. Monitor71642 remainsLIVE (poll16:NUTS250warm, primary12750total,J500016225total,prior5325total). Other runs unchanged; next full review thresholds unchanged.
- Read-only monitor71642 is now LIVE, same four PID/create-time/command checks and40poll/50s observation window; stops at any completion/terminal sampler or NUTS1000total. Poll1: NUTS232warm, primary12300total, J500015725total, widerprior4875total. Supersedes the no-live-monitor note below. No configuration changes; this is a verified sampling wait.
- Monitor78951 completed0 at its40-poll observation limit, with all four samplers still host-verified live: NUTS230warm, primaryPMMH12250total, J5000PMMH15675total, widerprior4825total. No live monitor remains; this did not stop sampling. Recent continuation turns were verified waits, with no configuration changes. User asked whether the proposal is plug-and-play: explained PMMH's derivative-free simulator/measurement-density kernel, while acknowledging that this experiment's preparation used gradient optimization/score metric. No request to change that preparation or restart the runs. Review thresholds remain unchanged.
- User asked what changed in the proposal. Verified the manifests and explained: PMMH uses regularized covariance from earlier pilot PMMH iterations501--1000, with scale adapted for500warm toward.234acceptance then frozen; this was additional tuning beyond a straight historical port. The control restores fixed diagonalSD.02. NUTS separately uses its revised covariance and initialstep.07. No new tuning requested or applied. Corrected stale gpu-comparison-v2 README links to current gpu-forward-nuts-v2 and pmrw-long, distinguishing current covariance choices. Monitor78951 remains LIVE atpoll29 (NUTS217warm, primary11925total, J500015350total, prior4500total).
- Follow-up is a verified wait after the completed control review (edbd598dd). Four remaining host PID/create-time/command identities revalidated live and advancing. New read-only monitor78951 LIVE, 40 polls/50s, stops at completion/terminal job or NUTS1000total. Poll1: NUTS183warm, primarylong11075total, J5000long14400total, widerprior3650total. This supersedes the no-live-monitor note below. No sampler settings changed.
- Control3609096 verified terminal, all four finite1500-iteration checkpoints and completion files hashed in completion-verification.json. Monitor9675 completed0; report2707 completed0. No live monitor remains. Existing four production jobs unchanged; last monitor: NUTS179warm, primarylong10975total, J5000long14300total, widerprior3550total.
- Three-row comparison generated under historical-proposal-control-v1/comparison: archived MH/current tuned PMMH/current target with old fixed proposal, common linear axes, traces1--1500 and densities501--1500. Comparison and all18 control traces/marginals visually inspected. Old proposal gives9.6--13.6% acceptance and clear chain separation/slow movement in trend and many other coordinates; it does NOT recreate original trend excursion to-1.22. Current tuned proposal explores trend more broadly. No causal attribution to prior/tempering from archived-vs-current comparison; old active MH sampling cell/clouds remain missing.
- review-1500 uses all1500 (zero adaptivewarmup), unlike comparison densitywindow501--1500: worstRhat2.706/minbulkESS4.73. These are not posterior estimates. New run README records evidence, limits and reproduction paths. This addresses finite-budget poor mixing, not complete NUTS-vs-PMMH reproduction. Next: NUTS500retained following500warm; both reference PMMH30000retained; doubledprior10000retained. No manuscript changes or push; publication hold persists, goal active.

## October 16: monitor resumed after primary continuation

- Read-only monitor9675 LIVE, now follows new primary3680785 and the other four unchanged samplers. Poll0 revalidated all PID/create-time/commands: NUTS155warm, primarylong10525total, J5000long13800total, widerprior3100total, historical-proposal control1075/1500. Stops at any terminal/completed job, NUTS1000total, or40polls/50s. Next likely substantive review is control completion; generate its three-row comparison and inspect all18 traces, preserving distinct targets when discussing archived MH. Prior review/extension was progress; this turn is a verified wait. No configuration or manuscript changes; publication hold persists.

## October 16: primary PMMH continuation verified live

- New primary PMMH continuation3680785/create1792126631.34 runs onGPU1 under1e2549a1f, `gpu-comparison-v2/pmrw-long`; launch JSON and log `/tmp/dmop-bayes-pmrw10000-long.log`. Host identity verified live during startup. All4 checkpoints advanced beyond10500 with BITWISE-identical parent trace/info/step/time prefixes and unchanged frozen covariance/step; finite saved states. pmrw-long-startup-verification.json records checks. Startup monitor21914 completed0; no live monitor remains at this entry.
- README navigation now points to both30,000-draw reference continuations and the separate1500-iteration historical-proposal control. Next immediate review is control3609096 completion1500 (new comparison command in historical-mh/README.md), then posterior references at30000retained, widerprior10000retained, NUTS500retained following500warm. All other running PIDs unchanged; previous primary3137405 terminal complete. No ms/SI edits or push; goal active, publication hold persists.

## October 16: primary J10000 PMMH complete and reviewed

- Monitor71308 completed0 at primary10500total. PID3137405 verified terminal; all4 complete finite checkpoints and hashes saved in pmrw-completion-verification.json. Report83190 completed0, `gpu-comparison-v2/review-pmrw-10000` freezes10000retained/chain. All18 traces/marginals plus trend panels visually inspected: chain3 has a late beta1/2/3 excursion (beta3 near-7), unequal seasonal tails and remaining immigration-tail differences. Layouts sound; not converged.
- WorstRhat1.05519(beta3), minimum bulkESS53.04/tailESS38.32(beta3); acceptance.2008/.2506/.2053/.1840. TrendRhat1.01095, bulkESS171.43, mean100beta=-.42786, descriptive95%interval[-.66677,-.20109]. Improved overall Rhat from8325drawsnapshot but still insufficient joint agreement; do not infer prior robustness or NUTS agreement.
- PREPARED exact20000-additional-draw continuation to30000retained at gpu-comparison-v2/pmrw-long; protocol saves parent/driver/review hashes, load_parent validates all frozen sources and states. Commit then launchGPU1 preserving deterministicflags and frozen tuning/RNG. Long runs provide posterior reference; user-requested1500-iteration drift comparison remains separate. Do not pool overlapping parent/continuation.
- Other jobs live at monitor end: NUTS3476536 at150warm, J5000long3505775 at13675total, widerprior3500694 at3000total, original-proposal3609096 at950total. No monitor remains live. No ms/SI edit or push; publication hold persists.

## October 16: five-job monitor renewed

- Prior turns were verified sampling waits. Monitor44323 finished0 at its40-poll observation limit; this did not stop any sampler. New read-only monitor71308 LIVE now includes historical-proposal-control3609096 along with the four original jobs. Poll0 verifies all five PID/create-time/commands: NUTS128warm, primaryPMMH10075total, J5000long13200total, widerprior2575total, original-proposal control550/1500.
- Monitor stops at any terminal/completed job, NUTS1000total, or40polls/50s. Next substantive review is primaryPMMH10500total completion or control1500total completion; commands and preparation above. No configuration, manuscript or publication change. Goal remains active.

## October 16: fixed-length control comparison ready

- Extended historical-mh/compare_traces.py with optional --control and --output. At control completion, render original/current/fixed-proposal rows on shared linear scales, all1500 iterations and densities501–1500. Validates common current/control target, starts, model, fixedSD.02 and completion; refuses existing output. Command in historical-mh/README.md. The resulting plot still needs visual inspection after real control completion.
- Verified premature control review is rejected without creating output; default two-row numerical arrays reproduce frozen comparison exactly (report7712 terminal0). No sampler/core source change, tests or regenerated manuscript figures. Existing runs unchanged; next thresholds unchanged. Monitor44323 remains the original-four-job monitor; control3609096 separate. Publication hold persists.

## October 16: historical-proposal control live

- Launched control3609096/create1792124393.61 onGPU1 underd5ae529f3. `historical-proposal-control-v1/launch.json` records command; log `/tmp/dmop-historical-proposal-control-v1.log`. Host PID/create-time/command verified live; startup monitor98636 completed0. All4 saved checkpoints finite, starts and target/prior/particle/model/Pypomp/flags match primary exactly, proposal remains zSD.02 and identity factor, no adaptation. startup-verification.json saved. Review at1500iterations; use iterations501–1500 for comparisons matching old plotting discard, despite no adaptation in this control. Do not conflate control with exact historical replay.
- Original trend chain1 start/end imply only-1.029 log-density change under the current *trend marginal* N(-.5,.5^2) prior. This excludes other-coordinate prior effects and does not by itself identify the cause of old drift. Pending functions cell569 now completed; monitor44323 covers original4jobs only, not new control. Existing production unchanged. Publication hold remains.

## October 16: original MH drift recovered; proposal control prepared

- User clarified that reproducing the original MH chain drift is part of the requested comparison. Do not substitute the HMC divergence diagnostic for this question or treat all-method convergence alone as the intended figure. Preserve long runs as posterior references; compare finite-budget behavior and explain changed settings. No tuning to force a visual outcome.
- Recovered four original1500x21 `mh_eb` arrays from diffPomp8d2fa90 into `code/bayesian/historical-mh`, with hashes and historical notebook/model/filtering source. Original SI image SHA matches diffPomp HEAD. Original trend chain1 ends-1.218689; same-length original/current trace comparison generated and visually inspected. Original drift is real. All available cholera.ipynb history inspected: MH survives as a commented alternative, active cells save NUTS. Intended target divides objective and KDE logprior by600; proposal SD=range/100. Actual KDE clouds, initial states and tracing-time seeds absent, so exact fresh replay is not established. Earlier narrow NUTS archive did not cover these existing MH arrays.
- PREPARED `historical-proposal-control-v1`: four chains, current primary starts/prior/untempered Pypomp target/J10000 unchanged, old fixed diagonal proposal (z SD.02), no adaptation,1500 iterations,seeds2026111600..03,GPU1. This isolates historical proposal versus tuned current proposal/adaptation, not the target/prior difference. Launcher validates source/input hashes and refuses overwrite. Commit then launch; verify finite checkpoints and identical current target/starts. Full posterior convergence is not expected in this short diagnostic and is not its success criterion.
- Existing four production jobs left unchanged. Monitor44323 live at last check; pending functions cell569 needs resuming. No ms/SI changes or push; publication hold remains.

## October 16: remaining interim marginals and traces inspected

- Inspected all18 marginal and trace panels in both J5000 and doubled-prior-SD `review-interim-20261016` reports. Layouts/linear scales are readable. J5000 retains chain3's late beta1/2/3 excursion and uneven omega4/5/6 tails. Wider-prior800-draw report shows substantial chain separation in gamma, beta1/2/3, omega4/6, including late chain1 movement; pooled trend resemblance does not demonstrate prior robustness. No new sampler change justified by these early plots; continue scheduled runs. These are frozen snapshots, not current chain lengths.
- Read-only monitor44323 LIVE; stops at any terminal/completed sampler, NUTS1000total, or40polls at50s. Poll0 revalidated PID/create-time/commands: NUTS79warm, primary9075total, J5000long12075total, widerprior1575total. Earlier35116 completed0. No manuscript change; publication hold persists.

## October 16: verified wait after interim reports

- Previous goal turn made progress: three frozen interim reports committed locally as3f284d4e5 and primary plots visually inspected. This turn is a verified wait. Monitor3767 ended143 afterpoll18; no live monitor remains. Rechecked all four host process identities after termination: all running and advancing, NUTS76warm, primaryPMMH9000total, J5000long12000total, widerprior1500total. The monitor ending did not terminate sampling; no restart or configuration change. Next reviews remain primaryPMMH10500total completion, NUTS1000total, J5000long30500total and widerprior10500total. Publication hold and active goal unchanged.

## October 16: user-requested interim posterior reports

- Three frozen `review-interim-20261016` reports completed successfully: primary J10000 PMMH 8325 retained/chain (max Rhat1.0821, min bulkESS46.6), J5000 exact continuation11300 (1.0813,47.1), doubled-prior-SD J10000 study800 (1.8948,5.8). No run has all-parameter convergence; prior sensitivity remains too early to interpret. No overlapping continuation samples pooled.
- Primary trend panels, all18 marginals and traces visually inspected: layout sound, trend marginals broadly overlap (Rhat1.0089), but seasonal and immigration coordinates still show uneven excursions and tails. Trend mean100*beta=-0.4225; descriptive95% interval[-0.6574,-0.1970]. This does not establish joint convergence. Other two fresh reports have numerical checks only at this entry.
- Tuned NUTS last observed69/500 warmup, zero retained; first50 warmup movement plot remains the available reviewed NUTS figure. Existing samplers/configurations continue unchanged. Read-only monitor session3767 was live at last check; functions cell545 is complete. Next planned thresholds unchanged. No manuscript edits or push; publication hold persists and goal remains active.

## October 15: tuned NUTS first50 warmup reviewed; retain configuration

- Monitor36337 terminal0 at NUTS50warm/primaryPMMH8350total/J5000long11300total/prior825total, all four host identities live. Review6152 terminal0. `gpu-forward-nuts-v2/warmup-review-50` freezes exactly50updates and initial states; all18 movement panels visually inspected. No divergences, trajectory acceptance.595--.643, all200 trajectories depth5/31steps; medianbatch40.41s. Chains still separated and lower tails incompletely explored.
- Descriptive RMS movement vs old50-update review is higher in beta2/beta3/omega5 by1.44/1.44/1.47; movement_comparison.csv/provenance include all18 and preserve identical unconvergedPMMH reference. Different seeds/states and warmup adaptation prevent a paired efficiency claim or confirmation of the threefold local screen result. Retain current settings; no restart.
- Next NUTS review after500retained (1000total), following500warm. Existing primaryJ10000PMMH3137405 next review at10000retained completion; J5000long3505775 at30000retained; prior3500694 at10000retained. NUTS3476536 remains live onGPU0. No monitor remains live at this entry. No manuscript changes; prior/particle/method agreement and full convergence still outstanding. Publication hold persists; goal active.

## October 15: current experiment navigation updated

- corrected/README.md now points to the four live batches, distinguishes the tuned NUTS covariance from PMMH, links the completed10,000-draw report and exact30,000-draw extension, and marks wider-prior sampling as running. No sampler/model/protocol changes.
- Read-only monitor36337 is LIVE, checking all four PID/create-time/commands every50s. Latestpoll3: NUTS27warm, primaryPMMH7800total, J5000long10700total, prior275warm; all live. Stops at NUTS50warm, any completion/terminal process or30polls. Its expiry is not sampler termination. For the50warm review, run gpu-forward-nuts-v1/review_warmup.py --run-root code/bayesian/corrected/results/gpu-forward-nuts-v2. The shared script now accepts this argument (default old run preserved); CLI/help and diff check passed. Full command in new run README. Numeric review/plot logic and frozen PMMH reference are unchanged. Publication remains on hold; goal active.

## October 15: PMMH extension and prior study startup verified

- LIVE J5000 extension3505775/create1792121252.05 onGPU1, `gpu-comparison-v1/pmrw-long`, launched under bf47b2120. Log `/tmp/dmop-bayes-pmrw5000-long.log`; pmrw-long-launch.json/protocol record command/hashes. Four chains advance beyond10500 with EXACT old trace/info/step/time prefixes and unchanged fixed tuning. State check retains cached values. New total30000retained; never pool this with overlapping parent files.
- LIVE wider-prior3500694/create1792121116.93 onGPU1 has finite saved checkpoints. Verified same primaryJ10000 likelihood/model/centers/starts/flags and exactly doubled priorSD. Startup verification saved. Both new processes revalidated by host PID/create-time/command; monitor76562 terminal0. No monitor remains live.
- Other live runs unchanged: tuned NUTS3476536/create1792120402.49(GPU0; next review50warm), primaryJ10000PMMH3137405/create1792109819.68(GPU1; next report10000retained completion). SharedGPU1 now has three PMMH batches; account actual time, no controlled speed claim. Next J5000 report at30000retained; prior report after10000retained and only interpret after convergence. All-parameter/prior/particle/method agreement still incomplete; no ms/SI edits. Publication hold persists; goal active.

## October 15: J5000 10,000-draw review; wider-prior study live

- J5000 continuation3143786 verified terminal/completed10500total; completion-verification.json saved. Monitor74395 and report30988 terminal0. `gpu-comparison-v1/review-pmrw-10000` freezes all10000retained per chain; all18 traces/marginals and trend panels visually inspected. Rhat1.013--1.087, minbulkESS42.5(omega6), mintailESS57.5(beta3); omega4 worstRhat. Late chain3 beta2/beta3 excursion and uneven omega5/6 lower tails remain. Not converged.
- LIVE wider-prior PMMH3500694/create1792121116.93 onGPU1, protocol/source checks passed before launch. `prior-sensitivity-gpu-v1/launch.json`, log `/tmp/dmop-bayes-prior-sensitivity-gpu-v1.log`. SameJ10000 target/centers, prior SD2x,500warm+10000draws,freshseeds2026110800..03. Host process confirmed live; first saved checkpoint still to validate at this entry.
- PREPARED `gpu-comparison-v1/pmrw-long-protocol.json`: exact continuation from completedpmrw-continued,20000additionaldraws=30000retainedtotal. Frozen source, four parent hashes and finite states verified by load_parent. Commit then launchGPU1; no process yet. Do not pool parent+continuation. PrimaryJ10000PMMH3137405 and tunedNUTS3476536 unchanged. New NUTS movement review at50warm, primaryPMMH review at10000retained. Publication on hold; goal active.

## October 15: tuned NUTS replacement live; prior warmup retired

- LIVE3476536/create1792120402.49 onGPU0, `gpu-forward-nuts-v2/nuts`, launched under eb38ce7f1. Log `/tmp/dmop-bayes-gpu-forward-nuts-v2.log`; launch/protocol/startup-verification saved. All4 first2 checkpoints finite, zero divergences; full source hashes, model/target/prior/particle/flags and original starts verified. Seeds2026111400..03, common tuned covariance and initialstep.07,500warm+2000draws. No posterior samples yet.
- Old NUTS3262256/create1792113782.16 deliberately SIGTERM'd only after replacement validation, verified terminal. Preserved all four148-warmup checkpoints and termination.json. Batch iteration cost6744.85s, counted ONCE, not four times. Monitor87463 terminal0; earlier monitor4583 also terminal0. No monitor remains live at this entry.
- PMMH3137405(J10000) and3143786(J5000 continuation) remain unchanged. Next: host-revalidate them and review at planned10500total completion; J5000 completion permits prepared prior-sensitivity-gpu-v1 launch once process terminal. New NUTS movement review after50warm, full convergence after retained sampling. Cost accounting includes all pilots/screens/retired warmups. No manuscript changes or converged comparison. Publication remains held; local commits only. Goal active.

## October 15: covariance screen complete; replacement NUTS prepared

- Monitor4583 terminal0 at screen completion; benchmark3428797 verified terminal. All19 records finite. Paired summary independently reproduced rate ratios. Covariance-only ratio.844[.622,1.278] misses screen; covariance+2x steps ratio3.013[1.696,6.128] passes all gates, acceptance.622/zero divergences. All18 coordinate point rates improve; bs3/omega5 ratios7.95/2.82. Individual intervals broad (epsilon/bs6/omega3 include loss); conditional six-trial evidence, not convergence or ESS/sec.
- PREPARED `gpu-forward-nuts-v2`: same original four starts/target/prior/J10000/G1000float32forward/depth5; selected common covariance, initialstep.07 (geometric mean tested doubled steps.07092), fresh seeds2026111400..03,500warm+2000draws. Existing adaptation unchanged. Check-only verifies all input/source/screen hashes. Commit then launchGPU0; not launched at this entry.
- Keep old NUTS3262256/create1792113782.16 until finite replacement checkpoint and source/target validation; then preserve retirement record and warmup. All pilot/retired warmup/screen/new warmup costs count. PMMH3137405 and3143786 unchanged/live (latest6725total and9688total); next reviews at10500total completion, then wider-prior launch after J5000 terminal. Publication hold remains, local commits only. Goal active.

## October 15: proposal covariance benchmark live

- Followup: first compiled baseline transition finite (76.12s); timed trials underway. Added summarize.py for both candidate/base comparisons, paired-trial bootstrap intervals and per-coordinate losses; it refuses incomplete results. No paired conclusions yet. All four process identities revalidated live (NUTS111warm, primaryPMMH5425retained, J5000continuation8338retained). Run the summary only after complete.
- LIVE3428797/create1792118901.92 onGPU1, `nuts-metric-check-v1/check.py`, protocol committed441e45abf. Log `/tmp/dmop-nuts-metric-check-v1.log`; launch.json records command/source/protocol hashes. Revalidated after25s; cached likelihoods/forces match BITWISE (8.64s). First shared-kernel compilation in progress. No timed results yet; do not duplicate.
- All three production processes verified live at launch and unchanged. Current posterior/prior/model and particle counts are untouched. This benchmark only changes proposal covariance/step size, with the old run still progressing.
- Next: monitor complete19 records (1 compilation+18 timed across3 candidates), then produce paired-trial uncertainty and coordinate comparison analogous to nuts-depth-check-v1/summarize.py (candidate names replace depths). No candidate automatically adopted; apply committed screen and inspect losses in beta3/omega5. Account for all pilot/setup costs if changing production. Keep publication on hold. Other run-review/prior-sensitivity tasks remain unchanged; goal active.

## October 15: fixed-metric mismatch checked; proposal screen prepared

- Offline comparison on both frozen5000-draw PMMH reviews suggests the fixed metric misses correlated directions. Preconditioned pooled covariance condition numbers: old34.8(J5000)/42.35(J10000). A candidate0.75*J10000 covariance+0.25*old, trace-normalized, gives9.8 on independentJ5000 and1.5 on its construction snapshot; per-chain checks in geometry.json. Both snapshots unconverged, so this is proposal-tuning evidence only, not convergence or a fitted prior.
- PREPARED `nuts-metric-check-v1`: reuses frozen update54 extended states/source from depth check, targets unchanged, depth5. Compare baseline, candidate covariance at1x and2x saved steps, six paired trials/four states/cyclic order, one shared executable. Screen requires>=50% movement/time improvement, acceptance>=.5, zero divergences, no lower point rate for beta3/omega5, plus paired-uncertainty/coordinate review. All preparation costs must be retained if adopted. Commit then launchGPU1; not launched at this entry. Production remains unchanged.
- Latest host checks: NUTS3262256 at103warm, primaryPMMH3137405 at5175retained, J5000continuation3143786 at8038retained; all live. User's updated AGENTS instructions incorporated; publication hold remains. Next review thresholds unchanged; no ms/SI edits.

## October 15: primary PMMH 5,025-draw review inspected

- `gpu-comparison-v2/review-pmrw-5000` freezes5,025 retained draws/chain. All18 traces/marginals and four trend panels visually inspected. Rhat1.019--1.179, minimum bulkESS18.6/tailESS20.0; no convergence. Seasonal transmission beta2/beta3 now show strongest disagreement, with further uneven immigration-tail excursions. Trend marginal overlap is insufficient (chain2 ACF~.46 atlag100). Acceptance.198/.238/.212/.178. Better than first1000 review but still incomplete; do not interpret differences from separately seeded J5000 as a particle effect.
- Monitor48168 and report30512 terminal0. Existing production continues unchanged. Next primaryPMMH review at10,000retained completion; J5000 also at10,000retained completion, then prepared wider-prior launch once J5000 process is terminal. NUTS remains in warmup; depth5 retained after the completed paired screen. No new sampler or manuscript change. Publication on hold, local commits only.

## October 15: depth comparison complete; retain depth5

- `nuts-depth-check-v1` complete,3358954 verified terminal. All14 transitions finite (2 compilation+12 timed); cached state bitwise check passed. Paired summary verified all seeds/counts and reproduced the main rate ratio. Depth5/6 mean batch costs67.47/134.92s, movement-rate ratio1.214 with paired bootstrap[.788,1.626]. Candidate misses prespecified25% improvement; wide uncertainty and omega5 ratio.960 give no reason to restart. Both zero divergences/all trajectories depth-capped; mean acceptance.731/.663. Retain current depth5 production.
- Monitor21298 terminal0 at benchmark completion. Production identities revalidated live: NUTS3262256 at93warm, J10000PMMH3137405 at4850retained, J5000continuation3143786 at7663retained. No tuning, target, prior or source change to existing jobs.
- Next immediate review: primaryJ10000 PMMH at5000retained using report.py; inspect all18 traces/marginals and trend panels. J5000 next review at10000retained planned completion; launch prepared prior sensitivity only when its process is terminal. Continue NUTS warmup (500planned), then diagnose retained samples. No converged comparison yet. Publication remains on hold; local commits only.

## October 15: paired depth benchmark running on GPU1

- Followup: both compiled transitions finished finite (depth5 76.35s, depth6 143.01s including compilation); timed paired trials now running. All four process identities revalidated live. New summarize.py refuses partial results, verifies seeds/counts, reports coordinate ratios and paired-trial bootstrap intervals. Run it after complete; do not infer a winner from the two compilation records. Screen protocol and production unchanged.
- LIVE depth benchmark3358954/create1792116803.27, `results/nuts-depth-check-v1/check.py`, launched under8213c7abd. Log `/tmp/dmop-nuts-depth-check-v1.log`; launch.json records command/source/protocol hashes. Host process verified running after30s. Cache check passed BITWISE for all four stored likelihoods and forces (8.83s). Compiling first depth5 transition at last check; no timing result yet. Do not duplicate.
- Production3262256/3137405/3143786 all revalidated live at benchmark launch and remain unchanged. The benchmark sharesGPU1 with PMMH; local same-device comparisons must not be presented as controlled production ESS/sec.
- Next: check benchmark completion/paired results and numerical diagnostics before deciding whether depth6 merits a new warmup configuration. Never silently alter the current run. Review primaryPMMH at5000retained and J5000 atcompletion; wider-prior launch still waits for J5000 terminal completion. Publication still on hold. Goal active.

## October 15: NUTS warmup reviewed; depth comparison prepared

- Monitor47054 terminal0 at NUTS50warm/J10000PMMH3600retained/J5000PMMH6338retained. Samplers were all verified live; monitor exit is just its threshold. Review session45464 terminal0; layout-fix63499 terminal0.
- `gpu-forward-nuts-v1/warmup-review-50`: all18 movement inspected. No divergences, trajectory acceptance.638--.687, all200 trajectories forced to depth5 (31steps). Weakest RMS jumps omega5=.20--.24 times provisional PMMH SD; omega6 range1.06--1.92 SD. No convergence claim. Initial figure title/legend overlapped; fixed and rerendered from the SAME frozen positions, then visually inspected.
- PREPARED `nuts-depth-check-v1`: frozen update54 states, same target/J10000/G1000forward force, compare depth5/6 with six paired independent four-chain transitions each, alternating order; compilation separate. Local geometric-mean coordinate squared-jump/time screen, not ESS. Candidate requires>=25% improvement, acceptance>=.5, zero divergences; inspect paired variability/coordinate losses before any restart. Source/input hashes verified, script parsed. Commit then launch onGPU1; current three production jobs unchanged. No benchmark process launched at this entry.
- Publication remains on hold per user. Prior sensitivity and full convergence remain pending. Next: launch/check depth benchmark, continue primaryPMMH to5000retained review and J5000 tocompletion; then wider-prior study. No ms/SI edits.

## October 15: user asked for status; publication remains on hold

- User answered the publication question with "hang on. how is it going?" This is NOT permission to push. Hold publication; local sampling continues. They also explicitly invoked the neurotypical skill, already read in this session. Keep status focused on outcomes and the bottleneck, not minute-by-minute counters.
- Explained candidly: PMMH improves but worst reviewed Rhat1.13 and immigration tails remain poorly explored; NUTS has no retained samples and is the runtime bottleneck, with a full run roughly1--2days at observed rates. Current proper reference prior replaces empirical KDE, but sensitivity remains untested.
- Monitor47054 still live, poll9: NUTS37warm, J10000PMMH3175retained, J5000PMMH5863retained. functions cell372 completed; resume the PTY monitor directly. Next movement review remains at50warm, not a convergence claim.

## October 15: warmup review prepared; monitor active

- `gpu-forward-nuts-v1/review_warmup.py` prepares the first50-update all18 movement plot and depth/acceptance summaries. It refuses to run before50 or overwrite an existing review. Grey bands/relative movement use the frozen, explicitly UNCONVERGED J5000 PMMH5063-draw snapshot; these are descriptive scales only, not a convergence test. Prepared but not yet executed or visually reviewed.
- Read-only monitor PTY session47054 is LIVE, checking the same three sampler PID/create-time/commands every50s. Last poll2: NUTS28warm, J10000PMMH2950retained, J5000PMMH5638retained. Monitor stops at NUTS50warm, primaryPMMH5000retained, J5000completion, or30polls. Resume with write_stdin; its expiry is not sampler termination.
- At NUTS50: run the review script with PYTHONPATH=/tmp/dmop-bayes-deps and MPLCONFIGDIR=/tmp/dmop-matplotlib using the pypomp Python, inspect movement.png and audit before deciding tuning. No changes to production or the prior-sensitivity source hashes. Publication approval remains pending.

## October 15: prior-tail interpretation checked against library model

- Read current Pypomp dacca._rproc: immigration is exp(dot(seas,omegas)). Checked all5,017 supplied covariate rows: seasonal basis minimum0, row sums0.9999999999999989--1.0000000000000013. Subtracting c from all six coefficients multiplies immigration by exp(-c), approaching zero environmental immigration. README records the conditional implication: a positive limiting likelihood would make an unrestricted flat prior on that common level improper. We have NOT evaluated the limiting likelihood or established posterior impropriety for a proposed flat prior.
- This supports keeping a proper reference prior while replacing the chain-specific empirical KDE. It does not prove why individual omega5/6 chains mix slowly or establish prior sensitivity. The planned doubled-SD comparison remains necessary.
- Same host processes revalidated live at20warm/2750retained/5388retained (NUTS/J10000PMMH/J5000PMMH). No sampling change. Publication approval still pending; do not push without it. Next convergence-review thresholds unchanged.

## October 15: publication approval pending; sampling continues

- Commit a5348b587 (longer PMMH review) is LOCAL ONLY. Automatic approval review rejected `git push origin main` twice: the destination is public and the posterior draws/diagnostic artifacts require specific publication approval. Read-only GitHub check confirms pypomp/dmop is public and authenticated viewer has ADMIN permission; that evidence did not satisfy review. Do not retry or bypass the rejection without new explicit approval.
- A user question is pending: publish a5348b587 including draws/figures to public pypomp/dmop, or keep local. No reply received as of this entry. This limits publication only; continue local experiments and commits.
- Same three GPU jobs revalidated live by host PID/create time/command after the rejection: NUTS3262256 at19warm, J10000PMMH3137405 at2675retained, J5000continuation3143786 at5313retained. No restart. The next useful reviews remain NUTS50warm, J10000PMMH5000retained, and J5000completion. Goal active.

## October 15: J5000 PMMH review after 5,063 retained draws

- Saved `gpu-comparison-v1/review-pmrw-5000`: four chains, 5,063 retained draws each. All18 traces/marginals and four trend panels visually inspected. Rhat1.012--1.133, minimum bulkESS25.3/tailESS30.8; improved agreement but no convergence. Seasonal omega5/omega6 lower tails remain unevenly explored. Trend overlap alone is insufficient.
- Fixed report.py warmup legend background so its marker cannot cross the text; regenerated and inspected the report. No numerical calculation or model change. Prepared prior-sensitivity protocol's report hash refreshed transparently; all27 source hashes verified.
- Revalidated host PID/create-time/command: forwardNUTS3262256 live at18warm, J10000PMMH3137405 at2625retained, J5000continuation3143786 at5263retained. No sampler restart or tuning change. Latest neutral user question about old settings was answered from archived code; continuing the active goal.
- Next: NUTS movement/depth review at50warm; primaryPMMH review at5000retained; J5000 next review at10000retained completion, then launch prepared wider-prior study once its prerequisite process is terminal. Continue toward converged NUTS/PMMH, prior/particle agreement and final figures; no ms/SI edits this turn. Goal active.

## October 15: validated forward NUTS is now the live primary run

- LIVE new NUTS3262256/create1792113782.16/GPU0 at results/gpu-forward-nuts-v1/nuts. Launched under e1f25c604 with same target/prior/starts/covariance, J10000/G1000float32forward, both deterministic flags,500warm+2000draws/depth5/freshseeds2026111000--03. Log /tmp/dmop-bayes-gpu-forward-nuts-v1.log. Revalidate host identity; do not duplicate.
- All four first checkpoints finite/zero divergences; startup-verification.json checks target/model hash, priors, starts, covariance and frozen source. First update88.57s includingcompile with oldNUTS still sharingGPU0. Full newproduction throughput pending; no controlledspeedupclaim.
- OLD NUTS3137251 verified and terminated at29warm updates. results/gpu-comparison-v2/nuts/termination.json records identity, terminal verification and checkpoint hashes. Do not restart it. Both PMMH jobs remain unchanged:3137405 J10000 in gpu-comparison-v2/pmrw and3143786 J5000 in gpu-comparison-v1/pmrw-continued.
- Full forward kernel benchmark3244703 and replay3251125 verified terminal/complete. CPU/GPU exact partial-save recovery passed,16tests passed. Model.py unchanged. Prior-sensitivity source hashes refreshed for optional runner changes, settings/seeds unchanged.
- Next: monitor forwardNUTS; review its movement/forced-depth fraction at50warm updates. Review J10000 PMMH at5000retained (last seen1900). Launch prepared wider-prior PMMH after J5000 completion. Convergence/prior/particle checks, controlled cost and final manuscript figures remain pending. Goal active; no ms/SI edits.

## October 15: forward mode validated for full NUTS and recovery

- force_modes.py implements a scalar-projection jacfwd wrapper. run.py/batch.py/device_check.py accept --force-mode forward (defaultreverse); continuation drivers restore the recorded mode from frozen source and reject mixed batch modes. model.py remains byte-identical (c4804a1b44da...). Generic float32-objective test caught a gradient dtype mismatch; explicit cast to parameter dtype fixes it. All16 relevant tests pass.
- Real Dhaka uneven-checkpoint replay passes BITWISE on CPU and GPU, all arrays excepttimes. CPU session59698 terminal0; GPU replay3251125/create1792113437.93 verified terminal and passed. Verification JSONs in forward-force-check-v1; full fixtures /tmp/dmop-forward-recovery-{cpu,gpu}-v1.
- Full four-chain forward NUTS J10000/G1000/depth5 benchmark complete: first110.56s includingcompile, then100.55/100.81s,31steps, allfinite/zero divergences. Benchmark3244703/create1792113249.2 was running at last hostcheck but result now complete; verify terminal before reporting retirement. SharedGPU1 vsoldproductionGPU0 prevents a controlledspeedupclaim. Separate likelihoodreplicatesdisabled; NaN emptymeans in initial progresslog were loggingonly, nowfixed.
- PREPARED new gpu-forward-nuts-v1 protocol/launcher: same target/prior/starts/covariance, G1000float32forward,J10000,500warm+2000draws, depth5, step.03, freshseeds2026111000--03, GPU0, checkpoint1. Commit then launch. RetireoldNUTS3137251 only after finite newcheckpoint and matchingtarget/source checks; keepbothPMMH jobs unchanged. No newproduction launch yet at this entry.
- Prior-sensitivity protocol sourcehashes refreshed transparently for optional runner changes; its seeds/settings/model remain unchanged and defaultforce reverse. Next: launchvalidatednewNUTS, verifythenretireoldwarmup withPIDidentityrecord; preserve allcosts. ContinuePMMH/priorparticlechecks and finalmanuscriptwork. Goalactive.

## October 15: forward force benchmark gives a promising cost reduction

- New results/forward-force-check-v1, protocol f92e8c0db, evaluates differentiation only. CPU G20/batch2 and GPU G1000/batch4 checks passed. Scalar objectives identical, scaled maximum force difference1.56e-6 on GPU (threshold1e-4), repeated force/value evaluations bitwise identical under both optimized deterministic flags.
- Direct isolated model-copy forward AD median1.070s vsreverse4.688s per force batch. More useful WRAPPED candidate (commit4939f6d99) applies jacfwd to the scalar projection of the existing value_force function: median1.008s vsreverse4.595s. It needs NO model.py change. Full gradients differ slightly due tofloat32 rounding; do not claim bitwise equality across AD modes.
- GPU benchmark3228541/create1792112770.67 and wrapped3234591/create1792112947.65 both verified terminal/absent with complete results. Production3137251/3137405/3143786 revalidated live and unchanged. No production integration or restart yet.
- NEXT: extract/test a forward_value_force wrapper, benchmark complete NUTS updates and verify exact interrupted-batch replay using the wrapper. Then consider an explicit new NUTS configuration. Preserve model.py byte-for-byte so target hashes and prior/particle comparisons remain valid. Optional runner/continuation configuration must record force mode, with old snapshots defaulting toreverse. The prepared wider-prior protocol hashes runner source; refresh its source hashes transparently if integration changes runner files before its launch.
- Current force check costs are under shared GPU load, not controlled sampler speedup/ESS evidence. Keep primary PMMH sampling and prior/particle checks. Main/SI unchanged; goal active.

## October 15: first J10000 PMMH report inspected

- Monitored the same three verified host processes until primary PMMH reached1500 total. Monitor session25235 terminal0; sampler processes remain live. Last monitor: NUTS18 warmup, J10000 PMMH1000 retained, J5000 continuation3513 retained per chain.
- gpu-comparison-v2/review-pmrw-first1000 freezes exactly1000 retained draws/chain. All18 trace/marginal panels and four trend panels visually inspected. Rhat1.094--1.722, min bulkESS6.58; seasonal beta1--3 and omega4 especially poor, omega6 lower tail explored unevenly. Trend Rhat1.179/ESS16.3. Acceptance.162/.285/.188/.176. No convergence.
- Preserve and continue the planned frozen run; next PMMH review around5000 retained or earlier numerical failure. Do not compare this1000-draw snapshot causally with longerJ5000 snapshots. More particles have not removed slow exploration, but this is not evidence they worsen it.
- NUTS review remains scheduled at50 warmup updates for depth-cap/movement assessment. Wider-prior study remains prepared/unlaunched untilJ5000 finishes. No manuscript or sampler changes. Goal active.

## October 15: NUTS early adaptation inspected

- Live NUTS3137251 verified by host PID/create time/command. At11 saved warmup updates all chains had zero divergences but every trajectory hit depth5 with31 integration steps and no U-turn. Step adaptation is active (next steps roughly.029--.040); recent batch cost~135s. early-warmup-review.json freezes checkpoint hashes and summaries.
- This is not convergence evidence. Reaching the cap may restrict movement; inspect again at50 warmup updates (all18 movement, trajectory acceptance and forced-depth fraction) before deciding whether any explicit depth change is justified. Preserve current run; no restart or tuning change this turn.
- Next main PMMH diagnostic review remains at~1000 retained draws/chain. Previous verified monitor had450 retained; J5000 continuation2913 retained. Wider-prior study is prepared and must wait for J5000 completion. Manuscript unchanged; goal active.

## October 15: offline sensitivity comparison added

- New sensitivity.py compares frozen report.py outputs for --kind prior or particles. Requires same method/model and permits only the intended target difference, verifies proportional prior-SD scaling, retains separate-chain marginal curves and labels unconverged comparisons. Table includes18-coordinate intervals, SD ratios, mean shifts/combined MCSE. It makes no automatic equivalence/insensitivity claim.
- Three tests passed for valid single-factor changes and rejection of confounded, identical and incorrectly scaled comparisons. No sensitivity results yet; wider-prior launcher remains prepared/unlaunched pending J5000 completion.
- All three existing GPU processes revalidated live this turn; no production source, target, tuning or manuscript change. Run this comparison only on frozen reports, interpret after convergence and visual review.

## October 15: wider-prior PMMH comparison prepared

- results/prior-sensitivity-gpu-v1 contains a frozen protocol and guarded launcher: J10000, four fresh seeds2026110800--03, prior_scale2, same fixed coordinates/centers/starts/covariance as primary,500 warmup+10000 draws. This doubles prior SDs; likelihood remains unchanged. No new fitted prior.
- Host --check-only passed input/source hashes and confirmed prerequisite3143786 is still live/incomplete. NO sensitivity run launched and no background scheduler installed. Launch manually after J5000 pmrw-continued completes and its host process exits. It will useGPU1; primary NUTS/PMMH untouched.
- All three existing process identities were revalidated this turn. Prior-sensitivity assessment requires convergence in both settings and all18 marginal/interval comparisons, especially omega5/omega6; do not claim prior independence based on current concentration.
- Next: monitor the existing jobs, launch sensitivity when its prerequisite finishes, assess primary convergence and particle agreement. Goal active; manuscript unchanged.

## October 15: longer PMMH traces reviewed

- Revalidated all three current host processes by PID, create time and command; they remain live. NUTS warmup unchanged.
- New gpu-comparison-v1/review-pmrw-continued-20261015 freezes2338 retained draws per chain through2838. All18 traces/marginals and four trend panels visually inspected. Rhat1.033--1.235, min bulkESS12.7; no convergence. Trend is better mixed than the worst seasonal immigration coefficients.
- omega5/omega6 show long lower tails and slow exploration; scrutinize them in the planned prior-scale comparison. Concentration relative to the current prior alone is not a sensitivity test. Keep current frozen runs going; do not restart from this interim review.
- Report outputs/interpretation saved; no manuscript or production code change. Next review J10000 after retained sampling accumulates; convergence, prior/particle checks and final figures still pending.

## October 15: production identities revalidated and records saved

- Host PID/create-time/command checks confirm all three current batches live: NUTS3137251, J10000 PMMH3137405, J5000 PMMH continuation3143786. At check they had 4,375,2788 total updates respectively. No restart or tuning change.
- Noise-prefix check confirms the expanded128 reused the original32 seeds. J5000 re-evaluation differs by at most9.1e-13 log units; J10000 agrees bitwise. prefix-verification.json records this. Do not pool the two sets.
- Saving launch, startup, retirement, exact-continuation and completed diagnostic records. Still no converged NUTS/PMMH comparison. Next: let current batches accumulate; review retained draws, then prior sensitivity, particle agreement and final figures. Main/SI unchanged.

## October 15: deterministic J10000 production running

- Primary results/gpu-comparison-v2 launched under protocol c328bb670. LIVE NUTS3137251/create1792109818.16/GPU0 and PMMH3137405/create1792109819.68/GPU1. Both J10000, prior scale1, same starts/fixed covariance,500 warmup; NUTS G1000 float32 force/depth5/2000 draws, PMMH10000 draws. Both use --xla_gpu_deterministic_ops=true --xla_gpu_enable_scatter_determinism_expander=true. Logs /tmp/dmop-bayes-gpu-comparison-v2-{nuts,pmrw}.log.
- Both primary batches saved finite checkpoints; first NUTS update159.6s including compilation, zero divergences. startup-verification.json records source/input checks. New PMMH passed125 warmup updates at check. Current throughput implies a multi-day NUTS run; user informed. No convergence claim.
- Old GPU NUTS3052229 deliberately stopped/verified terminal at22 warmup iterations, preserved under gpu-comparison-v1/nuts/termination.json.
- J5000 PMMH retained for particle sensitivity. Old3052372 deliberately stopped at2513; exact batch continuation LIVE3143786/create1792110008.97/GPU1 under gpu-comparison-v1/pmrw-continued. Checkpoint interval25 instead of1; original10500 total and frozen tuning unchanged. All four restored likelihood/force differences0. First resumed prefixes match parent exactly; pmrw-continuation-verification.json records it. Use continuation only, never parent+continuation as separate chains. Log /tmp/dmop-bayes-pmrw5000-continued.log.
- Optimized deterministic benchmark3115959 and followup3121299 are terminal/complete. Median75.18s per four-chain15-step batch atJ5000/G1000; exact GPU partial-save replay passed ALL numerical arrays under optimized flags. Artifacts copied into repo.
- Expanded plateau diagnostic128replicates: J5000 variance1.0105[.751,1.298], J10000 .3027[.240,.363], ratio3.338[2.391,4.694]; cached5000 likelihood1.831 log units above estimated mean and fixed-theta refresh acceptance.143. Initial32 are included in128, never pool. This motivated J10000; it is local evidence, not a global adequacy proof.
- Next: monitor three LIVE batches, review retained samples after warmup; avoid further restarts without evidence. Assess all18 convergence, prior-scale sensitivity and particle-count agreement, then controlled cost and final figure/manuscript work. Main/SI unchanged; goal active.

## October 15: optimized deterministic timing and expanded noise check

- Ordinary deterministic benchmark3098070 deliberately terminated after490s without a recorded NUTS update; termination.json verifies PID/create time and partial outputs. Do not resume it.
- LIVE optimized benchmark3115959/create1792109171.44 onGPU1, flags --xla_gpu_deterministic_ops=true --xla_gpu_enable_scatter_determinism_expander=true. First initialization14.96s, compiled first15-step batch90.25s, next batches76.08/75.18s so far, finite/zero divergences. Check completion and third timing. Launch/log under gpu-precision-check-v1/optimized-deterministic-batch4-launch.json.
- Initial posterior-noise job3112729 is terminal and complete. Plateau105 iterations at chain0 iteration827: J5000 loglik variance1.728, J10000 .302 (32 replicates); recorded likelihood1.441 above estimated logmean, fixed-theta refresh acceptance .168. This suggests noise contributes, not that it is the sole cause.
- LIVE sequential followup parent3121299/create1792109329.80 onGPU0: first full optimized deterministic partial-save replay (/tmp/dmop-batch-continuation-gpu-optimized-v1), then128 replicates at the plateau for both J5000/J10000. Parent/source/tasks recorded under posterior-noise-check-v1/followup*.json and followup.py. Reference8-step run finished; resumed stage pending/in progress. Use parent process+children and task logs; do not duplicate.
- Expanded128 includes original32 seeds; do not pool. Once complete run posterior-noise-check-v1/summarize.py for bootstrap diagnostic intervals. No production particle-count change yet.
- All source/tests/protocols committed; noise summary tests2passed. Both production3052229(NUTS warmup) and3052372(PMMH sampling) remain unchanged. Goal active.

## October 15: posterior-state likelihood noise check prepared

- Current GPU production and deterministic benchmark remain live; benchmark3098070 had no timed update yet after~200s, verified process (not a failure). Keep monitoring it.
- New noise_check.py evaluates fresh PF replicates in batches at saved states, preserving zero likelihood estimates and reporting fixed-parameter refresh acceptance only at the recorded particle count. No parameter proposals or sampling changes.
- Frozen protocol/results/posterior-noise-check-v1 uses chain0 longest postwarmup plateau through1400, then chain0/1/3 states at1400. J5000 vs10000,32 independent replicates per point, batches16, GPU1. Inputs and source checkpoint hashes saved. Launch after committing protocol; not launched in this entry.
- This diagnostic addresses lower PMMH chain0 acceptance. It does not certify posterior convergence or justify changing the target/prior. NUTS remains in warmup and PMMH continues retained sampling. Goal active.

## October 15: deterministic production-size benchmark live

- LIVE detached benchmark PID3098070/create1792108682.20 onGPU1. Command/log in gpu-precision-check-v1/deterministic-mixed-batch4-launch.json; log /tmp/dmop-bayes-deterministic-mixed-batch4.log. Protocol a26dff2e9. Revalidate host process; do not duplicate.
- Session37490 deterministic repeated-force test is terminal0 and passed. Session92016 GPU replay terminal0 but comparison failed bitwise as recorded; session6814 repeated-default-force terminal0 confirmed nondeterminism. All tests after flag-recording changes:9 passed (batch continuation plus scalar recovery).
- Both production batches remain unchanged/live. Finish actual-size timing, then run check.py with deterministic flags and a new output /tmp/dmop-batch-continuation-gpu-deterministic-v1. That complete deterministic replay is still TODO.
- Production files record initial flags empty. Any switch to deterministic execution must be an explicit new warmup/configuration, not silent continuation. Keep current run until a validated concrete replacement. PMMH evaluates no force and had bitwise-stable likelihoods.
- Convergence, local PF-noise check for sticky PMMH chain0, prior/particle sensitivity and manuscript results remain outstanding. Goal active.

## October 15: deterministic GPU force option verified at small size

- Session37490 completed. With XLA_FLAGS=--xla_gpu_deterministic_ops=true, all five repeated four-chain J100/G20 force and likelihood evaluations match bitwise. Postcompile force~4.9s. Result copied to batch-continuation-check-v1/force-repeatability-deterministic.json.
- run.py, batch.py and device_check.py now record XLA_FLAGS; both continuation drivers reject different flags. Existing production snapshots unchanged and their flags were verified empty. Do not use deterministic flags to silently continue those existing chains.
- Committed next protocol gpu-precision-check-v1/deterministic-mixed-protocol.json: actual J5000/G1000, four lanes/depth4, three timed postcompile updates onGPU1. Launch after commit; current production unchanged.
- Next after timing: run the complete GPU partial-save replay with deterministic flags from the start (fresh reference; old reference used nondeterministic flags), then decide a documented production transition. Exact deterministic-force repetition alone is not full replay verification.
- Previous batch replay under empty flags was numerically close, not exact; preserve its failure and comparison records. No production restart, posterior convergence or manuscript change.

## October 15: GPU replay exposes force nondeterminism

- GPU partial-save replay completed after widening only the batch force-cache allowance from16 to64 eps32 times maximum force magnitude. Float64 likelihood tolerance remains1e-9; cached values are never replaced.
- GPU replay is NOT bitwise exact: keys, particle keys, acceptance/divergence/trajectory decisions and covariance match exactly; z differences up to1.03e-11, loglik differences up to2.97e-10, force differences up to.00195. Details in gpu-replay-differences.json. Do not report an exact GPU pass. CPU replay remains exact.
- Repeat identical GPU force calls five times: force ranges .00208/.00242/.00269/.00354 across chains, while likelihoods match bitwise. This occurs without continuation. Installed libjax_common.so exposes xla_gpu_deterministic_ops.
- LIVE verification session37490 tests same force with XLA_FLAGS=--xla_gpu_deterministic_ops=true onGPU1; output /tmp/dmop-batch-continuation-gpu-v1/force-repeatability-deterministic.json. Three evaluations seen (~5s postcompile); resume handle, do not duplicate. Earlier GPU recovery session92016 completed. Initial test harness3074902 is terminal.
- Production3052229/3052372 still running unchanged, both verified XLA_FLAGS empty. Next: inspect deterministic check; if it works, benchmark actual J5000/G1000 batch before deciding how to use it. Record execution flags in future configurations/continuations. No existing production state changed.
- First GPU PMMH report review-pmrw-early freezes371 retained draws/chain. All18 traces and trend panels visually inspected; slow mixing persists (Rhat1.14--1.63). Chain0 retained acceptance~.09 initially/~.115 over later200; others~.23--.33. A local PF-noise check at its sticky state may distinguish likelihood noise from proposal scale. No such check launched yet.
- Goal active; NUTS still warmup (~9), PMMH~1100total. No manuscript change or convergence claim.

## October 15: batch continuation implemented and CPU replay verified

- New batch_extend.py imports frozen parent model/kernel/batch code, keeps lane order and cached state, and continues each lane at its own iteration. Supports partial saves with unequal counts; completed lanes remain masked in the original batch shape. Prior/target unchanged.
- Seven new tests passed: partial checkpoint loading, rejection of changed source/lane/state/terminal record, and unequal-lane exact replay for both NUTS and PMMH. A real two-chain CPU completed-run extension reproduced every numerical checkpoint array exactly (six total iterations; only times differ).
- Batch force validation retains original cached values; allows float32 fusion rounding at16 ulps of the maximum force entry. Float64 likelihood tolerance remains1e-9. Scalar extender still rejects batch checkpoints. Exactness established by replay, not this tolerance.
- Next committed check: results/batch-continuation-check-v1/check.py on GPU1, four real Dhaka chains with partial checkpoints at3/2/3/2 during six-step warmup. Output /tmp/dmop-batch-continuation-gpu-v1; copy verification after completion. No GPU recovery test launched yet in this entry.
- Production GPU processes3052229(NUTS) and3052372(PMMH) verified live; around4 and499 warmup iterations respectively. No restarts or manuscript changes. Goal active.

## October 15: primary comparison now running on GPUs

- Committed protocol db6e2194b launched both batches via results/gpu-comparison-v1/launch.py. LIVE NUTS PID3052229/create1792107342.38/GPU0; LIVE PMMH PID3052372/create1792107343.84/GPU1. Read their launch JSON and revalidate host identity before any action. Logs /tmp/dmop-bayes-gpu-comparison-v1-{nuts,pmrw}.log.
- Both batches saved finite float64 target checkpoints; NUTS initial update zero divergences,31 integration steps/depth5; PMMH around0.8s per four-chain update. startup-verification.json records exact checked checkpoints. Still warmup; no convergence claim.
- Verified and deliberately terminated old CPU NUTS PIDs2917966/68/70/72 at75 iterations and PMMH PIDs2932368/69/70/71 at2925/2900/2900/2175. Each termination.json records command, create time, checkpoint hash and verified terminal. Do not restart or count as independent production evidence.
- Current configuration: J5000 float64 target, NUTS G1000 float32 force/alpha.95/depth5; common proper prior scale1; identical four starts and fixed pooled-old-warmup covariance;500 step-adaptation iterations then2000 NUTS /10000 PMMH draws. All old preparation costs retained. Original ms/SI/figures unchanged.
- IMPORTANT continuation limitation: scalar re-evaluation of a mixed-force batched checkpoint differs by ~6e-5 in four force entries due to rounding (likelihood check passes). extend.py now explicitly rejects batch_size>1 rather than claiming exact scalar recovery. Implement a continuation with the original batch shape and frozen source before extending/recovering these new runs. Do not relax old scalar validation tolerances. No such batch continuation exists yet.
- Next: monitor GPU adaptation/mixing; implement/test batch continuation; run report.py after enough retained draws. Convergence, prior/particle sensitivity, controlled cost comparison and final manuscript figures remain outstanding. Goal active.

## October 15: GPU production comparison ready

- Mixed-force GPU benchmark completed: median43.64s per four-chain 15-step batch versus85.94s full64. Float64 likelihoods at all four inputs match exactly. Finite, zero divergences; no posterior samples.
- batch.py validated: scalar-vs-batched NUTS/PMMH RNG/state/diagnostics agreement, fixed-metric adaptation, two model precision tests and two recovery tests (7 total). A real Dhaka two-chain smoke completed and saved valid checkpoints; inputs/results in /tmp/dmop-bayes-batch-smoke-v1, verification JSON committed.
- PMMH report review-pmrw-20261015-b freezes950 retained per chain; all18 traces visually inspected. Still slow mixing: Rhat1.10--1.58, minimum bulkESS6.90. No convergence.
- New GPU production protocol at results/gpu-comparison-v1: shared starts at old PMMH iteration1950, fixed pooled-warmup covariance (not a prior fit),500 step-adaptation iterations; NUTS depth5/2000 draws on GPU0 and PMMH10000 draws on GPU1. Prior/target unchanged; NUTS force32 only. Initialization cost explicitly retained.
- Launch after this commit with launch.py nuts / launch.py pmrw, then verify finite checkpoints before retiring verified old CPU processes. Old results remain development archives. No production GPU launch yet at this entry.
- Goal active: convergence, prior/particle checks, controlled cost comparison, final plots and minimal manuscript edits remain.

## October 15: mixed proposal precision and GPU batch runner

- Full64 GPU batch4 finished: postcompile median85.94s for four 15-step updates, finite and zero divergences. PID3025464 verified terminal.
- Added optional float32 proposal force while retaining float64 PF target/prior. Two model tests passed, including target identity, deterministic forces and batched force execution. Production configurations unchanged.
- Added batch.py for independent NUTS/PMMH chains with separate RNGs, adaptation and ordinary per-chain checkpoints. Scalar-vs-batched known-target tests pending; no production batch run yet.
- Next GPU check is mixed-force-protocol.json, same fixed inputs/J5000/G1000/depth4, three postcompile batches. Launch after committing this protocol.
- All four CPU NUTS and four recovered PMMH processes verified live on host; keep until a validated replacement. No convergence or manuscript change.

## October 15: GPU batching evidence and completed depth4 pilot

- Main precision checks complete: float32/float64 each128 PF evaluations at
  four saved warmup points. precision-comparison.json and its reproducible
  summarize_precision.py report log-average-likelihood differences .050/.062/
  .332/.108. Point2 bootstrap95% interval [.071,.582] excludes0; others include0.
  Do NOT claim all differences are Monte Carlo noise or global equivalence.
- Float32 GPU batch2 and batch4 complete/terminal: median33.53s per pair and
  39.11s per four-chain update, each15 integration steps, no divergences in
  these short fixed-point checks. Scalar float32 median34.50s. Shared GPU
  load and different points preclude an ESS/sec speedup claim.
- LIVE float64-batch4 PID3025464 on GPU0, launched under protocol bf150d29f.
  Its launch JSON/log are under gpu-precision-check-v1; revalidate PID/create
  time. It checks whether batching is worthwhile without changing precision.
  All other GPU benchmark jobs are complete; no production GPU chain yet.
- Longer NUTS pilot completed120 total/60 retained. Report generated and all18
  traces visually inspected: substantial drift, trend ESS4.49, minimum bulk
  ESS1.59, zero divergences, all15-step trajectories forced to stop at depth4.
  This is tuning evidence, not convergence. Monitor43525 is terminal0.
- Verified pilot PID2932372 terminal, then moved PMMH chain3 PID2932371 from
  core28 to released core10; affinity_change.json records it. Thus PMMH0..3
  now use cores8/12/14/10. Main NUTS0..3 still use cores0/2/4/6, untouched.
- Next: finish full-precision batch timing and select the GPU execution plan
  using measured throughput and numerical evidence. GPU batching will need
  a checkpointed multi-chain runner with scalar-versus-batched kernel checks.
  Reuse existing kernels; preserve independent seeds, complete traces, warmup
  adaptation, prior and untempered target. No such production runner exists yet.
  Consider trajectory depth and covariance adaptation using the completed
  pilot; current main chains have only~55 warmup iterations. Do not mistake
  lack of divergences for mixing. Current CPU runs remain live and must not
  be duplicated or discarded without a concrete replacement protocol.
- All main/SI text and original figure assets remain unchanged. Convergence,
  prior/particle checks, final same-hardware comparison and manuscript edits
  still outstanding; goal active.

## October 15: float32 GPU and precision checks underway

- Previous goal turn completed actual float64 GPU timing and verified original
  float32 settings. Current new device_check.py supports precision, particle
  count and batch size explicitly. model.py still defaults64 but honors an
  explicit JAX_ENABLE_X64=false before importing data. Existing sampler source
  snapshots unchanged. test_model.py passes (1 test,13.56s).
- Protocol62fb55242: results/gpu-precision-check-v1/inputs.npz freezes four CPU
  warmup positions/metrics. Float32 GPU0 PID2992754 and float64 GPU1 PID2992755
  are BOTH complete/terminal. Each has16 PF replicates per point. All finite;
  precision differences are within broad Monte Carlo uncertainty, not enough
  to claim equivalence. Float32 single-chain15-step NUTS postcompile timings
  32.85/34.50/35.02s, zero divergences, versus prior shared-GPU64 median60.46s.
- Followup protocol397aa89c9 committed before launch:128 replicates/point in
  each precision (use128 alone, because first16 seeds repeat), plus batch2
  float32 NUTS timing. LIVE detached float64-reps128 PID3001017 on GPU1;
  LIVE float32-batch2 PID3003176 on GPU0 (read its launch JSON).
  Once batch2 finishes, launch float32-reps128 on GPU0 with --reps128 and
  --kernel-reps0, same inputs and precision32. This launch remains TODO.
- GPU benchmarks coexist with unrelated work; no outside processes touched.
  Production remains four CPU NUTS chains plus recovered PMMH and pilot.
  Monitor43525 continues; revalidate handles rather than restarting anything.
  Goal active: convergence, prior/particle checks, final figures and edits remain.

## October 15: user asked about GPU and original configuration

- User objected to CPU use, then asked for the original experiment settings.
  Answer verified against archived sampling_cells.py and arrays: original
  TFP NUTS J1000/alpha.95, fixed step.05, depth5, four sequential chains,
  1500 draws with first500 discarded, IF2 M40/J10000 KDE per chain; likelihood
  AND log prior divided600. All four saved arrays are float32. Device was not
  recorded in the archived sampling source. Current runs are float64 and
  J5000/G1000, an additional change requiring attention in hardware comparison.
- Completed actual current-config GPU benchmark, session68980 terminal0,
  results/gpu-current-config-check-v1. GPU0 RTX3090 shared with unrelated
  workloads (both GPUs ~65% utilized before benchmark; no external process
  touched). Same current model, J5000/G1000, depth4, float64, saved chain0 point.
  Three postcompile15-step NUTS updates took60.46/61.42/58.58s. Recent CPU
  updates median53.89s, but these are different states and shared hardware:
  this does NOT establish dedicated-GPU performance or ESS/sec. Original
  float32 and batched GPU execution have NOT been benchmarked for this target.
- All CPU samplers remain unchanged/live at last process verification. Ongoing
  monitor session43525 watches all nine detached runs and exits on pilot
  completion/interruption or after~34min. Resume that handle, do not duplicate.
  Goal remains active; no converged new posterior or replacement figures.

## October 15: verified sampling wait and PMMH affinity change

- Previous turn made progress: exact PMMH/pilot recovery and comparison tools.
  Revalidated all nine detached processes by PID/create time; no duplicate run.
- Moved PMMH recovery chains0/1/2 from E-cores22/24/26 to released P-cores8/12/14,
  changing all threads and recording affinity_change.json. State and tuning
  unchanged. Chain3 remains on core28. When the pilot finishes, its core10 can
  be used for chain3 after verifying process completion. Mixed-core wall times
  still cannot establish a NUTS/PMMH speedup.
- Main NUTS passed15 warmup iterations per chain, zero divergences; the depth4
  pilot passed80 total/20 retained, also zero divergences. All saved trajectories
  still hit the depth limit before turning. Await the pilot and later warmup
  evidence before selecting further depth changes; do not restart live runs.
- PMMH chains0/1/2 passed1250 total and chain3 passed1150. Reports and manuscript
  remain provisional/unchanged respectively; no convergence or prior-insensitivity
  claim. Current live process IDs and output paths are in the entry below.

## October 15: recover interrupted PMMH and NUTS pilot

- Previous turn made progress: verified four NUTS startups and matching inputs.
  This turn found original PMMH handles61014/83054/40197/54096 and pilot59154
  absent, with their host PIDs absent too. No completion or Python error; cause
  unknown. PMMH checkpoints all1075/4000, pilot72/120.
- Protocol b5a5f032a committed before exact recovery. New detached processes in
  results/sampling-recovery-v1: pmrw-chain0..3 PIDs2932368..2932371 on
  cores22/24/26/28; nuts-depth4-pilot PID2932372 on core10. Read launch JSON
  for create times and logs. All five saved likelihood/force checks match
  exactly. All four PMMH recoveries advanced to1100 and the pilot to74.
  recovery-verification.json verifies unchanged parent traces, diagnostics,
  steps and timings followed by new finite samples.
- Fixed recovery validation to respect PMMH cached-zero-force convention;
  two recovery tests pass, including corruption detection and warmup replay.
  No target, seed, tuning, initialization or total draw count changed.
- Main NUTS PIDs2917966/68/70/72 remain live. MALA stays stopped. Use recovered
  PMMH paths in reports; do not pool parent and continuation as independent.
- Added compare.py for all-coordinate NUTS/PMMH agreement from frozen reports.
  Development generation passed on completed depth3 NUTS (30 draws) versus
  original PMMH prefixes (75 draws each); visible disagreement, no convergence.
  Visual inspection found/fixed an overlapping legend/title. Reports now freeze
  exact plotted arrays and include NUTS divergences in numerical checks.
  Final posterior, sensitivity, timing and manuscript integration remain unfinished.

## October 15: user restored NUTS versus PMMH as the primary comparison

- The user objected to MALA becoming the main experiment. All eight live
  MALA/reference or sensitivity processes were verified and stopped; checkpoints
  and termination reasons are retained. They are development evidence only.
- Four PMMH chains remain running unchanged; depth4 NUTS pilot remains running.
- candidate-nuts-4chain-v1/protocol.json specifies four independent NUTS chains,
  J5000/G1000, alpha .95, common proper prior, depth4, 500 warmup/1000 retained
  initially. They use PMMH initial positions and metrics, not MALA checkpoints.
  Protocol committed as a2a6a8282 before launch. Detached live PIDs
  2917966/2917968/2917970/2917972 on cores0/2/4/6; each chainN-launch.json
  records create time and command. Logs /tmp/dmop-bayes-nuts-candidate-chainN.log.
  Verified all four processes/configurations and their first5 warmup iterations.
  startup-verification.json confirms finite traces/forces, exact matched inputs
  and common target/prior with PMMH. Monitor session19324 finished normally;
  detached sampler processes remain live. No convergence claim.
  All four PMMH chains have entered retained sampling (1025/4000).
- Prior MALA chains0/2 unexpectedly exited143; exact warmup recovery passed
  saved-state and replay tests before those recovery jobs were also stopped
  for this scope correction. No scientific failure or cause of exit established.
- Need NUTS/PMMH convergence, prior/particle checks and same-core timing before
  replacing paper figures. No manuscript edits in this continuation.

## October 15: full-force NUTS complete; fixed-seed force check

- This continuation included a verified wait on live session83471 (monitor
  cell252 ended normally when that session exited0). Full-force NUTS pilot
  completed60 transitions with30 retained. Its report is generated and the
  complete18-coordinate trace was visually inspected: substantial drift,
  trend bulk ESS2.57, no posterior-convergence claim. Retained position changes
  .933, mean trajectory acceptance .550, zero divergences,7 steps per draw.
  Depth3 reached every time. Do not extend this short configuration by default;
  the already-running depth4 pilot59154 is the next NUTS evidence.
- Moved wider-prior chain2, actual host PID2832635, from E-core16 to released
  P-core14 at22:04:18UTC, all9 threads. affinity_change.json records it; no
  target/state/tuning change. Chain3 remains on E-core18, a candidate for
  moving to core10 after depth4 NUTS finishes. No other process touched.
- Primary MALA chains passed iteration400 covariance adaptation; metrics are
  finite/positive and recent acceptance .12--.72. Their step adaptation remains
  active until1000. The current12 long chains are still in warmup.
- Added/completed conditional_force_check.py, session45243 terminal, output
  conditional-force-check-v1. AtJ1000, two directional derivatives match
  alpha0 within1e-8--1e-9 steps (-68.8037,9.5360); alpha.95 gives22.5336,
  -48.8375. At1e-7 and larger, resampling creates huge finite-difference jumps.
  This confirms local pathwise behavior but NOT global smoothness or better
  alpha0 mixing. No extra alpha0 sampler branch launched; do not infer that
  a local derivative match resolves the conditional target's roughness.
- Active handles unchanged: MALA56838/62806/75156/55847; RW61014/83054/
  40197/54096; wider prior99174/32188/35673/41249; depth4 NUTS59154.
  Full-force NUTS83471 and HMC16291 are terminal. Goal still active; final
  posterior/prior comparison, particle checks and manuscript edits remain.

## October 15: HMC review complete; prior sensitivity underway

- Prior turn made substantive tuning/protocol progress. This continuation
  revalidated the new four-chain handles and observed advancing checkpoints.
  Primary MALA chains have reached25--50 warmup updates; all finite and moving.
  RW candidates passed475 warmup. No retained primary posterior draws yet.
- HMC session16291 completed normally:100 transitions in3899.68s,50 retained,
  parameter acceptance .64, particle-seed acceptance .48. Generated its report
  and inspected the complete18-coordinate trace: substantial drift remains,
  trend bulk ESS2.07/50. Do not extend this configuration. It is preserved
  as a completed tuning trial, not promoted to manuscript evidence.
- Started proper-prior sensitivity in parallel under protocol b42f948a3,
  committed before launch. results/prior-sd2-pmmala-4chain-v1 doubles every
  prior SD (prior_scale2), keeping Pypomp, normalized measurement, J5000/G1000,
  alpha.95 unchanged. Four saved independent initial states/metrics from v2
  inputs; fresh seeds2026102800--03;1000warm+3000draws,step.475,cap2.
- Sensitivity handles: chain0 session99174(core8), chain1 32188(core12),
  chain2 35673(core16), chain3 41249(core18). Logs
  /tmp/dmop-bayes-prior-sd2-chain{0,1,2,3}.log. Do not duplicate. Their mixed
  P/E cores prohibit using elapsed times for method speedup comparisons.
  When NUTS pilots release cores14/10, moving the two E-core sensitivity
  processes there with a recorded affinity change could shorten completion.
- Main MALA handles56838/62806/75156/55847 and RW61014/83054/40197/54096
  continue unchanged. NUTS pilots83471 and59154 remain live. No other sampler
  branches are planned before inspecting these results. Still need independent
  convergence, prior comparison, particle checks and final manuscript figures.

## October 15: faster four-chain candidates now running

- Same-core timings completed: sessions49664 and4577 terminal. Median full
  transition cost on core8 is9.383s (G5000) versus3.437s (G1000), a2.73x
  cost ratio, with J5000 likelihood in both. See timing-force-counts.json.
  This is NOT a posterior ESS/sec speedup. Excluded each first added step;
  parent prefixes are duplicated and must not be pooled as independent draws.
- Committed cheaper-force protocol257ec835b BEFORE replacing the four
  full-force candidates. Exact iteration250 checkpoints/positions/metrics
  preserved in candidate-pmmala-g1000-4chain-v2/inputs, with hashes.
  Verified host PIDs2716071/74/75/69 were only the task's four v1 processes,
  then terminated them at21:37:07UTC. Their handles34823/66148/60444/98321
  confirmed exit143. termination.json records reason, retained checkpoint,
  replacement and3262.72s elapsed preparation per chain. This was deliberate
  tuning from measured cost, not a restart due to observation timeout.
- NEW primary MALA handles: chain0 session56838(core0), chain1 62806(core2),
  chain2 75156(core4), chain3 55847(core6). Logs
  /tmp/dmop-bayes-cheap-candidate-chain{0,1,2,3}.log. All verified live,
  configs show clean Pypomp, J5000 likelihood/G1000 force, same prior/alpha,
  seeds2026102700--03. Start from saved independent parent warmup positions,
  fresh1000 warmup then3000 retained draws; step.475 cap2, adapt only warmup.
  No retained draws reused. Account for parent optimization/warmup costs
  separately; the new clock alone understates total preparation.
- Four original matching RW chains remain live under61014/83054/40197/54096.
  NUTS depth4 session59154(core10) is live; full-force NUTS83471(core14) and
  HMC16291(core12) continue. Core8 and core20 are free after completed timing
  and pilot work. No GPU task launched; both GPUs were occupied by other work.
- Prior sensitivity, doubled-particle checks, adequate independent-chain
  convergence, final same-hardware method comparison and manuscript update
  remain. Do not claim the old figure's qualitative mixing is reproduced yet.

## October 15: smaller-force pilots complete; depth and timing checks

- Sessions86306 (NUTS G1000/J5000) and83983 (joint MALA G1000/J5000)
  completed normally and reports were generated. Their handles are terminal.
  NUTS has30 retained draws, no divergences, position changes28/30, and all
  trajectories reach depth3 (7 integration steps). Its all-coordinate trace
  and trend panels were visually inspected: far too short for convergence.
  MALA has100 retained draws, acceptance .37 vs .51 for the full-force pilot.
  Its average squared trend jump is .000959 vs .001533 for full force, but
  it has strong drift (trend ESS1.48 vs16.6). These short nonstationary ESS
  values must not select a winner. Trend panels inspected; no paper figure
  promoted.
- New depth4 NUTS pilot: protocol fc65e73f3 committed before launch,
  session59154 on core10, /tmp/dmop-bayes-nuts-depth4.log. Same target,
  G1000/J5000, original shared pilot start, step .1123,60warm+60draws,
  seed2026102600. Records the turning flag as well as depth, unlike the first
  frozen pilot. No further trajectory pilots are currently planned.
- Serial same-core kernel timing: full-force continuation session49664 on
  core8, /tmp/dmop-bayes-timing-full.log,20 extra frozen transitions in
  results/timing-pmmala-j5000-core8. At210/220 in latest check. AFTER it
  completes, run the cheaper-force parent pilot-pmmala-g1000-j5000 via
  extend.py,20 draws on the SAME core8, into timing-pmmala-g1000-j5000-core8.
  Exclude the first added transition from timing (compilation). Compare only
  appended times; copied parent arrays are not independent new draws.
- report.py now emits diagnostics.tex using the current SI's compact table
  layout, plus explicit pooled-chain diagnostic and KDE conventions in its
  audit. This avoids reusing the archive's within-chain extrema convention.
  Report regeneration passes. SI itself still unchanged pending valid samples.
- All eight primary candidates remain live and in warmup; full-force NUTS
  session83471 and HMC16291 remain live. No candidate restart or cancellation.

## October 15: gradient particle-count comparison

- Previous turn made implementation/reporting progress. This continuation
  revalidated all twelve live main/pilot session handles (34823,66148,60444,
  98321,61014,83054,40197,54096,83471,86306,83983,16291). None had stopped.
  The metric MALA pilot is already complete; do not restart it.
- Added force_check.py and completed session4900, results/force-count-check-v1.
  It uses candidate0's saved optimized start, the common score-derived metric,
  16 independent force replicates per count, sequentially on P-core8. Timings
  exclude first compilation. J1000/2500/5000 take1.624/3.901/7.931s per force;
  corresponding whitened-force noise traces are11.588/5.301/2.659.
  All raw forces are finite. Inputs, arrays, source and environment saved.
- This confirms a cost/noise tradeoff, not a preferred sampler or convergence.
  Leave running chains unchanged. Full-force NUTS is at22/60 transitions;
  smaller-force NUTS has entered retained sampling. Inspect completed pilots
  before adding further candidates. Main chains still in warmup.
- Core8 is free again after this check. No new long sampler launched. Current
  figure templates preserve the main layout; no manuscript replacements yet.

## October 15: Pypomp continuation and pilot review

- The four MALA candidates are at100/1000 warmup and the four matched RW
  candidates at175/1000 as of this check. All eight remain live under the
  handles below; no retained candidate samples yet. These are multi-hour
  computations. Preserve their frozen configurations and do not relaunch.
- Metric-scaled MALA pilot session52196 completed200 transitions in1984s.
  Its report is generated under pilot-pmmala-metric-j5000/report and all
  three figures were visually inspected. Trend bulk ESS is16.6/100 draws;
  several other coordinates have ESS near1--4 and visible drift. This is
  tuning evidence, not convergence or a replacement manuscript figure.
- NUTS full-force pilot remains live at17/60 transitions; cheaper-force
  NUTS at27/60, cheaper-force MALA at110/200, HMC at50/100. No replacement
  NUTS recipe has been selected. Their original handles remain below.
- report.py now retains the full negative ACF range, places panel letters
  outside the plotting area, and labels chains/prior/warmup on the complete
  parameter grids. The revised trend plot was visually checked. Report
  regeneration and syntax/diff checks pass. No TeX document changed.
- README now distinguishes conditional and joint parameter/randomness
  updates; results navigation includes NUTS and the four-chain candidates.
  extend.py also records continuation affinity and rejects truncated parents.
- Previous push c0b590207 confirmed on origin/main. Next: inspect completed
  cheaper-force/NUTS pilots, assess full independent candidates, then prior
  sensitivity and controlled same-core timing. Preserve the old paper layout
  and make only results-supported manuscript changes. Goal remains active.

## October 15: independent posterior candidates launched

- Committed protocol550e5e614 before launch: four independent chains under
  results/candidate-pmmala-4chain-v1. Pypomp, common proper prior, J5000,
  alpha .95; seeds2026102300--03; independent .12-standardized-coordinate
  initial jitter,400 optimization updates atJ1000,1000warmup,3000retained.
  Score-metric initialization, step cap2, adaptation frozen after warmup.
- Live handles: chain0 session34823 core0; chain1 session66148 core2;
  chain2 session60444 core4; chain3 session98321 core6. Logs
  /tmp/dmop-bayes-candidate-chain{0,1,2,3}.log. At175optimization updates
  all four were improving with finite objectives. No candidate is yet final.
- Also implementing NUTS to preserve the manuscript's original sampler if
  corrected runs converge. Isolated dependencies /tmp/dmop-nuts-deps:
  BlackJAX1.3, jaxopt.8.5, fastprogress1.0.3. Pypomp itself unchanged.
  `nuts.py` supplies deterministic MOP forces to a symmetric integrator,
  retains true unscaled PF energies/weights and MH-updates the explicit
  particle seed. METHOD.md discusses the volume/reversal and admissible-tree
  argument, with Hoffman--Gelman primary source. Known Gaussian extended
  target with a nonconservative force passes120k transitions; integrator
  reversal/Jacobian also pass (2 tests,6.34s). This is not a Dhaka mixing claim.
- NUTS pilot session83471 core14, log /tmp/dmop-bayes-pilot-nuts-metric.log;
  J5000,maxdepth3,30warm+30draws, common start/score metric, step .3.
  Same-target MALA/HMC/RW pilots continue as described below. Old figures
  and both TeX documents remain unchanged; main/SI revision waits on results.


### Latest changes before the next continuation

- Isotropic joint-MALA pilot session70285 completed200 transitions in1812s;
  report generation log is /tmp/dmop-bayes-pmmala-report.log. Do not relaunch the sampler.
- Cheaper-force NUTS session86306 was moved from E-core18 to now-free P-core10
  at21:01:58UTC, host PID2735244. All9 threads moved; chain state unchanged.
  Its affinity_change.json records the timing discontinuity. Discard timing
  intervals straddling that change when comparing cost. Other pilots retain
  their original affinities. No other process was changed or stopped.
- All four primary chains have entered MCMC warmup. Four matched RW chains
  are also live. Candidate acceptance settings and prior are unchanged.
- New NUTS outputs additionally record the turning flag, so reporting can
  distinguish reaching max depth from being stopped by the depth cap.
  Earlier pilot snapshots lack this extra field; do not invent it for them.
- code/README now points to the Pypomp rerun as well as the archive. Neither
  ms.tex nor si.tex has been edited during these runs. Next actions remain:
  inspect completed pilots, choose an efficient corrected NUTS/MALA recipe,
  verify all-coordinate convergence, perform sensitivity and controlled
  timing checks, regenerate figures and revise/compile the manuscripts.

### Latest live run map and checks

- All four MALA candidates finished400 optimization updates with finite
  objectives (-3751,-3751,-3753,-3755 including prior) and entered MCMC warmup.
  Their handles34823/66148/60444/98321 remain live; do not relaunch.
- Same-target RW protocol cfe20cb0a was committed before launch. Four chains
  under results/candidate-pmrw-4chain-v1 share the exact four optimized starts
  and score-derived initial metric. They therefore do not have derivative-free
  preparation. J5000,1000warm+3000draws, seeds2026102400--03. Live sessions:
  chain0 61014(core22), chain1 83054(core24), chain2 40197(core26),
  chain3 54096(core28). Logs /tmp/dmop-bayes-rw-chain{0,1,2,3}.log.
  Timing.md explicitly says these E-core vs MALA P-core wall times cannot
  substantiate a speedup. A serial same-core timing check is still required.
- NUTS/HMC/MALA original pilot handles are still live below. NUTS with full
  J5000 force is expensive; added a separate force-particle count while
  retaining J5000 in the acceptance target. Exact target-invariance test
  passed (39.48s); model.py defaults preserve existing runs. Known-target
  kernel + NUTS suite:6passed,23.48s, only optional jaxopt deprecation warning.
- Cheaper-force pilots: NUTS G1000/J5000 session86306(core18), log
  /tmp/dmop-bayes-pilot-nuts-g1000.log; joint MALA G1000/J5000 session83983
  (core20), log /tmp/dmop-bayes-pilot-pmmala-g1000.log. Source snapshots
  preserve all earlier configurations. No convergence conclusion yet.
- report.py now reads/hashes one atomic checkpoint snapshot, checks common
  model source and actual completion lengths, and reports NUTS divergences,
  step counts and depth saturation. extend.py checks JAX precision/version
  and NUTS dependency versions as well as Pypomp source/backend; it preserves
  per-iteration timings. No manuscript change yet. Remaining gates unchanged.

## October 15: corrected Bayesian rerun in progress

- User authorized rerunning the untempered target, replacing the empirical
  prior if feasible, and subsequently requested current Pypomp rather than
  diffPomp. Qualitative continuity of the figures is preferred, but new
  results need not be identical. Convergence is determined numerically and
  visually, never by matching old marginals.
- New `code/bayesian/corrected/` uses Pypomp's Dhaka simulator and functional
  PF/MOP APIs, with 18 active parameters and one common proper Normal prior
  on estimation coordinates (explicit old-search-range center/scale).
  The empirical KDE is absent. No archived source or original draw changed.
- Added explicit pseudo-marginal random state and corrected MALA/fixed-length
  HMC proposals. Independent random-state refresh uses a likelihood MH ratio;
  rejection retains that state. No NUTS validity claim. Numerical force uses
  stabilized library MOP; endpoint acceptance uses a normalized Gaussian
  observation model, dropping the additive floor and killing invalid paths.
  This small model change and Pypomp initial normalization are documented.
- Preflight J1000 in float32 completed (initial log-likelihood variance 2.21,
  density-floor effect zero in 16 evaluations). MOP and PF numerical values
  diverge through floating-point genealogy; endpoint correction deliberately
  uses PF. Explicit x64 preflight now runs in session 64724; log
  /tmp/dmop-bayes-preflight-x64.log. Do not duplicate a live process.
- Kernel tests check a nonconservative force's reversible, volume-preserving
  leapfrog and an analytically known extended Gaussian target. Initial HMC
  test tuning was near a half-period resonance and mixed poorly; reduced
  integration length. Retest session 25283. Runner now saves full checkpoints,
  source snapshots, target/configuration, warmup adaptation and diagnostics.
- X64 preflight is complete: log-likelihood variance 3.28, maximum MOP/PF
  discrepancy 9.1e-12, density-floor effect 6.4e-12 across 16 seeds.
  Pilot MALA J1000 is live in session 10897; log
  /tmp/dmop-bayes-pilot-mala.log. It runs 50 optimization updates, 100 warmup
  and 100 sampling transitions; these are tuning draws, not final inference.
- Next: inspect pilot, then independent
  final chains and prior/particle sensitivity. Regenerate existing figure
  layouts, update main/SI minimally, compile and inspect. Goal NOT complete.
  No manuscript changes or final replacement figures yet.


### Pypomp pilot status (same task, latest)

- All four kernels (conditional MALA, fixed-length HMC, joint particle MALA,
  and particle random-walk MH) pass the known-target test, together with
  reversibility/determinant and force-regularization checks: 3 tests, 10.70s.
  `METHOD.md` gives exact acceptance ratios; no posterior approximation is
  hidden in the MOP force. Joint particle MALA follows Nemeth et al. 2016.
- The first conditional-MALA J1000 pilot completed (session 10897 terminal).
  Its step size collapsed to 2.14e-5; retained parameter/randomness acceptance
  .38/.04. It does NOT converge. Report/plots generated and inspected;
  no manuscript image is replaced. Endpoint sorting barely reduces paired
  log-ratio variance (5.64 to5.48 in24 pairs), so it is not selected.
- Complete preflights: J5000 reduces variance substantially. Authoritative
  `preflight-j5000-full-x64` sets x64 before importing model data and gives
  variance .262 over16 seeds, max MOP/PF discrepancy8.7e-12, floor effect
  5.9e-12. Earlier x64 preflights imported library covariates before changing
  precision; retained as development artifacts, not final settings. GPU1
  J1000 timing was worse than CPU; no GPU worker remains. One CPU core gives
  about1.6s/PF and6.2s/force atJ5000.
- Live tuning jobs: session70285 joint particle MALA J5000, log
  /tmp/dmop-bayes-pilot-pmmala.log (100warm+100draws, accepting .3-.5 early);
  session85177 matched particle random-walk MH J5000, log
  /tmp/dmop-bayes-pilot-pmrw.log (same initial vector and prior).
  These read frozen source snapshots in their own result directories.
- Hessian-based proposal metric failed with nonfinite second derivatives;
  preserved attempt.py/failure.json under results/metric-j1000. No such
  matrix was used. Replacement score-second-moment preconditioning is a
  proposal-tuning heuristic only (not Fisher/posterior covariance), running
  session16923, log /tmp/dmop-bayes-score-metric.log. It saves64 scores and
  a regularized Cholesky factor. Runner accepts --metric for subsequent
  candidates and freezes tuning after warmup.
- `report.py` regenerates the old four-panel trend layout plus18-coordinate
  traces and common-prior overlays; it checks settings/seeds before pooling.
  Still no independent four-chain final run, sensitivity analysis, replacement
  figures, or manuscript integration. Goal remains active.

### Latest continuation handles

- Joint MALA without a metric remains session70285 (about60/200 at last
  update). Matched RW session85177 completed all200 transitions in354s;
  report generated and visually checked. These are not converged final runs.
- Score preconditioning completed64 replicates, saved as
  results/score-metric-j1000/chol.npy. New pilots use this fixed factor:
  session52196 joint MALA, /tmp/dmop-bayes-pilot-pmmala-metric.log;
  session16291 four-step HMC, /tmp/dmop-bayes-pilot-hmc-metric.log.
  MALA accepted7/10 early moves and reached the current step cap .5;
  inspect whether a higher cap or longer trajectories are needed before
  freezing final settings. HMC is slower and may still be compiling or
  computing its first5 transitions; poll its handle, do not duplicate it.
- Added extend.py to continue completed chains exactly into new directories,
  using saved source, RNG, particle state, covariance and step size. A new
  save/restore replay test passes bit-for-bit. Four tests pass in11.80s.
- Figures from the completed pilots have been inspected. Fixed crowded tick
  labels for tiny ranges. Neither pilot provides final posterior inference.
  No TeX change this turn, so there is no new compilation obligation yet.
  Finish pilot comparisons, select settings, run four independent chains,
  continue to the stated numerical/visual gates, then prior and particle
  sensitivity, matched-target comparison, main/SI edits and visual compilation.

## October 15: benchmark follow-up completed

- All 28 SPX tuning fits completed. Alpha .97, rate .001 with cosine decay,
  and warm=100 won; alpha .995/1, rate .01 and warm=300 did not improve the
  tuning median. No claim that these candidates exhaust useful tuning.
- The selected recipe is already represented by the 20 archived final IFAD
  fits. Reused all 20 with unchanged 24x5000 evaluations and original timings.
  Checked model/data/Pypomp/settings/seeds and reviewed the historical
  runner difference (Gaussian DS19 additions only, no IFAD changes).
  `audit_spx_retuning.py` verifies 28 tuning fits and 80 archived SPX finals.
  Raw files, selection, hashes and protocol deviation are in spx-retuning.
- Automatically launched duplicate final fits were stopped and preserved in
  abandoned-repeat, including a brief accidental restart after an audit
  assertion. They are excluded from reports. No SPX worker remains needed.
  Session 87247 and restarted session 84469 are terminal (SIGTERM).
- Report now verifies both fresh replacement and unchanged-recipe reuse.
  Ten report tests pass. Regeneration preserves all numerical results and
  the corrected Dhaka E range; provenance now includes the SPX tuning.
  SI explicitly states alpha=.97 and its existing cosine schedule.
- Bayesian additions above are also implemented. Local main/SI builds pass;
  new SI pages 59–61 and main pages 13–14 were visually inspected. The native
  compiler was attempted for both but cannot access project macros.tex.
  Main has one unrelated pre-existing overfull theorem-table line.
  Final SI/main build after the alpha wording passes. Regenerated final and
  progress-detail figures were visually checked; the SI has no overfull
  boxes or undefined references. Host process inspection confirms no
  task-owned SPX worker remains. All authorized work is complete except
  original prior overlays/kernel diagnostics that cannot be recovered from
  the saved archive; these limitations are explicit, and no new Bayesian
  runs were requested. Changes are committed and pushed on main.

## October 15: Bayesian implementation recovered and documented

- User authorized a short SI implementation/prior subsection and diagnostic
  table, asked to preserve the preliminary promise in main text, and later
  said not to spend much effort on chain diagnostics. Main edits are local:
  retain promising preliminary faster exploration, qualify the historical
  iteration comparison, remove the incorrect 6.3%-of-MLE claim, and refer
  technical details to SI. No new sampler runs or broad validation study.
- Recovered exact original Figure 6 chains from hetankevin/diffPomp
  revision 65e64d3e24d7d82839b413067c30b4224dee43c9. Current diffPomp's figure
  hash matches the manuscript. `code/bayesian` preserves source, four arrays,
  provenance and ArviZ 0.22.0 diagnostics with a regenerable SI table.
- Material findings: 19 finite sampled coordinates; rho/c fixed, constant
  omega unused by the seasonal simulator. Separate joint KDE prior per
  chain; Scott factor 10000^(-1/23); both log prior and particle log objective
  divided by N=600. Historical source alpha=.95 (later notebook says .97),
  J=1000, fixed NUTS step .05, depth 5, 1500 draws, 500 discarded. NumPy
  randomness is captured at JAX tracing; no augmented pseudo-marginal state.
- R-hat across these different-prior runs is descriptive only (1.03–2.45).
  ESS/MCSE computed per chain, without pooling distinct targets. Diagnostics
  do not establish stationarity or exact posterior sampling. The 6.3% value
  is .02*.95^40*sqrt(600) on the transformed perturbation scale, not a
  measured KDE width. Removed it from the caption.
- Prior clouds, tracing-time seeds, kernel results and timings were not
  archived. Do not fabricate overlays, divergences, depth counts or ESS/s.
  Original sampler code is preserved for inspection, not executed by the
  diagnostic script. User says clouds are in diffpomp, but the inspected
  tracked archive does not save them; no further question is pending.
- SI build and visual review of new pages 59–61 pass. Native compiler cannot
  access macros.tex. Main build passes, with an existing unrelated overfull
  theorem-table line; final rebuild/visual check pending caption changes.
  SPX retuning below is still running and must be integrated when complete.

## October 15: proposed response to review points 6 and 7 (chat only)

- Read the attached `report.pdf` points 6/7 against the Bayesian section,
  introduction, SI S8, and the manuscript's cholera/discussion prose.
  User requested critique and minimal proposed additions in chat, with
  details deferred to SI; do not insert these proposals without approval.
- Proposed dropping the 700,000-to-500 speedup and unverified convergence
  claims, defining the intended empirical-prior target and actual sampler
  in SI, and adding all-parameter diagnostics and prior/sample overlays.
  A broad matched-target efficiency study is unnecessary if the example
  is explicitly exploratory. Diagnostics do not prove target correctness.
- Verified Fasiolo arXiv:1411.4564v2: 1.4 million iterations, half discarded;
  different transition scheme, gamma process noise and negative-binomial
  observations. Neal arXiv:1206.1901 explains why an approximate force need
  not bias an appropriately corrected Hamiltonian sampler. Neither source
  validates the existing NUTS kernel. The generating sampler and raw chains
  were not located in current dmop/quant Python/notebook sources. Do not
  invent its potential, randomness treatment, bandwidth or diagnostics.
- No Bayesian manuscript edits or new Bayesian experiments were made.
  The already-running SPX retuning remains active below.

## October 15, 2026: acting on Ed's benchmark comments (in progress)

- Tasks: retune SPX IFAD on separate starts and refit all 20 final starts;
  retain panel E with [-3840,-3740] detail limits and a full-range companion;
  clarify warm starts and spx naming; document local covariance ranks and
  parameter-dependent support; distinguish fine-grid backward-sampling
  issues from singular densities; add restrained benchmarking discussion.
  Do not expand into a general complexity/extension study or send email.
- `retune_spx.py` runs four disjoint CPU workers. Protocol: fresh tuning
  seed 2026101501, four starts, alpha {.97,.995,1}, initial rate {.001,.01},
  existing cosine decay; compare the winning recipe at warm=100 vs 300,
  then refit 20 original final starts with 24x5000 independent evaluations.
  Output `results/spx-retuning/`; live log `/tmp/dmop-spx-retuning.log`.
  Existing SPX config already used cosine decay and alpha .97. Its Fisher
  score uses AD of the fixed-path objective, not hand-coded derivatives.
- `covariance_support.py` and its note establish a crucial distinction:
  Dhaka's local covariance range is span(e,v), where
  v=(0,-gamma-m,gamma,0,0,m), and changes with m/gamma. Daphnia's implemented
  order-1.5 covariance has rank eight in the strictly positive interior,
  but rank three at its specified initial state. Do not call both uniformly
  rank-deficient/"irregular". Saved numerical checks confirm the algebra.
- Still to do: finish tuning/final fits, select all-20 replacement in report,
  adjust E and captions, SI prose/rank explanation, tests/build/visual review.

### Presentation and density explanation completed; SPX jobs continue

- Panel E's final/distribution and progress detail views now use
  [-3840,-3740]. The cropped final plot identifies seven omitted estimates
  (IF2 1, DS19 3, CTDD21 3); the new `final_likelihoods_full` preserves all
  points, and all data remain in every summary. Captions identify D/E warm
  starts and the identical archived Dhaka source. Visually checked the plot.
- SI now defines `pypomp.models.spx`, explains that Fisher scores still use
  fixed-path AD, gives the Dhaka support plane and Daphnia interior/boundary
  ranks, and distinguishes these from the fine-grid mixing issue discussed
  by Karppinen et al. The covariance floor is explicitly at the endpoint.
  Added a short benchmarking paragraph, without calling SPX saturated.
- The report supports the all-20 SPX replacement when the completion marker
  appears, validates starts and chosen settings, and never chooses per-run
  winners. Nine report tests pass, including retaining a worse replacement
  fit. Local SI build passes with no errors, undefined references or
  overfull boxes. Final SPX integration and final compile/visual check remain.
- Visually reviewed SI pages 67, 68 and 79 after the density additions.
  Native compilation was attempted and cannot access project `macros.tex`;
  the successful local build is the compilation check for this project.

## October 9, 2026: add DS19 implementation table for panels A--E

- Added Table S-6 near the start of S12, mapping each model to its DS19
  transition density, parameter update and evaluation likelihood. Later
  table numbers update automatically. The Gaussian/SPX fitting and evaluation
  transitions agree; Daphnia/Dhaka use separate Gaussian fitting models.
- Explained that Daphnia and Dhaka already use local order-1.5 moments,
  composed across an observation interval using linearization. This is not
  a claim of strong order 1.5 for the resulting block Gaussian. Dhaka's
  single-step covariance has rank at most two in six dimensions; Daphnia
  uses quarter-day steps, and a single coarse observation-interval step
  has not been validated. No fitting procedure or result changed.
- Local SI build passes with no errors, undefined references or overfull
  boxes. Visually reviewed the explanation and table, and made the table
  columns ragged right for readability. Native compiler still lacks access
  to `macros.tex`; `git diff --check` passes.

## October 9, 2026: explain DS19 density requirements and comparison limits

- Added to S11 why SAEM and the implemented score update both require
  process densities: the objective contains their logarithms, score updates
  differentiate them, and backward smoothing evaluates them. Explained
  singular single-step Dhaka Euler transitions and the unavailable monthly
  endpoint density, versus simulator-based IF2/IFAD.
- Expanded S12's Daphnia description with interval lengths and differing
  boundary rules. Discussed the Gaussian fitting/evaluation mismatch,
  100 versus 1,000 fitting particles, sampled-path gradient error and
  averaged-score optimization as possible contributors, not established
  causes. Explicitly noted the absent Daphnia SAEM comparison and proposed,
  unperformed comparison of Gaussian/Euler objectives along saved searches.
- Clarified that the completed Dhaka optimizer comparison retained the same
  Gaussian transition and therefore does not isolate approximation error.
- Verification: built-in compiler attempted (project `macros.tex` unavailable);
  local in-place SI build passes with no errors, undefined references or
  overfull boxes. Revised pages visually reviewed; `git diff --check` passes.
  No numerical code, results or equations changed.

## October 9, 2026: revise benchmark prose against the manuscript

- Reread the cholera application and computational-efficiency sections of
  `ms.tex` as the writing sample. Reworked S12's results, SAEM explanations,
  computation-time discussion and experimental settings to use the authors'
  direct account of the experiments. Removed table-by-table narration and
  generic qualifications. Revised both benchmark caption fragments as well.
- Preserved all equations, algorithm settings and numerical results; no code,
  fits or plots changed. Verified displayed equations against HEAD and that
  the source before S12 is unchanged. `git diff --check` passes.
- Native compilation still cannot access `macros.tex`. The existing in-place
  local SI build succeeds, with no errors, undefined references or overfull
  boxes. Reviewed the compiled results prose and page breaks. Existing
  amsthm/hyperref warnings remain.

## October 9, 2026: benchmark reports integrated and verified

- Finished the requested benchmark deliverables. Existing `si.tex` now includes
  the five-model results discussion, three comparison figures and the central
  table, followed by the full cold/warm Dhaka score-versus-SAEM comparison and
  its table. These are Figures S-10--S-13 and Tables S-6--S-7 in Section S12.
  All numerical jobs were already complete; no fitting source or runs changed.
- Fixed the missing optimization-detail legend by reserving a figure footer.
  A/B starting-point labels are now one-based integers; its table preserves
  the 15.5 median iteration count. Regenerated both reports from all completed
  inputs and updated result/publication READMEs with current navigation.
- The prose reports actual outcomes: no consistent IFAD advantage on the
  smaller examples; higher median/best likelihoods on Daphnia and Dhaka.
  Numerical SAEM is higher in only 2/20 cold and 4/20 warm A/B pairs. Its
  median paired differences are -608.83 and -0.62, respectively. Full ranges,
  failures and outliers remain visible; detail panels are explicitly cropped
  linear views with corresponding full-range plots. Bands use alpha .10.
- Verification: regenerated reports pass all completeness/provenance checks;
  12 targeted report/A/B tests pass. The preceding saved independent audit
  covers all 80 Daphnia fits and 40 A/B pairs. Visually inspected every new
  figure and both tables in the compiled SI. `make dmop/si.pdf` succeeds
  in place (87 pages), with no undefined references, TeX errors or overfull
  boxes. Existing amsthm/hyperref warnings remain. Built-in compilation was
  attempted but cannot find the project dependency `macros.tex`; source and
  editor were preserved, using the established local build instead.
- No numerical jobs or requested benchmark integration work remain. The
  sections below are historical progress records, not the current job state.

## October 9, 2026: all numerical jobs complete; final reports generated

- Rechecked after the user's status request: all eight fitting subprocesses
  and the report waiter have exited. All four Daphnia batches are complete
  (80 final fits, 20 starts); corrected Dhaka A/B is complete (40 pairs:
  20 cold and 20 warm). Both report jobs generated their full artifacts.
- `audit_results.py` passes all 80 raw Daphnia final evaluations and all
  40 A/B pairs, including checkpoint selection, Q acceptance, reused-score
  provenance and raw likelihood/MCSE reconstruction. Saved audit:
  `code/benchmarks/results/final-daphnia-and-saem-audit.json`.
- Daphnia median likelihoods: IFAD -861.297586, MPIF -864.978475,
  DS19 -867.021166, CTDD21 -865.026518; one DS19 failure retained.
  Warm A/B medians: score -3771.543503, SAEM -3772.440844. SAEM is
  higher in 4/20 warm pairs; median paired difference is -.619493.
  Cold A/B is unchanged: SAEM higher in 2/20 pairs, median paired
  difference -608.825345. The warm start greatly reduces discrepancies
  but the numerical SAEM update does not improve the overall comparison.
- Visually inspected all three cross-model panels and the full A/B plot.
  All use linear axes and retain failures/outliers. One presentation issue
  remains: `optimization_detail` lacks a visible legend despite the
  generator requesting an outside legend; fix its layout before SI use.
  The A/B start-index ticks are fractional (0, 2.5, ...); use integer,
  preferably one-based labels for the final publication figure.
- No numerical jobs remain. Required next work: fix these figure layout
  details without refitting; review tables; add the prepared figure/table
  fragments and concise actual-results prose to existing `si.tex`; attempt
  native compilation, run the in-place local build if needed, inspect the
  changed pages, update READMEs/handoff, and commit/push final artifacts.
  The goal remains active; generated reports alone do not complete it.

## October 9, 2026: twelve warm pairs audited

- Warm starts 9--11 completed; all 32 complete A/B pairs pass the full
  audit (20 cold, 12 warm). New Euler likelihoods (initial, score, SAEM):
  9 (-3756.297483, -3755.768214, -3755.204008);
  10 (-3791.836165, -3792.611014, -3795.287949);
  11 (-3893.986468, -3893.893114, -3933.479175).
  The last pair has SAEM 39.586061 log-likelihood units below score;
  raw replicate summaries and fixed-Q acceptance pass the audit. Retain
  this result and include its full range in the final warm-start plot.
- The three newly completed pair directories are saved here. Latest audit:
  `/tmp/dmop-corrected-pair-audit.json`. The prior eight-pair preview is
  historical and must not be presented as the latest full comparison.
- All eight workers remain live. Daphnia remains at 16/20 complete starts;
  the final four starts are fitting. Eight warm pairs, complete reports,
  SI integration and final compilation/visual checks remain outstanding.
  No frozen fitting/reporting source changed.

## October 9, 2026: sixteen Daphnia starts and nine warm pairs audited

- Daphnia now has 16/20 complete starts. Recomputed all 64 final
  likelihoods and MCSEs from raw per-unit replicates; all match. New
  fourth-group likelihoods (MPIF, IFAD, DS19, CTDD21):
  3 (-866.203513, -860.261047, -864.597464, -865.291328);
  8 (-866.507163, -860.965894, -863.516247, -864.396072);
  13 (-863.607608, -864.769183, -869.968223, -871.970273);
  18 (-869.537062, -863.561325, -867.357687, -868.548024).
  These 16 fits are successful; the earlier DS19 start-17 failure remains.
- Warm start 8 completed: initial Euler likelihood -3789.371005
  (MCSE .159980), score -3789.211443 (.240365), SAEM -3789.252831
  (.185231). All 29 complete A/B pairs pass the audit (20 cold, nine warm).
  Only this newly completed pair is saved here; live partial batches remain
  untracked. Audit snapshots: `/tmp/dmop-corrected-pair-audit.json` and
  `/tmp/dmop-daphnia-live-evaluation-audit.json`.
- All eight subprocesses remain live. Each Daphnia worker now has only
  its fifth start remaining. Eleven warm pairs and the final reports/SI
  integration are still outstanding. No frozen source changed.

## October 9, 2026: eight warm pairs audited and previewed

- Warm starts 5--7 completed; all 28 complete A/B pairs pass the audit
  (20 cold, eight warm). New Euler likelihoods (initial, score, SAEM):
  5 (-3788.346183, -3788.225969, -3790.741260);
  6 (-3772.307038, -3772.315437, -3774.985745);
  7 (-3760.581522, -3763.810860, -3758.240660).
- Generated and visually reviewed `/tmp/dmop-saem-ab-warm-eight/`, with
  the plotting script, paired/summary CSVs, audit JSON and comparison
  PNG/PDF. The figure explicitly shows the first eight of 20 warm pairs;
  all axes are linear and difference bars are twice paired MCSE. Score
  is higher in six pairs and SAEM in two. This remains a preliminary
  preview outside the SI and official 40-pair report.
- Only the three newly completed pair directories are committed here.
  All eight workers remain live. Daphnia remains at 12/20 complete starts;
  remaining searches, final reports, SI integration and final checks
  are outstanding. No frozen fitting/reporting source changed.

## October 9, 2026: twelve Daphnia starts evaluated

- Third-group starts 7 and 12 completed all four evaluations. Recomputed
  all 48 saved final likelihoods and MCSEs from raw per-unit replicates;
  all match. Latest snapshot: `/tmp/dmop-daphnia-live-evaluation-audit.json`.
  New likelihoods (MPIF, IFAD, DS19, CTDD21):
  7 (-865.819883, -861.166092, -868.086986, -863.639140);
  12 (-863.790313, -861.044346, -865.870326, -865.150047).
  These eight fits are successful; the earlier DS19 start-17 failure
  remains in the data and must stay in final failure counts and plots.
- Daphnia now has 12/20 complete starts; all four workers advance to their
  fourth starts. Corrected A/B remains at 20 cold and five warm complete
  pairs; workers 1--3 are fitting SAEM after finishing their next score fits.
  All eight subprocesses are live. Full reports, SI integration and final
  compilation/visual checks remain pending. No fitting source changed.

## October 9, 2026: fifth warm pair and ten Daphnia starts audited

- Warm start 4 completed and passed the A/B audit: initial Euler
  log-likelihood -3756.536937 (MCSE .098147), score -3759.205608 (.121781),
  SAEM -3761.634421 (.285623). Both final estimates are below the initial
  estimate. All 25 completed pairs pass the audit (20 cold, five warm).
  Saved this complete pair; the remaining 15 warm pairs are still running.
- Daphnia starts 2 and 17 now have all four evaluations. Likelihoods
  (MPIF, IFAD, DS19, CTDD21):
  2 (-863.101211, -860.737705, -873.175516, -864.736433);
  17 (-864.022364, -865.640910, -878.765441, -877.397745).
  DS19 start 17 retains `nonfinite_update`; all others here are complete.
  Recomputed all 40 final likelihoods and MCSEs across ten completed starts
  from raw per-unit replicates. All match. Temporary audit snapshots remain
  `/tmp/dmop-{daphnia-live-evaluation,corrected-pair}-audit.json`.
- All eight fitting processes remain live. Other third-group Daphnia
  searches and remaining warm pairs continue. Full reports and SI
  integration remain pending; no frozen source changed.

## October 9, 2026: Daphnia DS19 failure retained

- At zero-based final start 17, DS19 accepted five updates and then
  produced a nonfinite score on its sixth attempt (finite fitting
  log-likelihood -916.925841, NaN score norm). Verified the last saved
  parameter vector is finite at iteration 205 and the fit JSON records
  `nonfinite_update`. No rerun or omission: the existing pipeline evaluates
  this last estimate and carries the failure status into plots and tables.
- The 190.23-second recorded fitting time includes the shared MPIF warm
  start; DS19 continuation itself ended after 3.29 seconds. CTDD21 and
  independent evaluations for this start are still pending. This result
  must remain a failure even if its final likelihood is competitive.
- All eight subprocesses remain live; no frozen source changed. Complete
  counts remain eight Daphnia starts and 24 A/B pairs (20 cold, four warm).

## October 9, 2026: first four warm pairs audited and previewed

- Warm starts 1--3 completed; all 24 completed A/B pairs pass the full
  audit (20 cold, four warm). New Euler log-likelihoods (initial, score,
  SAEM): 1 (-3761.386114, -3763.034330, -3763.489515);
  2 (-3768.131797, -3768.580306, -3768.771922);
  3 (-3775.662563, -3775.024334, -3775.599359).
- Generated and visually reviewed `/tmp/dmop-saem-ab-warm-preview/`:
  `comparison.{png,pdf}`, `plot.py`, `paired.csv`, `summary.csv`, `audit.json`.
  This explicitly shows only the first four of 20 warm pairs, using linear
  axes and twice paired MCSE for differences. It is outside the SI and
  official full-study report. Among these four, score is higher in three;
  all absolute paired differences are below one log-likelihood unit.
  No conclusion about the full warm comparison is warranted yet.
- All eight worker processes continue. Daphnia remains at eight complete
  starts; its third group is fitting. Remaining warm pairs, Daphnia
  searches, final reports and SI integration are still outstanding.
  Frozen fitting/reporting source is unchanged.

## October 9, 2026: eight final Daphnia starts evaluated

- Starts 0, 1, 5, 6, 10, 11, 15 and 16 now have all four final method
  evaluations. Recalculated all 32 log-likelihoods and MCSEs from the ten
  raw replicates for each of eight mesocosms; all agree with checkpoints.
  Snapshot: `/tmp/dmop-daphnia-live-evaluation-audit.json`. This does not
  certify the unfinished five-start batches or the full 20-start study.
- Newly complete starts (MPIF, IFAD, DS19, CTDD21):
  1: (-865.214432, -861.859893, -865.151186, -865.572389);
  6: (-862.884996, -857.131953, -864.009556, -860.967115);
  11: (-865.743639, -861.580744, -871.777049, -864.091729);
  16: (-864.742519, -863.680370, -877.160630, -868.104683).
- All eight fitting processes are live and accumulating CPU time; Daphnia
  workers proceed to their third starts. The corrected A/B study remains
  at 20 complete cold pairs and one complete warm pair. Completed warm
  data are pushed; live partial Daphnia directories remain untracked until
  their batches finish. Full reports and SI integration remain pending.

## October 9, 2026: first corrected warm pair complete

- Warm start 0 completed both 850-second fits and independent Euler
  evaluations. Log-likelihoods (MCSE): initial -3769.681021 (.118786),
  score -3770.771569 (.142097), numerical SAEM -3769.895942 (.156746).
  Neither final point improves on this starting estimate. This is one
  pair, not the full warm-start comparison.
- All 21 completed pairs pass `audit_results.py --dhaka-ab`, including
  raw likelihood/MCSE recalculation, selected checkpoint timing, and
  fixed-Q acceptance. Audit: `/tmp/dmop-corrected-pair-audit.json`.
  Only the complete `warm/start000` directory is saved in this commit.
- All eight fitting subprocesses remain live. The next four Daphnia
  DS19 fits (starts 1, 6, 11, 16) have finished; CTDD21 and evaluation
  follow. The remaining warm pairs, Daphnia searches, final reports and
  SI integration are outstanding. No frozen fitting source changed.

## October 9, 2026: results directory navigation clarified

- Expanded `code/benchmarks/results/README.md` with the exact inputs to the
  current report: selected J=500 Gaussian IFAD/IF2/DS19 replacements,
  CTDD21-only use of the original Gaussian final directories, the four
  SPX/Daphnia batches, and archived 100-start Dhaka sources. Earlier
  Gaussian alternatives are clearly identified as retained history.
- Linked the separate corrected 20-cold/20-warm score/SAEM study and the
  publication-output README; clarified that a missing completion status
  means unfinished. Verified all five local Markdown links and checked
  whitespace. No fitting or reporting source changed.
- All eight experiment subprocesses remain active. Complete cold A/B
  results are saved and previewed; warm A/B and final Daphnia fitting
  continue. Final figures/tables and SI results remain pending.

## October 9, 2026: complete corrected global-start comparison

- All 20 corrected cold pairs pass `audit_results.py` (raw likelihood and
  MCSE recalculation, selected checkpoints, fixed-Q acceptance and reused
  score provenance where applicable). Starts 17--19 are newly complete.
  Global-start medians: score -4050.488630, numerical SAEM -4834.427999;
  maxima: -3967.135085 and -3963.488588. SAEM is higher in 2/20 pairs.
  Two score fits stopped early and retain their last eligible estimates;
  all SAEM calls ended at their time budget. No fit is omitted.
- Generated and visually inspected the complete cold-only two-panel figure
  at `/tmp/dmop-saem-ab-global/comparison.{png,pdf}`. `plot.py` there refuses
  incomplete cold data and reruns the full audit; `paired.csv`, `summary.csv`
  and `audit.json` preserve its inputs/results. All axes are linear, failed
  score fits are crosses, and paired differences have twice paired MCSE.
  Its title explicitly states that the warm-start comparison is running.
  This preview is not inserted into the SI or the complete 40-pair report.
- All four A/B workers are now on warm starts 0--3. No warm pair has
  completed. Daphnia has completed four starts (16 fits) and is fitting
  the next four; all four second full-MPIF searches have finished.
  The same eight subprocesses remain live. Full warm comparison, complete
  Daphnia study, final publication figures/tables and SI results remain
  outstanding; goal active.

## October 9, 2026: search/evaluation settings added to SI

- Added S12.2, `sec:benchmark-settings`, giving the common 20-start design,
  SPX particle counts and iteration schedules, Daphnia's common 200-MPIF
  starting estimate and time-matched continuations, learning-rate schedules,
  and independent final/intermediate evaluation particle and replicate
  counts. The 100-search Dhaka comparison is explicitly linked to S11.
  Verified details against saved final configurations and the fitting code.
- Clarified that IFAD/DS19/CTDD21 Daphnia times include their shared warm
  start; the separate MPIF search has 680 iterations. No fitting source,
  settings, checkpoint selection or active run was changed.
- Native compiler attempts returned `No handler registered`. Local in-place
  build succeeded, with no errors, undefined references or overfull boxes.
  Visually inspected S76--S77. The SI is now 79 pages; results and figure
  fragments remain to be added after the full experiments complete.
  Build log: `/tmp/dmop-si-benchmark-settings-build.log`; page images:
  `/tmp/dmop-si-benchmark-settings-{76,77}.png`.
- All eight persistent fitting processes continue. Four complete Daphnia
  starts (16 method estimates) and 17 cold A/B pairs have been audited;
  remaining cold/warm A/B pairs and Daphnia starts are in progress.

## October 9, 2026: first four final Daphnia starts evaluated

- Final starts 0, 5, 10 and 15 now have complete MPIF/IFAD/DS19/CTDD21
  evaluations, all with successful status. Independently recalculated all
  16 saved likelihoods and MCSE values from their raw per-unit replicates;
  all match. Temporary audit record:
  `/tmp/dmop-daphnia-live-evaluation-audit.json`. This is a partial-run audit,
  not proof that any five-start batch or the 20-start comparison is complete.
- Likelihoods by start (MPIF, IFAD, DS19, CTDD21):
  0: (-864.678419, -867.404201, -872.316008, -872.010359);
  5: (-865.992761, -861.385910, -866.684644, -864.251885);
  10: (-862.849314, -861.015475, -865.709665, -864.902989);
  15: (-866.634277, -859.958297, -864.380405, -864.407348).
  Do not infer the complete study's ranking or omit the first start, where
  MPIF is best. All fitting stages and evaluations retain every search.
- The four persistent Daphnia workers are advancing to their next starts.
  All four corrected Dhaka A/B workers remain live; 17 cold pairs are
  complete and the first warm pair is underway. Full reports, results
  prose, SI integration and final visual/compilation checks remain pending.

## October 9, 2026: first corrected warm-start pair underway

- Cold start 16 completed and passed the 17-pair A/B audit. Euler
  likelihoods: initial -10545.466753, score -4131.177923 (MCSE .519321),
  SAEM -4498.273845 (.297115). Score used its time budget; SAEM completed
  14 eligible iterations, selecting at 819.54 s before its 850 s deadline.
- Worker 0 has begun warm start 0 (SAEM first under the alternating arm
  order). The other workers are finishing cold starts 17--19. No warm
  pair is complete. At the current budgets the remaining warm fits take
  roughly 2.5--3 hours including evaluation; this is an estimate, not a
  deadline or a reason to change the declared experiment.
- Verified the SAEM objective uses a fixed-size path buffer and compiles
  that shape before the fitting timer starts. No timing-caption or source
  change was needed. Frozen fitting code remains untouched.
- All four first DS19 Daphnia fits completed successfully at starts 0, 5,
  10 and 15 (539, 545, 536 and 514 updates respectively). CTDD21 and
  independent evaluation follow. The same eight actual fitting subprocesses
  remain live. Full reports, results prose and SI integration remain pending.

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
