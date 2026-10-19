# Historical proposal on the current target

This completed four-chain experiment restores the fixed diagonal proposal
specified in the old MH notebook: estimation-coordinate SD equal to one
hundredth of each search-range width. In the current standardized coordinates
this is SD 0.02. There is no adaptation. The current untempered Pypomp target,
proper prior, 10,000 particles and four initial states are unchanged from
`../gpu-comparison-v2/pmrw`. Seeds differ. `protocol.json`, `launch.py` and
`launch.json` record the settings and execution; `completion-verification.json`
records the final checkpoint hashes and process completion.

All four chains completed 1,500 iterations. Acceptance was 10.2%, 9.6%, 10.7%
and 13.6%. The trend traces move slowly and their empirical marginal densities
remain separated. This also occurs in many other coordinates, including
gamma, beta1--beta3 and the immigration coefficients. The run reproduces poor
mixing at this budget, but not the original trend chain's excursion to -1.22.
The current tuned proposal explores the trend coordinate more widely over
the same iteration count. This comparison concerns proposal choice and
adaptation together; it is not a timing comparison.

`comparison/same_length_traces.png` shows the archived MH draws, current tuned
PMMH and this control on common linear axes. Every trace uses iterations
1--1,500 and every density uses 501--1,500. The plotted arrays and input hashes
are frozen beside the figure. Reproduce it with the command in
`../../../historical-mh/README.md`, choosing a new output directory.

`review-1500/` contains all 18 trace and marginal panels, diagnostic tables
and frozen draws from `../../report.py`. That report uses all 1,500 iterations
because the control has no adaptive warmup; its density window therefore
differs from the three-row comparison. Worst rank R-hat is 2.706 and minimum
bulk ESS is 4.73. These are unconverged chains, not posterior estimates.
The comparison and all 18 trace and marginal panels were visually inspected.

Restoring the proposal does not restore the old experiment's target. The
archived source specifies division of both log likelihood and log KDE prior
by 600, with separately fitted priors and tracing-time randomness. An active
MH sampling cell and the fitted prior clouds have not been recovered. Thus
the original large excursions cannot yet be attributed to a particular
change, nor can an exact replay be claimed. See the historical archive for
the distinction between the saved draws and the surviving source.
