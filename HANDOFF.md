# Handoff

## Current state

The corrected IF2-warm-start 100-run experiment is active as the user service
`dmop-if2warm-corrected-final100-v2.service`. Do not start another GPU job
while it is running. At the September 23 status check, Corenflos had 96/100
checkpoints and Ditlevsen had its 10 reusable pilot checkpoints. The service
will finish Corenflos, run the other 90 Ditlevsen fits, then run final and
every-update Euler-20 evaluations, render PNG figures, and execute the audit.

Useful status command:

```sh
systemctl --user show dmop-if2warm-corrected-final100-v2.service \
  --property=ActiveState,SubState,Result,ExecMainPID
journalctl --user -u dmop-if2warm-corrected-final100-v2.service -n 50 --no-pager
```

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

## Outputs and next steps

- Corenflos fit: `corenflos/results/if2warm_ifad097_budget_j100_final_100`
- Ditlevsen fit:
  `ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100`
- Figures: `corenflos/results/if2warm_ifad097_comparison/figures`
- Invalid/earlier pilots were moved under `/tmp/dmop-if2warm-*`; the committed
  history also preserves earlier results.

When the service finishes, verify 100 checkpoints and 101-line final CSVs,
inspect all four PNGs, run `python -m corenflos.warm_audit` with the repository
PYTHONPATH, summarize paired changes from the IF2 checkpoint, then commit and
push the completed results. Do not report success before the audit passes.
