# Development runs for the untempered Pypomp calculation

The manuscript uses the completed NUTS report in
`gpu-forward-nuts-v2/review-nuts-2000` and PMMH reports in
`gpu-comparison-v2/review-pmrw-2000` and `review-pmrw-30000`.
`paper/` contains the manuscript diagnostic tables and figure provenance.

The completed identity-metric NUTS versus diagonal-PMMH comparison is in
`identity-adapt-nuts-v1/comparison`. The shorter-trajectory study is under
`short-trajectories-v1`; its `pipeline-status.json` records progress.
See [the implementation guide](../README.md) for reproduction commands.

The remaining directories preserve development and sensitivity studies.
Read their completion records before using results; their names alone do
not establish completion. Archived configuration and protocol paths record
the original layout. The restoration tool maps those archives into this
`results/` directory without changing their contents or checksums.

| Directory | Purpose |
|---|---|
| `gpu-precision-check-v1` | Matched-point float32/float64 likelihood checks and GPU batch-throughput measurements; see its README for completed results and live checks |
| `gpu-current-config-check-v1` | Current float64 J5000/G1000 NUTS on shared GPU0; excludes compilation in three measured updates. Does not assess the original float32 configuration or batched execution |
| `preflight-j5000-full-x64` | Authoritative precision check: x64 enabled before importing model data; full 600-observation PF/MOP agreement and floor check |
| Other `preflight-*` | Earlier precision, particle-count and CPU/GPU timing checks; retained for provenance |
| `coupling-j1000` | Paired nearby-parameter PF estimates, comparing endpoint sorting; little variance reduction, so sorting is not selected |
| `pilot-mala-j1000` | Completed conditional-MALA pilot; tiny steps and poor auxiliary refresh; rejected as a useful configuration |
| `pilot-pmmala-j5000` | Joint particle MALA with isotropic proposals |
| `pilot-pmrw-j5000` | Completed particle random-walk MH comparison on the same target and start |
| `metric-j1000` | Failed Hessian preconditioning attempt; no output matrix used |
| `score-metric-j1000` | 64 replicated MOP scores, regularized second moment used only for proposal preconditioning |
| `force-count-check-v1` | 16 replicated forces at each of 1,000/2,500/5,000 particles, at the same optimized point on core8; proposal tuning only |
| `conditional-force-check-v1` | Local fixed-seed derivative check: alpha0 matches within very small resampling regions, while larger perturbations cross discontinuities |
| `pilot-pmmala-metric-j5000` | Joint particle MALA with that covariance shape |
| `pilot-hmc-metric-j5000` | Completed fixed-length HMC pilot; substantial drift, not selected for extension |
| `pilot-nuts-metric-j5000` | Completed conditional NUTS pilot using 5,000 particles for both target and force; no divergences, but all-coordinate traces still drift |
| `pilot-nuts-g1000-j5000` | Completed conditional NUTS pilot with 1,000 force particles and 5,000 target particles; 30 retained draws, no divergences, all trajectories reach depth3; records a CPU-affinity change |
| `pilot-pmmala-g1000-j5000` | Completed joint particle MALA pilot with 1,000 force particles and 5,000 target particles; substantial drift remains |
| `pilot-nuts-depth4-g1000-j5000` | Follow-up with maximum depth4, under the committed `nuts-depth4-protocol.json` |
| `timing-pmmala-*-core8` | Short exact continuations of completed pilots for serial same-core timing; parent draws are duplicated and must not be treated as independent samples |
| `candidate-pmmala-4chain-v1` | Four independent optimization/warmup runs, deliberately stopped during warmup after the controlled timing comparison; termination records retained |
| `candidate-pmmala-g1000-4chain-v2` | Stopped development MALA candidates after the user clarified NUTS versus PMMH; exact saved v1 warmup states, 1,000 force/5,000 target particles, fresh warmup and seeds; protocol committed before launch |
| `candidate-nuts-4chain-v1` | Primary four-chain NUTS comparison, matched to the PMMH starts and initial metrics; read protocol and status for progress |
| `prior-sd2-recovery-v1` | Exact recovery of two interrupted MALA warmups; subsequently stopped with the other MALA development runs |
| `candidate-pmrw-4chain-v1` | Four matching random-walk chains interrupted at iteration1,075; immutable parents of the recoveries below |
| `sampling-recovery-v1` | Exact detached continuations of the four PMMH chains and depth4 NUTS pilot; launch records identify live processes. Development reports also test the method-comparison display |
| `prior-sd2-pmmala-4chain-v1` | Stopped MALA development sensitivity chains with twice the reference-prior SD; sensitivity remains to be checked for the final NUTS result |

See [timing.md](timing.md) before comparing elapsed times. The concurrent
candidates use different core classes and cannot establish a speedup.

Chain folders contain the complete warmup/retained trace and PRNG state in
`checkpoint.npz`, a source snapshot, parameter/prior settings and acceptance
records. `report/` contains diagnostic plots and numerical gates, including
failed gates. Parent chain files stay unchanged when `extend.py` continues a
chain into a new directory. A pilot report is not promoted by selecting its
best-looking parameter or omitting a poorly mixing run.
