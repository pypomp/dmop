# Development runs for the untempered Pypomp calculation

These are tuning and verification artifacts. They do not yet supply a final
four-chain posterior figure. Read each run's `status.json` or
`completion.json`; a directory name does not establish completion.

| Directory | Purpose |
|---|---|
| `preflight-j5000-full-x64` | Authoritative precision check: x64 enabled before importing model data; full 600-observation PF/MOP agreement and floor check |
| Other `preflight-*` | Earlier precision, particle-count and CPU/GPU timing checks; retained for provenance |
| `coupling-j1000` | Paired nearby-parameter PF estimates, comparing endpoint sorting; little variance reduction, so sorting is not selected |
| `pilot-mala-j1000` | Completed conditional-MALA pilot; tiny steps and poor auxiliary refresh; rejected as a useful configuration |
| `pilot-pmmala-j5000` | Joint particle MALA with isotropic proposals |
| `pilot-pmrw-j5000` | Completed particle random-walk MH comparison on the same target and start |
| `metric-j1000` | Failed Hessian preconditioning attempt; no output matrix used |
| `score-metric-j1000` | 64 replicated MOP scores, regularized second moment used only for proposal preconditioning |
| `pilot-pmmala-metric-j5000` | Joint particle MALA with that covariance shape |
| `pilot-hmc-metric-j5000` | Fixed-length HMC with that covariance shape and MH random-state refresh |

Chain folders contain the complete warmup/retained trace and PRNG state in
`checkpoint.npz`, a source snapshot, parameter/prior settings and acceptance
records. `report/` contains diagnostic plots and numerical gates, including
failed gates. Parent chain files stay unchanged when `extend.py` continues a
chain into a new directory. A pilot report is not promoted by selecting its
best-looking parameter or omitting a poorly mixing run.
