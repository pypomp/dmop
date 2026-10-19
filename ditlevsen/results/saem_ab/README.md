# Superseded unscaled SAEM experiment

This directory preserves the original Dhaka score/SAEM experiment. Its
numerical M-step could report L-BFGS-B convergence without moving. Fixed-path
replays and a regression test identified the problem; the corrected study
is in [../saem_ab_scaled/](../saem_ab_scaled/README.md).

All four original workers have stopped. Cold starts 0--7 have complete paired
fits and evaluations. Cold starts 8--11 have completed score fits and may
also contain interrupted SAEM checkpoints. None of those incomplete pairs
is marked complete. No warm-start pair was fitted here.

All 12 completed score fits, including failures, are eligible for reuse in
the corrected study under the matching protocol. Their metadata, parameters
and fitting times remain unchanged. The old SAEM fits are preserved for
diagnosis and must not be pooled with the corrected experiment. The frozen
`protocol.json` and `source/` describe this original implementation; current
workers intentionally reject its source hashes when run against corrected
code.
