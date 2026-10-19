# Compact result storage

Run outputs are ignored by default. Keep source, manuscript assets, small run
instructions/settings and required LaTeX table inputs as ordinary files.
Do not force-add entire result directories or duplicate source snapshots.

The original compact snapshot contains 6,985 previously tracked result files
in 195 archives. The October 19 identity-metric NUTS completion adds 158 files
in one 6.9 MB archive, including final checkpoints and comparison outputs.
`packed/manifest.json` lists all 7,143 files with their original paths, sizes,
Git blob IDs and SHA-256 hashes. The archives contain their exact bytes. Each
archive is smaller than 20 MB. All members have been independently verified.

Restore only the study needed for a reproduction, for example:

```sh
python artifacts/restore_results.py --prefix code/bayesian/corrected/results/gpu-forward-nuts-v2
python code/bayesian/corrected/paper_figures.py
```

The figure script also needs `gpu-comparison-v2`; restore that prefix first.
Restoration preserves the original directory layout and refuses to overwrite
different local files. Use `--destination /path/to/staging` to inspect a
separate restoration; `--verify-only` checks hashes without writing files.
Run without `--prefix` to restore all archived files. Files stay ignored locally.
The separate October 6 archive described in README.md still holds older bulk
outputs that were already untracked before this snapshot.

The history cleanup removes the four bulk result prefixes from past commits,
then restores the readable run guides/settings and the manuscript table at the
tip. It retains the code and manuscript history. The pre-cleanup repository,
including every local ref, is preserved and verified in
`../dmop-history-backup-20261019/repository-before.bundle` on the research host.
The backup is outside this repository and is not imported into Overleaf.

Run `python artifacts/check_file_budget.py` before committing. CI enforces a
1,800-file budget both for the current tree and for distinct paths across all
fetched history, leaving room below Overleaf's 2,000-file ceiling. New bulk
results must be verified and added to archives with a checksum inventory;
only reviewed manuscript assets should become individual tracked files.

A rewritten history changes commit IDs. Collaborators must preserve local
work and clone the compact repository afresh, or deliberately reset their
checkout to the new history. Merging an old branch back into the compact
repository would reintroduce the old paths and fail the file-budget check.
