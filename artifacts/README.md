# Archived experiment outputs

The working result directories stay in place. Git tracks compact configurations,
summary tables, selected-start inputs, and figures; per-update evaluations,
checkpoints, full optimization traces, global-search pickles, and compiled
documents are stored in the verified local archive instead.
`retained-results.txt` lists the 399 deliberately retained result files.

## October 6, 2026 snapshot

- Host: `kevtan-linux` (host name `kevtan`).
- Backup directory: `/home/kevin/storage-audit/dmop-cleanup-20261006T1855Z`.
- Archive: `results-and-generated-documents.tar.gz`.
- Archive inventory: `file-inventory.json`, with original relative paths,
  byte sizes, SHA-256 hashes, and whether each file was tracked before cleanup.
- Git backup: `repository-before.bundle` preserves every local Git ref,
  including cached remote branches; `refs-before.txt` records their identities.
- `status-before.txt`, `working.patch`, and `index.patch` preserve the
  pre-cleanup state. No tracked edits were present.
- All 131,071 archived files (508,280,867 logical bytes) were extracted into
  `restore-check/` and verified against their source checksums.
- The new J=1000 run is included, including its previously untracked files.
- Exact archive, inventory, and Git-bundle hashes are recorded in
  `results-archive-20261006.json` and the backup's `SHA256SUMS`.

This is a local archive on the same host, not an off-site backup or a public
download. A fresh clone contains summaries and figures, but bulk-output audits
and some plotting commands need the archive restored. Arrange access to this
existing archive with the repository maintainer; no new storage service was used.

## Verify and restore

On `kevtan-linux`, verify the archive and preserved Git state:

```sh
cd /home/kevin/storage-audit/dmop-cleanup-20261006T1855Z
sha256sum -c SHA256SUMS
git -C /home/kevin/OneDrive/Documents/UMich/Research/pypomp/dmop bundle verify "$PWD/repository-before.bundle"
```

Extract into a new staging directory first, avoiding any overwrite of a working
checkout:

```sh
restore_dir=$(mktemp -d /tmp/dmop-restore.XXXXXX)
tar -xzf results-and-generated-documents.tar.gz -C "$restore_dir"
python3 - "$restore_dir" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
items = json.loads(pathlib.Path("file-inventory.json").read_text())
for item in items:
    path = root / item["path"]
    assert path.stat().st_size == item["bytes"], str(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], str(path)
assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == {i["path"] for i in items}
print(f"Verified {len(items)} files")
PY
```

Copy the required result directories from the verified staging directory into
the same relative locations in a checkout. Keep existing files unless an
intentional replacement is needed. The archive also contains pre-cleanup
compiled documents; their sources remain tracked. Do not replace manuscript
sources with an old Git snapshot.

To inspect the preserved Git history separately, run
`git clone repository-before.bundle /path/to/new/recovery-checkout`; no history
rewrite is needed.

## Reproduction and future runs

The manifest records the exact pre-cleanup code commit and J=1000 command.
Retained per-run `configuration.json` and `evaluation_configuration.json` files
preserve seeds, particle counts, optimizer settings, and evaluation effort.
The J=1000 `manifest.json` freezes all four arms. Existing method READMEs and
`scripts/` give the other run commands. Reproduction requires the documented
Python/JAX dependencies, sibling pypomp package, and appropriate GPU hardware.

After future runs, archive and verify bulk outputs before deliberately adding
compact results. Do not force-add entire result directories. Keep a small
archive manifest with hashes, the producing code commit, run configuration,
and storage location. To check the Overleaf file budget, run
`git ls-files | wc -l`; stay below 2,000.

The cleanup changes only tracking in this checkout: its original local results
remain unchanged. In another clone, pulling a commit that untracks files can
remove previously tracked copies. Preserve that clone's local work before
updating, then restore required bulk outputs from the archive.

Overleaf's currently connected branch is not recorded in this checkout. Sync
the cleaned `main` tree intentionally; old Overleaf branches still contain the
old bulk tree and may reintroduce it if merged wholesale. No Overleaf settings
or remote branches were changed by this cleanup.

## Clean manuscript build

The 580-file tracked tree was exported into the backup's `clean-export/` and
built using the existing TinyTeX (TeX Live 2026) toolchain:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error ms.tex si.tex
```

Both documents compiled (23 and 68 pages) without undefined references or
citations, and without reading from the original working tree. The unchanged
TeX sources still report duplicate PDF destinations and one manuscript overfull
box. Logs are in `clean-build.log` and `clean-export/{ms,si}.log` beside the archive.
