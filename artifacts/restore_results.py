"""Verify and restore selected archived results without overwriting local work.

Run from any directory: python artifacts/restore_results.py --prefix
code/bayesian/corrected/results/gpu-forward-nuts-v2
Omit --prefix to restore all archived results. Restored outputs are ignored.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT/'artifacts/packed'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', action='append', default=[])
    p.add_argument('--destination', type=Path, default=ROOT)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    manifest = json.loads((PACKED/'manifest.json').read_text())
    chosen = {r['path']: r for r in manifest['files']
              if not a.prefix or any(r['path'].startswith(x.rstrip('/')+'/') or r['path']==x.rstrip('/') for x in a.prefix)}
    if not chosen:
        p.error('No archived files match the prefix')
    archives = {r['archive'] for r in chosen.values()}
    dest = a.destination.resolve()
    # Check existing files before writing anything. Refuse to overwrite differences.
    for path, record in chosen.items():
        rel = PurePosixPath(path)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe archive path')
        output = dest.joinpath(*rel.parts)
        if not output.resolve().is_relative_to(dest) or output.is_symlink():
            raise ValueError(f'Unsafe destination: {output}')
        if not a.verify_only and output.exists() and sha(output)!=record['sha256']:
            raise ValueError(f'Existing file differs; restore into a new --destination: {output}')
    seen = set()
    for archive in manifest['archives']:
        if archive['path'] not in archives:
            continue
        path = PACKED/archive['path']
        if sha(path)!=archive['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path}')
        with tarfile.open(path, 'r:gz') as tar:
            for member in tar:
                if member.name not in chosen:
                    continue
                record = chosen[member.name]
                if not member.isfile() or record['archive'] != archive['path'] or member.name in seen:
                    raise ValueError('Unexpected archive member')
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest()!=record['sha256'] or len(data)!=record['bytes']:
                    raise ValueError(f'Member checksum mismatch: {member.name}')
                seen.add(member.name)
                output = dest/member.name
                if a.verify_only or output.exists():
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
                    f.write(data); temporary=Path(f.name)
                temporary.chmod(int(record['mode'], 8)&0o777)
                # Atomic no-clobber install also protects a concurrently created output.
                try:
                    os.link(temporary, output)
                finally:
                    temporary.unlink()
    if seen != set(chosen):
        raise ValueError('Missing archive members')
    print(f'Verified {len(seen)} files in {len(archives)} archives' + ('' if a.verify_only else f'; restored to {dest}'))


if __name__ == '__main__':
    main()
