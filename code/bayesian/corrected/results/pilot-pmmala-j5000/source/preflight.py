"""Check unscaled Pypomp objectives and PF variability before sampler tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
import pypomp
from model import setup, NAMES, CENTER, SCALE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--particles', type=int, default=1000)
    p.add_argument('--reps', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = setup(a.particles)
    library = setup(a.particles, measurement='library')
    z = m['initial']
    rows = []
    for i in range(a.reps):
        key = jax.random.key(2026101500+i)
        start = time.monotonic()
        ll = float(m['loglik'](z, key))
        pf_seconds = time.monotonic()-start
        start = time.monotonic()
        value, force = m['value_force'](z, key)
        value, force = float(value), np.asarray(force)
        grad_seconds = time.monotonic()-start
        ll_library = float(library['loglik'](z, key))
        forward_error = value-float(m['logprior'](z))-ll_library
        rows.append(dict(seed=2026101500+i, loglik=ll, library_loglik=ll_library,
                         mop_pf_error=forward_error, force_finite=bool(np.isfinite(force).all()),
                         force_norm=float(np.linalg.norm(force)), pf_seconds=pf_seconds,
                         force_seconds=grad_seconds))
        print(rows[-1], flush=True)
        (a.output/'evaluations.json').write_text(json.dumps(rows, indent=2)+'\n')
    manifest = dict(particles=a.particles, reps=a.reps, names=NAMES, center=CENTER.tolist(),
                    scale=SCALE.tolist(), initial=z.tolist(), jax=jax.__version__,
                    pypomp=str(Path(pypomp.__file__).resolve()),
                    pypomp_revision=subprocess.check_output(['git','-C',str(Path(pypomp.__file__).parents[1]),
                                                              'rev-parse','HEAD'],text=True).strip(),
                    devices=[str(x) for x in jax.devices()],
                    source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in Path(__file__).parent.glob('*.py')},
                    loglik_variance=float(np.var([r['loglik'] for r in rows], ddof=1)),
                    max_mop_pf_error=max(abs(r['mop_pf_error']) for r in rows),
                    max_floor_effect=max(abs(r['loglik']-r['library_loglik']) for r in rows),
                    forces_finite=all(r['force_finite'] for r in rows))
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(manifest, flush=True)

if __name__ == '__main__':
    main()
