"""Compare forward and reverse differentiation of the unchanged MOP force.

The candidate is an isolated copy of model.py with only the differentiation
operator changed. Production source and running chains are never modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force-particles', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--reps', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 4 or min(args.force_particles, args.reps) < 1:
        parser.error('Positive counts required; batch size must be1--4')
    root = Path(__file__).resolve().parent
    protocol = json.loads((root/'protocol.json').read_text())
    source = root.parents[1]/'model.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol['model_sha256']:
        raise ValueError('Frozen model source changed')
    inputs = root.parent/'gpu-comparison-v1'/'inputs'
    for name, digest in protocol['inputs_sha256'].items():
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen input changed')
    args.output.mkdir(parents=True, exist_ok=False)
    original = source.read_text()
    old = '                value, force = jax.value_and_grad(target32)(z.astype(jnp.float32), key)'
    new = '''                def with_aux(x):
                    value = target32(x, key)
                    return value, value
                force, value = jax.jacfwd(with_aux, has_aux=True)(z.astype(jnp.float32))'''
    if original.count(old) != 1:
        raise ValueError('Expected exactly one mixed-precision differentiation site')
    (args.output/'reverse_model.py').write_text(original)
    (args.output/'forward_model.py').write_text(original.replace(old, new))
    reverse = load_module(args.output/'reverse_model.py', 'reverse_probe')
    forward = load_module(args.output/'forward_model.py', 'forward_probe')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pypomp
    if not jax.config.jax_enable_x64:
        raise ValueError('Acceptance target must remain float64')
    zs = jnp.asarray(np.stack([np.load(inputs/f'initial{i}.npy')
                              for i in range(args.batch_size)]), dtype=jnp.float64)
    keys = jnp.stack([jax.random.key(2026110900+i) for i in range(args.batch_size)])
    result = dict(utc=datetime.now(timezone.utc).isoformat(), devices=list(map(str,jax.devices())),
                  jax=jax.__version__, xla_flags=os.environ.get('XLA_FLAGS',''),
                  settings=vars(args)|{'output':str(args.output)},
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [source,args.output/'forward_model.py',Path(__file__)]},
                  pypomp_revision=subprocess.check_output(['git','-C',str(Path(pypomp.__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),
                  timings={}, complete=False)
    def save():
        temp = args.output/'result.tmp'
        temp.write_text(json.dumps(result,indent=2)+'\n')
        temp.replace(args.output/'result.json')
    save()
    outputs = {}
    for name, module in [('reverse',reverse), ('forward',forward)]:
        model = module.setup(10000,.95,1.,force_particles=args.force_particles,force_precision='float32')
        evaluate = jax.jit(jax.vmap(model['value_force']))
        times, values, forces = [], [], []
        for rep in range(args.reps+1):
            start = time.monotonic()
            value, force = evaluate(zs,keys)
            jax.block_until_ready((value,force))
            elapsed = time.monotonic()-start
            times.append(dict(seconds=elapsed,compile_included=rep==0))
            values.append(np.asarray(value)); forces.append(np.asarray(force))
            if not np.isfinite(values[-1]).all() or not np.isfinite(forces[-1]).all():
                raise ValueError('Nonfinite force evaluation')
            result['timings'][name] = times; save()
            print(json.dumps(dict(mode=name,rep=rep,seconds=elapsed)),flush=True)
        outputs[name] = (np.stack(values),np.stack(forces))
    np.savez_compressed(args.output/'evaluations.npz',z=np.asarray(zs),
                        **{name+'_'+field:arrays[i] for name,arrays in outputs.items()
                           for i,field in enumerate(['value','force'])})
    rv,rf = outputs['reverse']; fv,ff = outputs['forward']
    relative = float(np.max(np.abs(ff[0]-rf[0]))/(1+np.max(np.abs(rf[0]))))
    result['comparison'] = dict(same_value=bool(np.array_equal(rv[0],fv[0])),
                               max_absolute_force_difference=float(np.max(np.abs(ff[0]-rf[0]))),
                               scaled_max_force_difference=relative,
                               repeat_bitwise={name:bool(np.all(v==v[0]) and np.all(f==f[0]))
                                               for name,(v,f) in outputs.items()},
                               force_tolerance=protocol['force_tolerance'],
                               passed=bool(np.array_equal(rv[0],fv[0]) and relative<=protocol['force_tolerance']))
    result['complete']=True; save()
    print(json.dumps(result['comparison']),flush=True)


if __name__=='__main__':
    main()
