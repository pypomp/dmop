"""Launch the frozen wider-prior comparison after J5000 PMMH finishes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    code = root.parents[1]
    repo = code.parents[2]
    protocol = json.loads((root/'protocol.json').read_text())
    inputs = (root/protocol['inputs_dir']).resolve()
    for directory, hashes in [(inputs, protocol['inputs_sha256']),
                              (code, protocol['source_sha256'])]:
        for name, digest in hashes.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen file changed: {directory/name}')
    if (root/'pmrw').exists() or (root/'launch.json').exists():
        raise ValueError('Run already exists')
    prior_run = (root/protocol['launch_after']).resolve()
    complete = all((prior_run/f'chain{c}'/'completion.json').exists()
                   and json.loads((prior_run/f'chain{c}'/'completion.json').read_text())['complete']
                   for c in range(4))
    launch = json.loads((prior_run.parent/'pmrw-continuation-launch.json').read_text())
    try:
        process = psutil.Process(launch['pid'])
        alive = (abs(process.create_time()-launch['create_time']) < .1
                 and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        alive = False
    ready = complete and not alive
    command = ['/home/kevin/anaconda3/envs/pypomp/bin/python', str(code/'batch.py'),
               '--output', str(root/'pmrw'), '--method', 'pmrw',
               '--initials', *[str(inputs/f'initial{c}.npy') for c in range(4)],
               '--metrics', *[str(inputs/'chol.npy')]*4,
               '--seeds', *map(str, protocol['seeds'])]
    for field in ['particles', 'force_particles', 'force_precision', 'alpha',
                  'prior_scale', 'warmup', 'draws', 'step_size', 'max_depth', 'checkpoint']:
        command.extend(['--'+field.replace('_', '-'), str(protocol[field])])
    command.append('--fixed-metric')
    if args.check_only:
        print(json.dumps(dict(inputs_and_source_verified=True, ready=ready,
                              prerequisite_complete=complete,
                              prerequisite_process_live=alive, command=command), indent=2))
        return
    if not ready:
        raise RuntimeError('Wait for the J5000 run to complete and its host process to exit')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(protocol['gpu']), JAX_PLATFORMS='cuda',
               JAX_ENABLE_X64='true', XLA_FLAGS=protocol['xla_flags'],
               PYTHONPATH='/tmp/dmop-nuts-deps', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    log = '/tmp/dmop-bayes-prior-sensitivity-gpu-v1.log'
    with open(log, 'w') as stream:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(utc=datetime.now(timezone.utc).isoformat(), pid=child.pid,
                  create_time=psutil.Process(child.pid).create_time(), command=command,
                  log=log, gpu=protocol['gpu'], xla_flags=env['XLA_FLAGS'],
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],
                                                   cwd=repo,text=True).strip())
    (root/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
