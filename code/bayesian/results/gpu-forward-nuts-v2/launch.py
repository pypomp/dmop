"""Launch the frozen covariance-tuned NUTS configuration."""
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
    repo = code.parents[1]
    p = json.loads((root/'protocol.json').read_text())
    inputs = root/p['inputs_dir']
    for directory, hashes in [(inputs, p['inputs_sha256']), (code, p['source_sha256'])]:
        for name, digest in hashes.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen file changed: {directory/name}')
    screen = root.parent/'nuts-metric-check-v1'
    payload = (screen/'result.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == p['screen_result_sha256']
    assert json.loads(payload)['complete']
    summary = json.loads((screen/'paired_summary.json').read_text())
    assert summary['result_sha256'] == p['screen_result_sha256']
    assert all(summary['comparisons']['covariance_2x']['protocol_screen_gates'].values())
    if (root/'nuts').exists() or (root/'launch.json').exists():
        raise ValueError('Run already exists; do not duplicate it')
    command = ['/home/kevin/anaconda3/envs/pypomp/bin/python', str(code/'batch.py'),
               '--output', str(root/'nuts'), '--method', 'nuts',
               '--initials', *[str(inputs/f'initial{c}.npy') for c in range(4)],
               '--metrics', *[str(inputs/'chol.npy')]*4,
               '--seeds', *map(str, p['nuts']['seeds'])]
    for field in ['particles', 'force_particles', 'force_precision', 'force_mode',
                  'alpha', 'prior_scale', 'warmup']:
        command.extend(['--'+field.replace('_', '-'), str(p[field])])
    for field in ['draws', 'step_size', 'max_depth', 'checkpoint']:
        command.extend(['--'+field.replace('_', '-'), str(p['nuts'][field])])
    command.append('--fixed-metric')
    if args.check_only:
        print(json.dumps(dict(inputs_source_and_screen_verified=True, command=command), indent=2))
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(p['nuts']['gpu']), JAX_PLATFORMS='cuda',
               JAX_ENABLE_X64='true', XLA_FLAGS=p['xla_flags'],
               PYTHONPATH='/tmp/dmop-nuts-deps', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    log = '/tmp/dmop-bayes-gpu-forward-nuts-v2.log'
    with open(log, 'w') as stream:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(utc=datetime.now(timezone.utc).isoformat(), pid=child.pid,
                  create_time=psutil.Process(child.pid).create_time(), command=command,
                  log=log, gpu=p['nuts']['gpu'], xla_flags=env['XLA_FLAGS'],
                  revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo,text=True).strip(),
                  protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
    (root/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
