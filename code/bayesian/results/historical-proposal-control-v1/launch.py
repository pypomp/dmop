"""Launch the fixed historical-proposal control once, using current Pypomp."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import psutil

root = Path(__file__).resolve().parent
code = root.parents[1]
repo = code.parents[1]
p = json.loads((root/'protocol.json').read_text())
if (root/'launch.json').exists() or (root/'pmrw').exists():
    raise ValueError('Control already launched')
for name, digest in p['source_sha256'].items():
    assert hashlib.sha256((code/name).read_bytes()).hexdigest() == digest
for name, digest in p['inputs_sha256'].items():
    assert hashlib.sha256((root/'inputs'/name).read_bytes()).hexdigest() == digest
cmd = ['/home/kevin/anaconda3/envs/pypomp/bin/python', str(code/'batch.py'),
       '--output', str(root/'pmrw'), '--method', 'pmrw',
       '--initials', *[str(root/f'inputs/initial{c}.npy') for c in range(4)],
       '--metrics', *[str(root/'inputs/chol.npy')]*4,
       '--seeds', *map(str, p['seeds']), '--particles', str(p['particles']),
       '--force-particles', '1000', '--force-precision', 'float32',
       '--alpha', '.95', '--prior-scale', str(p['prior_scale']),
       '--warmup', str(p['warmup']), '--draws', str(p['draws']),
       '--step-size', str(p['step_size']), '--checkpoint', str(p['checkpoint']),
       '--fixed-metric']
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=str(p['gpu']), JAX_PLATFORMS='cuda',
           JAX_ENABLE_X64='true', XLA_FLAGS=p['xla_flags'],
           XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONPATH='/tmp/dmop-nuts-deps')
log = '/tmp/dmop-historical-proposal-control-v1.log'
with open(log, 'w') as f:
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=f,
                            stderr=subprocess.STDOUT, start_new_session=True)
record = dict(utc=datetime.now(timezone.utc).isoformat(), pid=proc.pid,
              create_time=psutil.Process(proc.pid).create_time(), command=cmd,
              log=log, gpu=p['gpu'],
              revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
(root/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record), flush=True)
