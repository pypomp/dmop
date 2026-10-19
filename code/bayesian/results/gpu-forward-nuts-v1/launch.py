"""Launch the committed forward-force NUTS configuration once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import psutil

root=Path(__file__).resolve().parent; code=root.parents[1]; repo=code.parents[1]
protocol=json.loads((root/'protocol.json').read_text()); settings=protocol['nuts']
if (root/'launch.json').exists() or (root/'nuts').exists():
    raise ValueError('Run already exists')
assert json.loads((root.parent/'forward-force-check-v1/gpu-recovery-verification.json').read_text())['passed']
inputs=(root/protocol['inputs_dir']).resolve()
for directory,hashes in [(inputs,protocol['inputs_sha256']),(code,protocol['source_sha256'])]:
    for name,digest in hashes.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
cmd=['/home/kevin/anaconda3/envs/pypomp/bin/python',str(code/'batch.py'),
     '--output',str(root/'nuts'),'--method','nuts',
     '--initials',*[str(inputs/f'initial{c}.npy') for c in range(4)],
     '--metrics',*[str(inputs/'chol.npy')]*4,'--seeds',*map(str,settings['seeds'])]
for field in ['particles','force_particles','force_precision','force_mode','alpha','prior_scale','warmup']:
    cmd.extend(['--'+field.replace('_','-'),str(protocol[field])])
for field in ['draws','step_size','max_depth','checkpoint']:
    cmd.extend(['--'+field.replace('_','-'),str(settings[field])])
cmd.append('--fixed-metric')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(settings['gpu']),JAX_PLATFORMS='cuda',
    JAX_ENABLE_X64='true',XLA_FLAGS=protocol['xla_flags'],PYTHONPATH='/tmp/dmop-nuts-deps',
    XLA_PYTHON_CLIENT_PREALLOCATE='false')
log='/tmp/dmop-bayes-gpu-forward-nuts-v1.log'
with open(log,'w') as stream:
    child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(utc=datetime.now(timezone.utc).isoformat(),pid=child.pid,
            create_time=psutil.Process(child.pid).create_time(),command=cmd,log=log,
            gpu=settings['gpu'],revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
(root/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
