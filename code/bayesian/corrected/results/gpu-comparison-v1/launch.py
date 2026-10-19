"""Launch one committed GPU comparison batch; never overwrite a run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import psutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('method',choices=['nuts','pmrw'])
a=p.parse_args()
root=Path(__file__).resolve().parent
repo=root.parents[4]
protocol=json.loads((root/'protocol.json').read_text())
settings=protocol[a.method]
if (root/f'{a.method}-launch.json').exists() or (root/a.method).exists():
    raise ValueError('Run already launched or output exists')
check=root.parent/'gpu-precision-check-v1/mixed-force-batch4/result.json'
checked=json.loads(check.read_text())
assert checked['complete'] and all(x['finite'] and not any(x['divergences']) for x in checked['kernel_timings'])
inputs=root/'inputs'
cmd=['/home/kevin/anaconda3/envs/pypomp/bin/python',str(root.parents[1]/'batch.py'),
     '--output',str(root/a.method),'--method',a.method,
     '--initials',*[str(inputs/f'initial{c}.npy') for c in range(4)],
     '--metrics',*[str(inputs/'chol.npy')]*4,
     '--seeds',*[str(s) for s in settings['seeds']],
     '--particles',str(protocol['particles']),'--force-particles',str(protocol['force_particles']),
     '--force-precision',protocol['force_precision'],'--alpha',str(protocol['alpha']),
     '--prior-scale',str(protocol['prior_scale']),'--warmup',str(protocol['warmup']),
     '--draws',str(settings['draws']),'--step-size',str(settings['step_size']),
     '--max-depth',str(settings['max_depth']),'--checkpoint','1','--fixed-metric']
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=str(settings['gpu']),JAX_PLATFORMS='cuda',
           JAX_ENABLE_X64='true',PYTHONPATH='/tmp/dmop-nuts-deps',XLA_PYTHON_CLIENT_PREALLOCATE='false')
log=f'/tmp/dmop-bayes-gpu-comparison-v1-{a.method}.log'
with open(log,'w') as stream:
    proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(utc=datetime.now(timezone.utc).isoformat(),pid=proc.pid,
            create_time=psutil.Process(proc.pid).create_time(),command=cmd,log=log,
            gpu=settings['gpu'],revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
(root/f'{a.method}-launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
