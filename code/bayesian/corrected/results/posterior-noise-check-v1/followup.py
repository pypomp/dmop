"""Run the committed GPU replay and expanded plateau-noise check sequentially."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent;code=root.parents[1]
python='/home/kevin/anaconda3/envs/pypomp/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',
    PYTHONPATH='/tmp/dmop-nuts-deps',XLA_PYTHON_CLIENT_PREALLOCATE='false')
tasks=[('replay',[python,str(code/'results/batch-continuation-check-v1/check.py'),
                 '--output','/tmp/dmop-batch-continuation-gpu-optimized-v1'],
                 '--xla_gpu_deterministic_ops=true --xla_gpu_enable_scatter_determinism_expander=true'),
       ('noise',[python,str(code/'noise_check.py'),'--inputs',str(root/'plateau-input.npz'),
                 '--output',str(root/'expanded'),'--reps','128','--batch-size','16'],'')]
state=root/'followup-state.json'
if state.exists():raise ValueError('Followup already launched; inspect its process before any recovery')
results=[]
for name,cmd,flags in tasks:
    env['XLA_FLAGS']=flags
    record=dict(task=name,command=cmd,xla_flags=flags,start_utc=datetime.now(timezone.utc).isoformat(),complete=False)
    results.append(record);state.write_text(json.dumps(results,indent=2)+'\n')
    with open(f'/tmp/dmop-bayes-followup-{name}.log','w') as log:
        proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=proc.returncode,complete=True,end_utc=datetime.now(timezone.utc).isoformat())
    state.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(record),flush=True)
