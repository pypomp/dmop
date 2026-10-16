"""Check exact GPU recovery across an uneven save during adaptive warmup.

Four short real Dhaka chains generate an uninterrupted reference. Earlier
checkpoints from different iterations form a simulated partial-save fixture.
After the reference process exits, recover the fixture and compare every
saved numerical array except elapsed times. This is not posterior sampling.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import psutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--force-mode',choices=['reverse','forward'],default='reverse')
a=p.parse_args(); a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
code=Path(__file__).resolve().parents[2];repo=code.parents[2]
original=code/'results/candidate-pmrw-4chain-v1'
python='/home/kevin/anaconda3/envs/pypomp/bin/python'
cmd=[python,str(code/'batch.py'),'--output',str(a.output/'reference'),
     '--initials',*[str(original/f'chain{c}/initial.npy') for c in range(4)],
     '--metrics',*[str(original/f'chain{c}/initial_chol.npy') for c in range(4)],
     '--seeds','50','51','52','53','--method','nuts','--particles','100',
     '--force-particles','20','--force-precision','float32','--max-depth','2',
     '--warmup','6','--draws','2','--step-size','.001','--checkpoint','1','--fixed-metric']
cmd.extend(['--force-mode',a.force_mode])
metadata=dict(utc=datetime.now(timezone.utc).isoformat(),command=cmd,
              source_sha256={name:hashlib.sha256((code/name).read_bytes()).hexdigest()
                             for name in ['batch.py','batch_extend.py','model.py','kernels.py','nuts.py','force_modes.py']},
              simulated_partial_save_iterations=[3,2,3,2],posterior_samples=False)
(a.output/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
captured={}
with (a.output/'reference.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=repo,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    created=psutil.Process(proc.pid).create_time()
    while proc.poll() is None:
        for c,wanted in enumerate([3,2,3,2]):
            f=a.output/'reference'/f'chain{c}/checkpoint.npz'
            if c not in captured and f.exists():
                payload=f.read_bytes()
                with np.load(io.BytesIO(payload)) as cp:
                    count=len(cp['trace'])
                if count==wanted:captured[c]=payload
                if count>wanted and c not in captured:
                    proc.terminate();proc.wait();raise RuntimeError('Missed checkpoint; preserve this failed check')
        time.sleep(.02)
    if proc.returncode:raise RuntimeError('Reference failed; inspect reference.log')
assert len(captured)==4
fixture=a.output/'partial-save'
for c,payload in captured.items():
    parent=a.output/'reference'/f'chain{c}';out=fixture/f'chain{c}'
    shutil.copytree(parent,out)
    (out/'completion.json').unlink()
    (out/'checkpoint.npz').write_bytes(payload)
    with np.load(io.BytesIO(payload)) as cp:
        status=json.loads((out/'status.json').read_text())
        status.update(iteration=len(cp['trace']),complete=False,phase='warmup',
                      seconds=float(cp['iteration_seconds'].sum()))
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
(fixture/'interruption.json').write_text(json.dumps(dict(verified_terminal=True,pid=proc.pid,
    create_time=created,simulated_partial_save=True,
    reason='Earlier checkpoints captured during the reference run; its process has now exited normally'),indent=2)+'\n')
resume=[python,str(code/'batch_extend.py'),'--parent',str(fixture),'--output',str(a.output/'resumed'),
        '--recover-interrupted','--checkpoint','1']
with (a.output/'resumed.log').open('w') as log:
    subprocess.run(resume,cwd=repo,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
checked={}
for c in range(4):
    with np.load(a.output/'reference'/f'chain{c}/checkpoint.npz') as ref, np.load(a.output/'resumed'/f'chain{c}/checkpoint.npz') as got:
        fields=[k for k in ref.files if k!='iteration_seconds']
        for k in fields:np.testing.assert_array_equal(ref[k],got[k],err_msg=f'chain{c}:{k}')
        checked[f'chain{c}']=fields
result=dict(passed=True,compared_fields=checked,resume_command=resume,
            state_check=json.loads((a.output/'resumed/state_check.json').read_text()),
            reference_configuration=json.loads((a.output/'reference/chain0/configuration.json').read_text()))
(a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'chains':4,'warmup':6,'iterations':8}),flush=True)
