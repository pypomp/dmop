import os
os.environ['JAX_ENABLE_X64']='true'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from pathlib import Path
import sys,json,time,hashlib
from datetime import datetime,timezone
root=Path('/home/kevin/OneDrive/Documents/UMich/Research/pypomp/dmop')
parent=root/'code/bayesian/corrected/results/candidate-nuts-4chain-v1/chain0'
sys.path.insert(0,str(parent/'source'))
import jax
import jax.numpy as jnp
import numpy as np
from model import setup
from kernels import make_kernel
out=root/'code/bayesian/corrected/results/gpu-current-config-check-v1'
out.mkdir(exist_ok=False)
import shutil
shutil.copy2(__file__,out/'benchmark.py')
cp=np.load(parent/'checkpoint.npz');cfg=json.loads((parent/'configuration.json').read_text())
np.savez_compressed(out/'inputs.npz',z=cp['z'],u=cp['u'],chol=cp['chol'],step=cp['next_step_size'])
result={'utc':datetime.now(timezone.utc).isoformat(),'devices':[str(d) for d in jax.devices()],'jax':jax.__version__,'x64':jax.config.jax_enable_x64,'particles':5000,'force_particles':1000,'alpha':.95,'max_depth':4,'prior_scale':1.,'parent':str(parent),'parent_iteration':len(cp['trace']),'parent_source_sha256':cfg['source_sha256'],'contention':'GPU0 shared with existing unrelated workloads; no process terminated.','cpu_recent_iteration_seconds':cp['iteration_seconds'][-10:].tolist(),'evaluations':[]}
assert jax.default_backend()=='gpu'
def save():
 (out/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
save();print(json.dumps({'phase':'setup','devices':result['devices']}),flush=True)
m=setup(5000,.95,1.,force_particles=1000)
init,kernel=make_kernel(m['loglik'],m['logprior'],lambda z,u:m['value_force'](z,u)[1],'nuts',4,4)
z=jnp.asarray(cp['z']);u=jax.random.wrap_key_data(jnp.asarray(cp['u']));chol=jnp.asarray(cp['chol']);eps=float(cp['next_step_size'])
t=time.monotonic();state=init(z,u);jax.block_until_ready(state);result['initialization_seconds']=time.monotonic()-t;result['cpu_gpu_loglik_difference']=float(state.loglik)-float(cp['loglik']);save();print(json.dumps({'phase':'initialized','seconds':result['initialization_seconds']}),flush=True)
for i in range(4):
 key=jax.random.key(2026103000+i);t=time.monotonic();new,info=kernel(state,key,eps,chol);jax.block_until_ready((new,info));secs=time.monotonic()-t
 row={'compile_included':i==0,'seconds':secs,'integration_steps':int(info[7]),'acceptance':float(info[2]),'divergent':bool(info[6]),'finite':bool(np.isfinite(np.asarray(new.z)).all() and np.isfinite(float(new.loglik)))}
 result['evaluations'].append(row);save();print(json.dumps(row),flush=True)
result['complete']=True;save()
