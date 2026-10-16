"""Continue a chain, preserving its tuning history and RNG state.

Write a new directory containing the complete old trace plus new draws. The
parent remains immutable. Completed chains retain frozen tuning. Explicit
interruption recovery resumes the original warmup schedule when necessary.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

WARMUP_RECIPE_SHA256='babfc71a989e5164e1ef5f64901df1d31e6169ca9fc7f363b53ccdec8f128f7a'


def warmup_recipe(source):
    start=source.index('        if i<a.warmup:')
    end=source.index('        if (i+1)%a.checkpoint',start)
    block=source[start:end]
    if hashlib.sha256(block.encode()).hexdigest()!=WARMUP_RECIPE_SHA256:
        raise ValueError('Unrecognized parent warmup rule; audit it before recovery')
    return block


def adapt_tuning(i, info, trace, chol, eps, metric_trace, warmup, max_step, method):
    """The saved run.py warmup rule, applied at its original iteration index."""
    import numpy as np
    import jax.numpy as jnp
    record=None
    if i<warmup:
        target={'mala':.57,'hmc':.65,'pmrw':.234,'pmmala':.4,'nuts':.65}[method]
        eps=float(np.clip(np.exp(np.log(eps)+.3/(1+i/25)**.6*(info[2]-target)),1e-5,max_step))
        if (i+1)%200==0 and i+1<warmup and i>=399:
            recent=np.array(trace)[max(0,i//2):,2:]
            covariance=np.cov(recent.T)
            covariance=.9*covariance+.1*np.diag(np.diag(covariance))+1e-5*np.eye(chol.shape[0])
            covariance*=metric_trace/np.trace(covariance)
            chol=jnp.asarray(np.linalg.cholesky(covariance))
            record={'iteration':i+1,'chol':np.asarray(chol).tolist()}
    return chol,eps,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--draws',type=int)
    p.add_argument('--recover-interrupted',action='store_true',
                   help='Resume the original run after a verified terminal interruption')
    p.add_argument('--checkpoint',type=int,default=25)
    a=p.parse_args()
    if (a.draws is not None and a.draws<1) or a.checkpoint<1:
        p.error('Positive counts required')
    if a.recover_interrupted:
        if a.draws is not None or (a.parent/'completion.json').exists():
            p.error('Recovery preserves the original run length and requires an incomplete parent')
        audit=a.parent/'interruption.json'
        if not audit.exists() or not json.loads(audit.read_text()).get('verified_terminal'):
            p.error('Record the verified terminal process in interruption.json before recovery')
    elif a.draws is None or not (a.parent/'completion.json').exists():
        p.error('Parent must be completed, with adaptation frozen')
    cfg=json.loads((a.parent/'configuration.json').read_text())
    source=a.parent/'source'
    for name,digest in cfg['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Changed parent source: {name}')
    # All model/kernel imports come from the parent source snapshot.
    sys.path.insert(0,str(source.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pypomp
    from model import setup
    from kernels import State,make_kernel,finite_force
    from run import write_json
    repo=Path(pypomp.__file__).resolve().parents[1]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if revision!=cfg['pypomp_revision'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True):
        raise ValueError('Pypomp source differs from parent')
    if [str(x) for x in jax.devices()]!=cfg['devices']:
        raise ValueError('Continue on the same backend as the parent')
    if jax.__version__!=cfg['jax'] or bool(jax.config.jax_enable_x64)!=cfg['x64']:
        raise ValueError('JAX version or precision differs from parent')
    if cfg.get('nuts_dependencies'):
        import importlib.metadata
        for name,version in cfg['nuts_dependencies'].items():
            if importlib.metadata.version(name)!=version:
                raise ValueError(f'Changed NUTS dependency: {name}')
    cp=np.load(a.parent/'checkpoint.npz')
    old=len(cp['trace'])
    total=cfg['warmup']+cfg['draws'] if a.recover_interrupted else old+a.draws
    if a.recover_interrupted and not 0<old<total:
        raise ValueError('Recovery requires a nonempty incomplete checkpoint')
    if not a.recover_interrupted and old!=cfg['warmup']+cfg['draws']:
        raise ValueError('Parent checkpoint does not contain its completed run')
    if not a.recover_interrupted and old<=cfg['warmup']:
        raise ValueError('No retained sampling in parent')
    state=State(jnp.asarray(cp['z']),jax.random.wrap_key_data(jnp.asarray(cp['u'])),
                jnp.asarray(cp['loglik']),jnp.asarray(cp['force']))
    key=jax.random.wrap_key_data(jnp.asarray(cp['key']))
    chol=jnp.asarray(cp['chol']);eps=float(cp['next_step_size'])
    traces=list(cp['trace']);infos=list(cp['info']);eps_trace=list(cp['step_size'])
    iteration_seconds=list(cp['iteration_seconds']) if 'iteration_seconds' in cp else [np.nan]*old
    metrics=json.loads((a.parent/'metrics.json').read_text()) if (a.parent/'metrics.json').exists() else []
    metrics=[item for item in metrics if item['iteration']<=old]
    metric_trace=None
    if old<cfg['warmup']:
        warmup_recipe((source/'run.py').read_text())
        initial_chol=jnp.asarray(np.load(a.parent/'initial_chol.npy'))
        metric_trace=float(jnp.sum(initial_chol**2))
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source,a.output/'source')
    for name in ['initial.npy','initial_chol.npy','warm_start.npy','optimization.npy']:
        if (a.parent/name).exists():
            shutil.copy2(a.parent/name,a.output/name)
    cfg.update(output=str(a.output),draws=total-cfg['warmup'],parent=str(a.parent.resolve()),
               parent_checkpoint_sha256=hashlib.sha256((a.parent/'checkpoint.npz').read_bytes()).hexdigest(),
               continuation_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
               continuation_affinity=sorted(os.sched_getaffinity(0)),
               recovery=a.recover_interrupted,
               continuation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    shutil.copy2(Path(__file__),a.output/'continuation_driver.py')
    write_json(a.output/'configuration.json',cfg)
    model_extra={'force_particles':cfg['force_particles']} if cfg.get('force_particles') is not None else {}
    m=setup(cfg['particles'],cfg['alpha'],cfg['prior_scale'],sort_particles=cfg.get('sort_particles',False),**model_extra)
    extra={'max_depth':cfg['max_depth']} if cfg['method']=='nuts' else {}
    _,kernel=make_kernel(m['loglik'],m['logprior'],lambda z,u:m['value_force'](z,u)[1],cfg['method'],cfg['steps'],**extra)
    if a.recover_interrupted:
        checked_loglik=float(m['loglik'](state.z,state.u))
        checked_force=np.asarray(finite_force(m['value_force'](state.z,state.u)[1]))
        np.testing.assert_allclose(checked_loglik,float(state.loglik),rtol=0,atol=1e-9)
        np.testing.assert_allclose(checked_force,np.asarray(state.force),rtol=1e-10,atol=1e-8)
        write_json(a.output/'recovery_state_check.json',dict(
            likelihood_difference=checked_loglik-float(state.loglik),
            maximum_force_difference=float(np.max(np.abs(checked_force-np.asarray(state.force)))),
            parent_iteration=old,original_total=total))
    start=time.monotonic()
    parent_seconds=json.loads((a.parent/'status.json').read_text())['seconds']
    for i in range(old,total):
        iteration_start=time.monotonic()
        key,k=jax.random.split(key)
        state,info=kernel(state,k,eps,chol)
        infos.append(np.asarray(info))
        iteration_seconds.append(time.monotonic()-iteration_start)
        traces.append(np.r_[float(state.loglik),float(m['logprior'](state.z)),np.asarray(state.z)])
        eps_trace.append(eps)
        if i<cfg['warmup']:
            chol,eps,metric=adapt_tuning(i,infos[-1],traces,chol,eps,metric_trace,
                                       cfg['warmup'],cfg['max_step_size'],cfg['method'])
            if metric is not None:
                metrics.append(metric)
        if (i+1)%a.checkpoint==0 or i+1==total:
            tmp=a.output/'checkpoint.tmp.npz'
            np.savez_compressed(tmp,trace=np.array(traces),info=np.array(infos),step_size=np.array(eps_trace),
                iteration_seconds=np.asarray(iteration_seconds),
                z=np.asarray(state.z),u=np.asarray(jax.random.key_data(state.u)),force=np.asarray(state.force),
                loglik=np.asarray(state.loglik),key=np.asarray(jax.random.key_data(key)),
                chol=np.asarray(chol),next_step_size=eps)
            tmp.replace(a.output/'checkpoint.npz')
            recent=np.asarray(infos)[-a.checkpoint:]
            status=dict(iteration=i+1,warmup=cfg['warmup'],total=total,complete=i+1==total,
                        phase='warmup' if i<cfg['warmup'] else 'sampling',seconds=parent_seconds+time.monotonic()-start,
                        loglik=float(state.loglik),step_size=eps,
                        parameter_acceptance_recent=float(recent[:,0].mean()),
                        randomness_acceptance_recent=float(recent[:,1].mean()))
            write_json(a.output/'status.json',status)
            write_json(a.output/'metrics.json',metrics)
            print(json.dumps(status),flush=True)
    write_json(a.output/'completion.json',status)

if __name__=='__main__':
    main()
