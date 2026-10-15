"""Run one checkpointed Pypomp Bayesian chain; outputs are never overwritten."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
import numpy as np
import pypomp
from model import setup, NAMES, CENTER, SCALE
from kernels import make_kernel, finite_force


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--particles', type=int, default=1000)
    p.add_argument('--sort-particles', action='store_true')
    p.add_argument('--alpha', type=float, default=.95)
    p.add_argument('--prior-scale', type=float, default=1.)
    p.add_argument('--method', choices=['mala', 'hmc', 'pmrw', 'pmmala', 'nuts'], default='mala')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--max-depth', type=int, default=5)
    p.add_argument('--seed', type=int, default=2026101600)
    p.add_argument('--jitter', type=float, default=.04)
    p.add_argument('--optimize', type=int, default=100)
    p.add_argument('--optimize-particles', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--step-size', type=float, default=.01)
    p.add_argument('--max-step-size', type=float, default=2.)
    p.add_argument('--checkpoint', type=int, default=25)
    p.add_argument('--metric', type=Path, help='NPY lower-triangular proposal factor')
    p.add_argument('--initial', type=Path, help='NPY vector in the standardized coordinates')
    a = p.parse_args()
    if min(a.particles,a.steps,a.draws,a.checkpoint,a.step_size,a.prior_scale) <= 0:
        p.error('Counts and scales must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.output/'source'
    source.mkdir()
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copy2(f, source/f.name)
    repo = Path(pypomp.__file__).resolve().parents[1]
    cfg = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
               jax=jax.__version__, x64=jax.config.jax_enable_x64,
               devices=[str(x) for x in jax.devices()], affinity=sorted(os.sched_getaffinity(0)),
               pypomp_revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
               pypomp_dirty=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)),
               source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')},
               names=NAMES, prior_center=CENTER.tolist(), prior_sd=(SCALE*a.prior_scale).tolist())
    if a.method == 'nuts':
        import importlib.metadata
        cfg['nuts_dependencies']={n:importlib.metadata.version(n) for n in ['blackjax','jaxopt','fastprogress']}
        if cfg['nuts_dependencies']['blackjax'] != '1.3':
            raise ValueError('This experiment pins BlackJAX 1.3')
        cfg['info_columns']=['position_changed','randomness_accepted','trajectory_acceptance',
                             'energy','randomness_log_acceptance','force_norm','divergent',
                             'integration_steps','tree_expansions']
    write_json(a.output/'configuration.json',cfg)
    m = setup(a.particles,a.alpha,a.prior_scale,sort_particles=a.sort_particles)
    warm_model = setup(a.optimize_particles,a.alpha,a.prior_scale) if a.optimize and a.optimize_particles != a.particles else m
    key = jax.random.key(a.seed)
    key, initkey = jax.random.split(key)
    z = (jnp.asarray(np.load(a.initial)) if a.initial else
         jnp.asarray(m['initial']) + a.jitter*jax.random.normal(initkey,(len(NAMES),)))
    np.save(a.output/'initial.npy',np.asarray(z))
    # Warm-start optimization changes only initialization, never the target.
    start = time.monotonic()
    moment, second = jnp.zeros_like(z),jnp.zeros_like(z)
    opt = []
    for i in range(a.optimize):
        key, k = jax.random.split(key)
        val, force = warm_model['value_force'](z,k)
        force = finite_force(force)
        moment=.9*moment+.1*force
        second=.999*second+.001*force**2
        rate=.002*(.1+.9*.5*(1+np.cos(np.pi*i/max(a.optimize,1))))
        z=z+rate*(moment/(1-.9**(i+1)))/(jnp.sqrt(second/(1-.999**(i+1)))+1e-8)
        opt.append(np.r_[float(val),np.asarray(z)])
        if (i+1)%25 == 0:
            np.save(a.output/'optimization.npy',np.asarray(opt))
            print(json.dumps({'phase':'optimization','iteration':i+1,'objective':float(val),
                              'seconds':time.monotonic()-start}),flush=True)
    np.save(a.output/'optimization.npy',np.asarray(opt))
    np.save(a.output/'warm_start.npy',np.asarray(z))
    get_force = lambda z,u: m['value_force'](z,u)[1]
    initialize, kernel = make_kernel(m['loglik'],m['logprior'],get_force,a.method,a.steps,a.max_depth)
    key, u = jax.random.split(key)
    state=initialize(z,u)
    if not np.isfinite(float(state.loglik)):
        raise RuntimeError('Initial particle filter collapsed; saved start needs revision')
    chol=jnp.eye(len(NAMES)) if a.metric is None else jnp.asarray(np.load(a.metric))
    if chol.shape != (len(NAMES),len(NAMES)) or not np.isfinite(chol).all() or np.any(np.diag(chol)<=0):
        raise ValueError('Invalid proposal metric')
    np.testing.assert_allclose(chol,np.tril(chol),atol=1e-12)
    metric_trace=float(jnp.sum(chol**2))
    np.save(a.output/'initial_chol.npy',np.asarray(chol))
    eps=a.step_size
    trace, info, eps_trace, metrics, iteration_seconds = [],[],[],[],[]
    target_accept={'mala': .57, 'hmc': .65, 'pmrw': .234, 'pmmala': .4, 'nuts': .65}[a.method]
    total=a.warmup+a.draws
    for i in range(total):
        iteration_start=time.monotonic()
        key,k=jax.random.split(key)
        state, diagnostics=kernel(state,k,eps,chol)
        diagnostics=np.asarray(diagnostics)
        iteration_seconds.append(time.monotonic()-iteration_start)
        trace.append(np.r_[float(state.loglik),float(m['logprior'](state.z)),np.asarray(state.z)])
        info.append(diagnostics)
        eps_trace.append(eps)
        if i<a.warmup:
            # Diminishing step adaptation, then freeze all tuning at warmup end.
            eps=float(np.clip(np.exp(np.log(eps)+.3/(1+i/25)**.6*(diagnostics[2]-target_accept)),1e-5,a.max_step_size))
            # Full covariance estimated from warmup, with a diagonal ridge.
            # Scale to unit mean variance so step adaptation stays continuous.
            if (i+1)%200==0 and i+1<a.warmup and i>=399:
                recent=np.array(trace)[max(0,i//2):,2:]
                covariance=np.cov(recent.T)
                covariance=.9*covariance+.1*np.diag(np.diag(covariance))+1e-5*np.eye(len(NAMES))
                covariance*=metric_trace/np.trace(covariance)
                chol=jnp.asarray(np.linalg.cholesky(covariance))
                metrics.append({'iteration':i+1,'chol':np.asarray(chol).tolist()})
        if (i+1)%a.checkpoint==0 or i+1==total:
            tmp=a.output/'checkpoint.tmp.npz'
            np.savez_compressed(tmp,trace=np.array(trace),info=np.array(info),step_size=np.array(eps_trace),
                                iteration_seconds=np.asarray(iteration_seconds),
                                z=np.asarray(state.z),u=np.asarray(jax.random.key_data(state.u)),
                                force=np.asarray(state.force),loglik=np.asarray(state.loglik),
                                key=np.asarray(jax.random.key_data(key)),chol=np.asarray(chol),next_step_size=eps)
            tmp.replace(a.output/'checkpoint.npz')
            recent=np.asarray(info)[-a.checkpoint:]
            status={'iteration':i+1,'warmup':a.warmup,'total':total,'complete':i+1==total,
                    'phase':'warmup' if i<a.warmup else 'sampling','seconds':time.monotonic()-start,
                    'loglik':float(state.loglik),'step_size':eps,
                    'parameter_acceptance_recent':float(recent[:,0].mean()),
                    'randomness_acceptance_recent':float(recent[:,1].mean())}
            write_json(a.output/'status.json',status)
            write_json(a.output/'metrics.json',metrics)
            print(json.dumps(status),flush=True)
    write_json(a.output/'completion.json',status)

if __name__=='__main__':
    main()
