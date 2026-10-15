"""Replay interrupted adaptive runs against the actual original run.py rule."""
from pathlib import Path
from types import SimpleNamespace
import textwrap
import jax
import jax.numpy as jnp
import numpy as np
from kernels import State,make_kernel
from extend import adapt_tuning,warmup_recipe

jax.config.update('jax_enable_x64',True)


def test_warmup_and_sampling_recovery(tmp_path):
    ll=lambda z,u:-.5*z@z+.2*z[0]*jax.random.normal(u)-.02*z[0]**2
    prior=lambda z:-.5*z@z
    init,step=make_kernel(ll,prior,lambda z,u:-1.7*z,method='pmmala')
    state=init(jnp.array([.2,-.1]),jax.random.key(31));key=jax.random.key(37)
    chol=jnp.array([[.8,0.],[.1,.5]]);eps=.5
    metric_trace=float(jnp.sum(chol**2));trace=[];infos=[];used_steps=[]
    original=compile(textwrap.dedent(warmup_recipe(Path(__file__).with_name('run.py').read_text())),
                     'original_run_warmup','exec')
    metrics=[];boundaries=[399,400,600,650]
    for i in range(700):
        key,k=jax.random.split(key);state,info=step(state,k,eps,chol)
        info=np.asarray(info);infos.append(info);used_steps.append(eps)
        trace.append(np.r_[float(state.loglik),float(prior(state.z)),np.asarray(state.z)])
        env=dict(i=i,a=SimpleNamespace(warmup=650,max_step_size=2.),eps=eps,
                 diagnostics=info,target_accept=.4,np=np,jnp=jnp,trace=trace,
                 metric_trace=metric_trace,chol=chol,metrics=metrics,NAMES=range(2))
        exec(original,env)
        eps=env['eps'];chol=env['chol']
        if i+1 in boundaries:
            np.savez(tmp_path/f'{i+1}.npz',z=np.asarray(state.z),
                u=np.asarray(jax.random.key_data(state.u)),force=np.asarray(state.force),
                loglik=np.asarray(state.loglik),key=np.asarray(jax.random.key_data(key)),
                chol=np.asarray(chol),next_step_size=eps,trace=np.asarray(trace))
    expected=np.asarray(trace)
    for boundary in boundaries:
        with np.load(tmp_path/f'{boundary}.npz') as cp:
            restored=State(jnp.asarray(cp['z']),jax.random.wrap_key_data(jnp.asarray(cp['u'])),
                           jnp.asarray(cp['loglik']),jnp.asarray(cp['force']))
            restored_key=jax.random.wrap_key_data(jnp.asarray(cp['key']))
            restored_chol=jnp.asarray(cp['chol']);restored_eps=float(cp['next_step_size'])
            restored_trace=list(cp['trace'])
        for i in range(boundary,700):
            np.testing.assert_equal(restored_eps,used_steps[i])
            restored_key,k=jax.random.split(restored_key)
            restored,info=step(restored,k,restored_eps,restored_chol)
            info=np.asarray(info)
            row=np.r_[float(restored.loglik),float(prior(restored.z)),np.asarray(restored.z)]
            np.testing.assert_array_equal(row,expected[i])
            np.testing.assert_array_equal(info,infos[i])
            restored_trace.append(row)
            restored_chol,restored_eps,_=adapt_tuning(i,info,restored_trace,restored_chol,
                restored_eps,metric_trace,650,2.,'pmmala')
        np.testing.assert_array_equal(restored_chol,chol)
        np.testing.assert_equal(restored_eps,eps)
