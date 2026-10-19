"""Run fixed-tuning NUTS depths 2/3 and PMMH sequentially on GPU0.

No manuscript assets are changed. Original runs stay untouched. No polling,
retries or automatic extensions: the worker reports after all three arms exit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parents[1]
REPO = CODE.parents[2]
ARMS = [('nuts-depth2', 'nuts', 2), ('pmmh', 'pmrw', 2), ('nuts-depth3', 'nuts', 3)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(p)


def prepare():
    import numpy as np
    if (ROOT/'protocol.json').exists():
        raise ValueError('Protocol already exists; do not overwrite')
    parents = dict(nuts=CODE/'results/gpu-forward-nuts-v2/nuts',
                   pmrw=CODE/'results/gpu-comparison-v2/pmrw-long')
    inputs = ROOT/'inputs'; inputs.mkdir(exist_ok=False)
    steps, hashes, configs = {}, {}, []
    for method, parent in parents.items():
        steps[method] = []
        for c in range(4):
            p = parent/f'chain{c}'
            cfg = json.loads((p/'configuration.json').read_text()); configs.append(cfg)
            assert json.loads((p/'completion.json').read_text())['complete']
            path = p/'checkpoint.npz'; hashes[str(path.relative_to(REPO))] = sha(path)
            with np.load(path) as cp:
                assert len(cp['trace']) == cfg['warmup']+cfg['draws']
                assert np.isfinite(cp['trace']).all()
                if method == 'nuts':
                    np.save(inputs/f'initial{c}.npy', cp['z'])
                np.save(inputs/f'{method}-chol{c}.npy', cp['chol'])
                steps[method].append(float(cp['next_step_size']))
    for field in ['particles','prior_center','prior_sd','prior_scale','names','pypomp_revision']:
        assert all(cfg[field] == configs[0][field] for cfg in configs)
    assert len({c['source_sha256']['model.py'] for c in configs}) == 1
    assert sha(CODE/'model.py') == configs[0]['source_sha256']['model.py']
    source = ROOT/'source'; source.mkdir()
    for p in CODE.glob('*.py'): shutil.copy2(p, source/p.name)
    protocol = dict(utc=datetime.now(timezone.utc).isoformat(), gpu=0,
        arms=ARMS, particles=10000, force_particles=1000, force_precision='float32',
        force_mode='forward', alpha=.95, prior_scale=1., warmup=0, draws=1100,
        discard=100, retained=1000, checkpoint=25, steps=steps,
        seeds=[2026101920,2026101921,2026101922,2026101923],
        xla_flags=configs[0]['xla_flags'], pypomp_revision=configs[0]['pypomp_revision'],
        parent_checkpoint_sha256=hashes,
        source_sha256={p.name:sha(p) for p in source.glob('*.py')},
        input_sha256={p.name:sha(p) for p in inputs.iterdir()},
        initialization='Same four final primary NUTS parameter states in all arms; fresh filter randomness. Each method retains its previously tuned matrix and lane-specific step sizes.',
        timing='Sequential on GPU0 outside MPS. Exclude first100 updates from ESS and retained timing; report their cost separately. Pilot preparation excluded. Other host GPU workloads may change.',
        assessment='Report all18 bulk/tail ESS per second, Rhat, divergences and depth saturation. Small ESS or between-chain separation prevents a firm efficiency conclusion. No selection or extension from interim results.',
        manuscript='No automatic manuscript/SI text or figure changes.')
    write(ROOT/'protocol.json',protocol)
    print(json.dumps({'prepared':True,'steps':steps}))


def report():
    import arviz as az
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    protocol=json.loads((ROOT/'protocol.json').read_text())
    rows=[]; summary={}; targets=[]
    for name, method, depth in ARMS:
        draws=[]; infos=[]; times=[]
        for c in range(4):
            p=ROOT/name/f'chain{c}'
            cfg=json.loads((p/'configuration.json').read_text())
            targets.append({k:cfg[k] for k in ['particles','prior_center','prior_sd','prior_scale','pypomp_revision','names']})
            assert json.loads((p/'completion.json').read_text())['complete']
            np.testing.assert_array_equal(np.load(p/'initial.npy'),np.load(ROOT/f'inputs/initial{c}.npy'))
            with np.load(p/'checkpoint.npz') as cp:
                assert len(cp['trace'])==1100 and np.isfinite(cp['trace']).all()
                np.testing.assert_array_equal(cp['step_size'],np.full(1100,protocol['steps'][method][c]))
                np.testing.assert_array_equal(cp['chol'],np.load(ROOT/f'inputs/{method}-chol{c}.npy'))
                draws.append(np.asarray(cfg['prior_center'])+cp['trace'][100:,2:]*np.asarray(cfg['prior_sd']))
                infos.append(cp['info'][100:].copy()); times.append(cp['iteration_seconds'].copy())
        for t in times[1:]:np.testing.assert_array_equal(t,times[0])
        seconds=float(times[0][100:].sum()); values=np.stack(draws); info=np.stack(infos)
        summary[name]=dict(retained_sampling_seconds=seconds,initial100_seconds=float(times[0][:100].sum()),
                           divergence_count=int(info[:,:,6].sum()) if method=='nuts' else None,
                           mean_integration_steps=float(info[:,:,7].mean()) if method=='nuts' else None,
                           depth_limit_fraction=float((info[:,:,8]>=depth).mean()) if method=='nuts' else None)
        for j,param in enumerate(cfg['names']):
            x=values[:,:,j]; bulk=float(az.ess(x,method='bulk'));tail=float(az.ess(x,method='tail'))
            rows.append(dict(arm=name,parameter=param,rhat=float(az.rhat(x,method='rank')),
                ess_bulk=bulk,ess_tail=tail,bulk_ess_per_second=bulk/seconds,tail_ess_per_second=tail/seconds))
    assert all(t==targets[0] for t in targets)
    dest=ROOT/'report';dest.mkdir(exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(dest/'diagnostics.csv',index=False)
    baseline=df[df.arm=='pmmh'].set_index('parameter')
    fig,axes=plt.subplots(1,2,figsize=(10,6),sharey=True,layout='constrained')
    for arm,color in [('nuts-depth2','#0072B2'),('nuts-depth3','#D55E00')]:
        part=df[df.arm==arm].set_index('parameter').loc[baseline.index]
        for ax,kind in zip(axes,['bulk','tail']):
            ratio=part[f'{kind}_ess_per_second']/baseline[f'{kind}_ess_per_second']
            ax.plot(ratio,np.arange(18),'.-',label=arm,color=color)
            summary[arm][f'{kind}_median_ratio_to_pmmh']=float(ratio.median())
            summary[arm][f'{kind}_coordinates_above_pmmh']=int((ratio>1).sum())
    for ax,kind in zip(axes,['Bulk','Tail']):
        ax.axvline(1,color='.4',ls='--');ax.set_xlabel(f'{kind} ESS/second relative to PMMH')
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.grid(alpha=.2)
    axes[0].set_yticks(np.arange(18),baseline.index);axes[0].invert_yaxis();axes[1].legend()
    for ext in ['pdf','png']:fig.savefig(dest/f'ess_per_second.{ext}',dpi=180)
    plt.close(fig)
    write(dest/'summary.json',summary)
    write(ROOT/'pipeline-status.json',dict(phase='complete',utc=datetime.now(timezone.utc).isoformat(),summary=summary))


def worker():
    p=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items(): assert sha(ROOT/'source'/name)==digest
    for name,digest in p['input_sha256'].items(): assert sha(ROOT/'inputs'/name)==digest
    for name,method,depth in ARMS:
        cmd=[sys.executable,str(ROOT/'source/batch.py'),'--output',str(ROOT/name),
             '--method',method,'--initials',*[str(ROOT/f'inputs/initial{c}.npy') for c in range(4)],
             '--metrics',*[str(ROOT/f'inputs/{method}-chol{c}.npy') for c in range(4)],
             '--seeds',*map(str,p['seeds']),'--particles','10000','--force-particles','1000',
             '--force-precision','float32','--force-mode','forward','--alpha','.95',
             '--prior-scale','1','--max-depth',str(depth),'--warmup','0','--draws','1100',
             '--step-sizes',*map(str,p['steps'][method]),'--fixed-metric','--checkpoint','25']
        with (ROOT/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            write(ROOT/'pipeline-status.json',dict(phase='sampling',arm=name,pid=proc.pid,command=cmd))
            result=proc.wait()
        if result:
            write(ROOT/'pipeline-status.json',dict(phase='failed',arm=name,returncode=result))
            raise SystemExit(result)
    report()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
    if a.prepare:return prepare()
    if a.worker:return worker()
    if a.report:return report()
    import psutil
    if (ROOT/'launch.json').exists():raise ValueError('Already launched')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    runtime=CODE/'.runtime';isolation=runtime/'short-nuts-mps';isolation.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',
        XLA_FLAGS=protocol['xla_flags'],XLA_PYTHON_CLIENT_PREALLOCATE='false',CUDA_MPS_PIPE_DIRECTORY=str(isolation),
        PYTHONPATH=str(runtime/'deps'),MPLCONFIGDIR=str(runtime/'matplotlib'))
    with (runtime/'short-trajectories.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=REPO,
                              env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),utc=datetime.now(timezone.utc).isoformat(),
                revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gpu=0)
    write(ROOT/'launch.json',record);print(json.dumps(record))


if __name__=='__main__':main()
