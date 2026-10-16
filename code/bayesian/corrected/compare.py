"""Compare frozen NUTS and PMMH reports on the same target.

Run report.py for each method first. This reads its saved arrays, not changing
live checkpoints. Differences in posterior means are shown relative to their
combined Monte Carlo standard errors; this is descriptive, not an equivalence
test or a replacement for independent-chain convergence checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from report import NAMES,LABELS,style


def load_report(folder, method):
    audit=json.loads((folder/'audit.json').read_text())
    if audit['method']!=method:
        raise ValueError(f'Expected {method}, found {audit["method"]}')
    payload=folder/'posterior_draws.npz'
    if hashlib.sha256(payload.read_bytes()).hexdigest()!=audit['draws_sha256']:
        raise ValueError('Report draws have changed')
    draws=np.load(payload)['draws']
    if draws.shape!=(len(audit['chains']),audit['retained'],len(NAMES)):
        raise ValueError('Report dimensions disagree with audit')
    table=pd.read_csv(folder/'diagnostics.csv').set_index('parameter').loc[list(NAMES)]
    # Detect a stale or unrelated table before interpreting its MCSE values.
    np.testing.assert_allclose(table['mean'],draws.mean(axis=(0,1)),rtol=1e-10,atol=1e-10)
    return audit,draws,table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nuts',required=True,type=Path,help='Frozen report directory')
    p.add_argument('--pmmh',required=True,type=Path,help='Frozen report directory')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    na,nd,nt=load_report(a.nuts,'nuts');pa,pd_,pt=load_report(a.pmmh,'pmrw')
    if na['target']!=pa['target'] or na['model_sha256']!=pa['model_sha256']:
        raise ValueError('Reports do not have the same model, likelihood particles and prior')
    a.output.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(index=NAMES)
    for label,t in [('nuts',nt),('pmmh',pt)]:
        for field in ['mean','sd','q025','q975','rhat','ess_bulk','ess_tail','mcse_mean']:
            table[f'{label}_{field}']=t[field].to_numpy()
    table['mean_difference']=nt['mean'].to_numpy()-pt['mean'].to_numpy()
    table['combined_mcse']=np.hypot(nt['mcse_mean'].to_numpy(),pt['mcse_mean'].to_numpy())
    table['difference_over_mcse']=table['mean_difference']/table['combined_mcse']
    table.index.name='parameter';table.to_csv(a.output/'comparison.csv')
    adequate=na['numerical_convergence'] and pa['numerical_convergence']
    audit={'nuts_report':str(a.nuts),'pmmh_report':str(a.pmmh),
           'report_hashes':{str(folder/'audit.json'):hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest()
                            for folder in [a.nuts,a.pmmh]},
           'same_target':True,'both_pass_numerical_checks':bool(adequate),
           'interpretation':'Mean differences divided by combined MCSE are descriptive. Unconverged-chain MCSE can be misleading. No equivalence or speedup is inferred.',
           'remaining_checks':['visual review','prior sensitivity','particle sensitivity','same-hardware timing']}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'savefig.dpi':180})
    fig,axes=plt.subplots(6,3,figsize=(11,15),layout='constrained')
    colors=['#0072B2','#D55E00']
    fig.legend(handles=[Line2D([],[],color=c,label=label) for c,label in zip(colors,['NUTS','PMMH'])]
               +[Line2D([],[],color='black',ls='--',label='Prior')],
               loc='outside upper center',ncol=3,frameon=False,
               title=None if adequate else 'Development samples: convergence checks have not passed')
    fig.supylabel('Density')
    for j,ax in enumerate(axes.flat):
        x=np.concatenate([nd[:,:,j].ravel(),pd_[:,:,j].ravel()]);lo,hi=x.min(),x.max()
        span=max(hi-lo,1e-8);grid=np.linspace(lo-.12*span,hi+.12*span,400)
        for draws,color in zip([nd,pd_],colors):
            # Show separate chains to expose disagreement that pooling hides.
            for chain in draws[:,:,j]:
                if np.std(chain)>1e-10:
                    ax.plot(grid,gaussian_kde(chain)(grid),color=color,lw=.8,alpha=.65)
                else:
                    ax.axvline(chain[0],color=color,lw=.8,alpha=.65)
        target=na['target']
        ax.plot(grid,norm.pdf(grid,target['prior_center'][j],target['prior_sd'][j]),'k--',lw=1)
        ax.set_title(LABELS[j]);style(ax)
    fig.savefig(a.output/'method_marginals.png');plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
