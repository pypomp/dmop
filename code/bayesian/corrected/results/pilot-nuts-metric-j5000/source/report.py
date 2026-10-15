"""Diagnose independent corrected chains and redraw the existing panel layout."""
import argparse
import hashlib
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

NAMES = ('gamma','m','epsilon','beta_trend','sigma','tau',
         *(f'bs{i}' for i in range(1,7)),*(f'omegas{i}' for i in range(1,7)))
LABELS = [r'$\log\gamma$',r'$\log m$',r'$\log\epsilon$',r'$100\beta_{\mathrm{trend}}$',
          r'$\log\sigma$',r'$\log\tau$']+[rf'$\beta_{i}$' for i in range(1,7)]+[rf'$\omega_{i}$' for i in range(1,7)]
COLORS = ['#00b9e8','#d0ca33','#fb65b3','#62dcb5']


def acf(x, maxlag):
    x=x-x.mean()
    n=len(x)
    f=np.fft.rfft(x,n=2*n)
    cov=np.fft.irfft(f*f.conj())[:maxlag+1]
    return cov/cov[0] if cov[0]>0 else np.r_[1.,np.full(maxlag,np.nan)]


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.25,ls='--')
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(style="sci", scilimits=(-3,4), useOffset=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('chains',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    configs=[json.loads((d/'configuration.json').read_text()) for d in a.chains]
    for field in ['particles','prior_scale','alpha','method','names','prior_center','prior_sd','pypomp_revision','warmup']:
        if any(c[field]!=configs[0][field] for c in configs):
            raise ValueError(f'Incompatible chains: {field}')
    if len(set(c['seed'] for c in configs)) != len(configs):
        raise ValueError('Repeated chain seeds')
    arrays=[np.load(d/'checkpoint.npz') for d in a.chains]
    warm=configs[0]['warmup']
    n=min(len(x['trace']) for x in arrays)
    if n-warm<20:
        raise ValueError('Not enough retained draws yet')
    raw=np.stack([x['trace'][:n,2:] for x in arrays])
    center=np.array(configs[0]['prior_center'])
    scale=np.array(configs[0]['prior_sd'])/configs[0]['prior_scale']
    full=center+raw*scale
    draws=full[:,warm:]
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for j,name in enumerate(NAMES):
        x=draws[:,:,j]
        rows.append(dict(parameter=name,mean=float(x.mean()),sd=float(x.std()),
            q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)),
            rhat=float(az.rhat(x,method='rank')) if len(x)>1 else float('nan'),
            ess_bulk=float(az.ess(x,method='bulk')),ess_tail=float(az.ess(x,method='tail')),
            mcse_mean=float(az.mcse(x,method='mean'))))
    table=pd.DataFrame(rows)
    table.to_csv(a.output/'diagnostics.csv',index=False)
    complete=all((d/'completion.json').exists() for d in a.chains)
    gates=dict(four_chains=len(arrays)>=4,complete=complete,
               rhat=bool(np.isfinite(table.rhat).all() and (table.rhat<=1.01).all()),
               ess=bool((table[['ess_bulk','ess_tail']]>=400).all().all()))
    audit=dict(chains=[str(d) for d in a.chains],warmup=warm,retained=n-warm,gates=gates,
               numerical_convergence=all(gates.values()),
               remaining_checks=['visual inspection','prior sensitivity','particle sensitivity'],
               source_sha256={str(d/'checkpoint.npz'):hashlib.sha256((d/'checkpoint.npz').read_bytes()).hexdigest()
                              for d in a.chains},
               acceptance=[dict(parameter=float(x['info'][warm:n,0].mean()),
                                randomness=float(x['info'][warm:n,1].mean())) for x in arrays])
    if configs[0]['method']=='nuts':
        audit['nuts']=[dict(divergences=int(x['info'][warm:n,6].sum()),
                            mean_integration_steps=float(x['info'][warm:n,7].mean()),
                            depth_limit_fraction=float((x['info'][warm:n,8]>=configs[0]['max_depth']).mean()))
                       for x in arrays]
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'savefig.dpi':220})
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    x=draws[:,:,3]
    lo,hi=x.min(),x.max()
    span=max(hi-lo,1e-8);grid=np.linspace(lo-.1*span,hi+.1*span,400)
    for c,chain in enumerate(x):
        color=COLORS[c%len(COLORS)]
        axes[0,0].hist(chain,bins=np.linspace(lo,hi,26),alpha=.45,color=color)
        if np.std(chain)>1e-10:
            axes[0,1].plot(grid,gaussian_kde(chain)(grid),color=color,label=f'Chain {c+1}')
        maxlag=min(100,len(chain)//4)
        axes[1,0].plot(np.arange(maxlag+1),acf(chain,maxlag),color=color)
        axes[1,1].plot(np.arange(1,n+1),full[c,:,3],color=color,lw=.65,alpha=.8)
    axes[0,0].set(xlabel=LABELS[3],ylabel='Count')
    axes[0,1].set(xlabel=LABELS[3],ylabel='Density')
    axes[0,1].legend(fontsize=9,frameon=False)
    axes[1,0].set(xlabel='Lag',ylabel='Autocorrelation',ylim=(-.15,1.05))
    axes[1,1].axvline(warm,color='k',ls='--',lw=1,label='End of warmup')
    axes[1,1].set(xlabel='Iteration',ylabel=LABELS[3])
    axes[1,1].legend(fontsize=9,frameon=False)
    for letter,ax in zip('ABCD',axes.flat):
        style(ax);ax.text(.02,.98,letter,va='top',transform=ax.transAxes,fontweight='bold',fontsize=18)
    fig.savefig(a.output/'trend_panels.png');plt.close(fig)
    # Keep all estimated coordinates visible; prior overlays use the actual
    # common proper prior, with axes set by draws rather than arbitrary limits.
    for kind in ['marginals','traces']:
        fig,axes=plt.subplots(6,3,figsize=(11,15),layout='constrained')
        for j,ax in enumerate(axes.flat):
            x=draws[:,:,j];lo,hi=x.min(),x.max();span=max(hi-lo,1e-8)
            grid=np.linspace(lo-.15*span,hi+.15*span,300)
            for c,chain in enumerate(x):
                color=COLORS[c%len(COLORS)]
                if kind=='marginals' and np.std(chain)>1e-10:
                    ax.plot(grid,gaussian_kde(chain)(grid),color=color)
                elif kind=='traces':
                    ax.plot(np.arange(1,n+1),full[c,:,j],color=color,lw=.5,alpha=.8)
            if kind=='marginals':
                ax.plot(grid,norm.pdf(grid,center[j],configs[0]['prior_sd'][j]),'k--',lw=1,label='Prior')
            else:
                ax.axvline(warm,color='k',ls='--',lw=1)
            ax.set_title(LABELS[j]);style(ax)
        fig.savefig(a.output/f'{kind}.png');plt.close(fig)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    main()
