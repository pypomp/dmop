"""Diagnose independent corrected chains and redraw the existing panel layout."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    if len({c['source_sha256']['model.py'] for c in configs}) != 1:
        raise ValueError('Model source differs between chains')
    payloads=[(d/'checkpoint.npz').read_bytes() for d in a.chains]
    arrays=[np.load(io.BytesIO(data)) for data in payloads]
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
    # Freeze the exact arrays used by this report for between-method review.
    np.savez_compressed(a.output/'posterior_draws.npz',draws=draws)
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
    # Same compact table layout as the SI archive, now for the common target.
    # ESS and MCSE combine chains; the archive used within-chain extrema
    # because its fitted priors differed. The new caption must say this.
    tex=[r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Coordinate & $\widehat R$ & Bulk ESS & Tail ESS & MCSE \\',r'\midrule']
    for label,row in zip(LABELS,rows):
        values=[row['rhat'],row['ess_bulk'],row['ess_tail'],row['mcse_mean']]
        cells=[format(value,spec) if np.isfinite(value) else r'\textemdash'
               for value,spec in zip(values,['.3f','.1f','.1f','.3g'])]
        tex.append(label+' & '+' & '.join(cells)+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    (a.output/'diagnostics.tex').write_text('\n'.join(tex)+'\n')
    lengths=[len(x['trace']) for x in arrays]
    complete=all((d/'completion.json').exists() and length==c['warmup']+c['draws']
                 for d,c,length in zip(a.chains,configs,lengths))
    gates=dict(four_chains=len(arrays)>=4,complete=complete,equal_lengths=len(set(lengths))==1,
               rhat=bool(np.isfinite(table.rhat).all() and (table.rhat<=1.01).all()),
               ess=bool((table[['ess_bulk','ess_tail']]>=400).all().all()))
    audit=dict(chains=[str(d) for d in a.chains],warmup=warm,retained=n-warm,
               method=configs[0]['method'],
               target={field:configs[0][field] for field in
                       ['particles','prior_scale','names','prior_center','prior_sd','pypomp_revision']},
               model_sha256=configs[0]['source_sha256']['model.py'],
               draws_sha256=hashlib.sha256((a.output/'posterior_draws.npz').read_bytes()).hexdigest(),
               diagnostic_convention='Rank split R-hat, bulk/tail ESS and mean MCSE across chains on displayed estimation coordinates',
               marginal_density_estimator='Gaussian KDE with scipy default Scott bandwidth, fitted separately to retained draws of each chain',
               available_iterations=lengths,truncated_iterations=[length-n for length in lengths],gates=gates,
               numerical_convergence=all(gates.values()),
               remaining_checks=['visual inspection','prior sensitivity','particle sensitivity'],
               source_sha256={str(d/'checkpoint.npz'):hashlib.sha256(data).hexdigest()
                              for d,data in zip(a.chains,payloads)},
               acceptance=[dict(parameter=float(x['info'][warm:n,0].mean()),
                                randomness=float(x['info'][warm:n,1].mean())) for x in arrays])
    if configs[0]['method']=='nuts':
        audit['nuts']=[dict(divergences=int(x['info'][warm:n,6].sum()),
                            mean_integration_steps=float(x['info'][warm:n,7].mean()),
                            max_depth_reached_fraction=float((x['info'][warm:n,8]>=configs[0]['max_depth']).mean()))
                       for x in arrays]
        for diagnostics,x in zip(audit['nuts'],arrays):
            if x['info'].shape[1]>9:
                diagnostics['forced_depth_limit_fraction']=float(((x['info'][warm:n,8]>=configs[0]['max_depth']) &
                    (x['info'][warm:n,9]==0) & (x['info'][warm:n,6]==0)).mean())
        gates['no_divergences']=all(d['divergences']==0 for d in audit['nuts'])
        audit['numerical_convergence']=all(gates.values())
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'savefig.dpi':220})
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    x=draws[:,:,3]
    lo,hi=x.min(),x.max()
    span=max(hi-lo,1e-8);grid=np.linspace(lo-.1*span,hi+.1*span,400)
    autocorrelations=[]
    for c,chain in enumerate(x):
        color=COLORS[c%len(COLORS)]
        axes[0,0].hist(chain,bins=np.linspace(lo,hi,26),alpha=.45,color=color)
        if np.std(chain)>1e-10:
            axes[0,1].plot(grid,gaussian_kde(chain)(grid),color=color,label=f'Chain {c+1}')
        maxlag=min(100,len(chain)//4)
        autocorrelation=acf(chain,maxlag)
        autocorrelations.append(autocorrelation)
        axes[1,0].plot(np.arange(maxlag+1),autocorrelation,color=color)
        axes[1,1].plot(np.arange(1,n+1),full[c,:,3],color=color,lw=.65,alpha=.8)
    axes[0,0].set(xlabel=LABELS[3],ylabel='Count')
    axes[0,1].set(xlabel=LABELS[3],ylabel='Density')
    axes[0,1].legend(fontsize=9,frameon=False)
    acf_min=min(float(np.nanmin(values)) for values in autocorrelations)
    axes[1,0].set(xlabel='Lag',ylabel='Autocorrelation',ylim=(min(-.15,acf_min-.05),1.05))
    axes[1,1].axvline(warm,color='k',ls='--',lw=1,label='End of warmup')
    axes[1,1].set(xlabel='Iteration',ylabel=LABELS[3])
    axes[1,1].legend(fontsize=9,frameon=True,facecolor='white',edgecolor='none',framealpha=1.)
    for letter,ax in zip('ABCD',axes.flat):
        style(ax);ax.set_title(letter,loc='left',fontweight='bold',fontsize=18)
    fig.savefig(a.output/'trend_panels.png');plt.close(fig)
    # Keep all estimated coordinates visible; prior overlays use the actual
    # common proper prior, with axes set by draws rather than arbitrary limits.
    for kind in ['marginals','traces']:
        fig,axes=plt.subplots(6,3,figsize=(11,15),layout='constrained')
        handles=[Line2D([],[],color=COLORS[c%len(COLORS)],label=f'Chain {c+1}')
                 for c in range(len(draws))]
        handles.append(Line2D([],[],color='k',ls='--',
                              label='Prior' if kind=='marginals' else 'End of warmup'))
        fig.legend(handles=handles,loc='outside upper center',ncol=len(handles),frameon=False)
        if kind=='traces':
            fig.supxlabel('Iteration')
        else:
            fig.supylabel('Density')
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
