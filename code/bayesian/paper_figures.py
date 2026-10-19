"""Reproduce the common-target Bayesian manuscript figures from frozen reports.

Run from any directory. Writes vector PDFs and PNG previews to imgs/pmcmc,
and the diagnostic table, comparisons and input hashes to results/paper.
No simulations, fitting or selection of draw windows is performed here.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from compare import load_report
from report import LABELS, NAMES, acf

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / 'results'
COLORS = ['#0072B2', '#B59800', '#CC5799', '#009E73']


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='.88', lw=.45, ls='--')
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(width=.6, length=3, pad=2)


def save(fig, name):
    for suffix in ('pdf', 'png'):
        fig.savefig(ROOT / 'imgs/pmcmc' / f'{name}.{suffix}', dpi=300)
    plt.close(fig)


def legend(fig, prior=True):
    handles = [Line2D([], [], color=c, lw=1.2, label=f'Chain {i+1}')
               for i, c in enumerate(COLORS)]
    if prior:
        handles.append(Line2D([], [], color='.2', ls='--', lw=1, label='Prior'))
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
               frameon=False, handlelength=1.7, columnspacing=1.3)


def grid_for(*samples):
    lo = min(x.min() for x in samples)
    hi = max(x.max() for x in samples)
    span = max(hi-lo, 1e-8)
    return np.linspace(lo-.10*span, hi+.10*span, 400)


def main():
    paths = {
        'nuts': RESULTS / 'gpu-forward-nuts-v2/review-nuts-2000',
        'pmmh': RESULTS / 'gpu-comparison-v2/review-pmrw-2000',
        'reference': RESULTS / 'gpu-comparison-v2/review-pmrw-30000',
    }
    reports = {k: load_report(p, 'nuts' if k == 'nuts' else 'pmrw')
               for k, p in paths.items()}
    na, nd, nt = reports['nuts']
    pa, pd_, pt = reports['pmmh']
    ra, rd, rt = reports['reference']
    assert nd.shape == pd_.shape == (4, 2000, 18)
    assert rd.shape == (4, 30000, 18)
    assert all(a['target'] == na['target'] and a['model_sha256'] == na['model_sha256']
               for a in (pa, ra))
    np.testing.assert_array_equal(pd_, rd[:, :2000])
    out = RESULTS / 'paper'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 9,
                         'axes.linewidth': .6, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8), sharex='col', sharey='col',
                             layout='constrained')
    grid = grid_for(nd[:, :, 3], pd_[:, :, 3])
    for row, (method, draws) in enumerate([('NUTS', nd), ('PMMH', pd_)]):
        for c, chain in enumerate(draws[:, :, 3]):
            axes[row, 0].plot(np.arange(1, 2001), chain, color=COLORS[c], lw=.4, alpha=.75)
            axes[row, 1].plot(grid, gaussian_kde(chain)(grid), color=COLORS[c], lw=1.1)
            axes[row, 2].plot(np.arange(101), acf(chain, 100), color=COLORS[c], lw=1.1)
        axes[row, 0].set_ylabel(method + '\n' + LABELS[3])
        axes[row, 1].plot(grid, norm.pdf(grid, na['target']['prior_center'][3],
                                       na['target']['prior_sd'][3]), color='.2', ls='--', lw=.9)
        axes[row, 1].set_ylabel('Density')
        axes[row, 2].set_ylabel('Autocorrelation')
        axes[row, 2].axhline(0, color='.5', lw=.5)
    for ax, label in zip(axes[1], ['Retained iteration', LABELS[3], 'Lag']):
        ax.set_xlabel(label)
    for letter, ax in zip('ABCDEF', axes.flat):
        style(ax)
        ax.set_title(letter, loc='left', fontweight='bold')
    legend(fig)
    save(fig, 'common_target_trend')

    # Fixed display thinning; diagnostics and numerical comparisons use all draws.
    reference_display = rd[:, 24::25, :]
    assert reference_display.shape == (4, 1200, 18)
    for kind in ('marginals', 'traces', 'reference', 'chains'):
        fig, axes = plt.subplots(6, 3, figsize=(7.2, 9.0), layout='constrained')
        if kind == 'chains':
            fig.legend(handles=[
                Line2D([], [], color='#0072B2', lw=1., label='NUTS: four chains'),
                Line2D([], [], color='#A33D00', lw=1., ls='--', label='PMMH: four chains')],
                loc='outside upper center', ncol=2, frameon=False)
        elif kind == 'reference':
            fig.legend(handles=[
                Patch(facecolor='#0072B2', edgecolor='#0072B2', alpha=.25,
                      label='NUTS'),
                Line2D([], [], color='#A33D00', lw=1.5, ls='--',
                       label='PMMH (every 25th draw)')],
                loc='outside upper center', ncol=2, frameon=False)
        else:
            legend(fig, prior=kind == 'marginals')
        fig.supylabel('Estimation coordinate' if kind == 'traces' else 'Density', fontsize=8)
        if kind == 'traces':
            fig.supxlabel('Retained iteration', fontsize=8)
        for j, ax in enumerate(axes.flat):
            grid = grid_for(nd[:, :, j], rd[:, :, j]) if kind == 'reference' else grid_for(nd[:, :, j])
            if kind == 'chains':
                grid = grid_for(nd[:, :, j], pd_[:, :, j])
                pooled = np.concatenate([nd[:, :, j].ravel(), pd_[:, :, j].ravel()])
                bandwidth = float(np.sqrt(gaussian_kde(pooled).covariance[0, 0]))
                for draws, color, line in ((nd, '#0072B2', '-'), (pd_, '#A33D00', '--')):
                    for chain in draws[:, :, j]:
                        kde = gaussian_kde(chain, bw_method=bandwidth / np.std(chain, ddof=1))
                        ax.plot(grid, kde(grid), color=color, ls=line, lw=.9, alpha=.85)
            elif kind == 'reference':
                nuts_values = nd[:, :, j].ravel()
                pm_values = reference_display[:, :, j].ravel()
                # Use a common absolute bandwidth so visual differences are not
                # driven by the unequal displayed sample counts.
                pm_kde = gaussian_kde(pm_values)
                bandwidth = float(np.sqrt(pm_kde.covariance[0, 0]))
                nuts_kde = gaussian_kde(nuts_values,
                    bw_method=bandwidth / np.std(nuts_values, ddof=1))
                density = nuts_kde(grid)
                ax.fill_between(grid, 0, density, color='#0072B2', alpha=.20)
                ax.plot(grid, density, color='#0072B2', lw=1.2)
                ax.plot(grid, pm_kde(grid), color='#A33D00', lw=1.5,
                        ls=(0, (4, 2)))
            else:
                for c, chain in enumerate(nd[:, :, j]):
                    if kind == 'traces':
                        ax.plot(np.arange(1, 2001), chain, color=COLORS[c], lw=.4, alpha=.75)
                    else:
                        ax.plot(grid, gaussian_kde(chain)(grid), color=COLORS[c], lw=1.)
                if kind == 'marginals':
                    ax.plot(grid, norm.pdf(grid, na['target']['prior_center'][j],
                                           na['target']['prior_sd'][j]), color='.2', ls='--', lw=.8)
            style(ax)
            ax.set_title(LABELS[j])
        save(fig, 'common_target_' + kind)

    tex = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
           r'& \multicolumn{4}{c}{NUTS} & \multicolumn{4}{c}{PMMH} \\',
           r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
           r'Coordinate & $\widehat R$ & Bulk ESS & Tail ESS & MCSE & $\widehat R$ & Bulk ESS & Tail ESS & MCSE \\',
           r'\midrule']
    bold_cells = {}
    fields = ('rhat', 'ess_bulk', 'ess_tail', 'mcse_mean')
    formats = ('.3f', '.0f', '.0f', '.3g')
    for name, label in zip(NAMES, LABELS):
        rows = [nt.loc[name], pt.loc[name]]
        displayed = [[format(row[field], fmt) for field, fmt in zip(fields, formats)]
                     for row in rows]
        winners = []
        for j, field in enumerate(fields):
            values = [float(cells[j]) for cells in displayed]
            scores = [abs(v - 1) if field == 'rhat' else
                      -v if field in ('ess_bulk', 'ess_tail') else v for v in values]
            winners.append([i for i, score in enumerate(scores) if score == min(scores)])
        bold_cells[name] = dict(zip(fields, winners))
        cells = [label]
        for i in range(2):
            cells.extend(r'\textbf{' + value + '}' if i in winners[j] else value
                         for j, value in enumerate(displayed[i]))
        tex.append(' & '.join(cells) + r' \\')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (out / 'diagnostics.tex').write_text('\n'.join(tex) + '\n')
    for key, table in [('pmmh', pt), ('reference', rt)]:
        comparison = pd.concat({'nuts': nt, key: table}, axis=1)
        comparison.to_csv(out / f'comparison-{key}.csv')
    manifest = {'reports': {k: str(p.relative_to(ROOT)) for k, p in paths.items()},
                'inputs_sha256': {str((p/f).relative_to(ROOT)): hashlib.sha256((p/f).read_bytes()).hexdigest()
                                  for p in paths.values() for f in ('audit.json', 'posterior_draws.npz', 'diagnostics.csv')},
                'main_window': 'First 2000 retained draws after 500 warmup updates, both methods',
                'reference_window': 'All 30000 retained PMMH draws for diagnostics; display uses iterations 25,50,...,30000 after warmup (1200 per chain)',
                'chain_window': 'First2000 retained draws after500warmup, all four chains of both methods, no thinning or pooling of displayed curves',
                'chain_density': 'One KDE per chain. Within each coordinate all eight curves use the same absolute Gaussian bandwidth, Scott rule on pooled16000draws from both methods.',
                'scales': 'Linear; main figure shares axes by column',
                'density': 'Main/prior panels: separate Gaussian KDE per chain, scipy Scott bandwidth. Reference overlay: pool four chains per method; PMMH display thinned by25, Scott bandwidth from thinned PMMH, same absolute bandwidth for NUTS.',
                'timing': 'Equal-iteration main comparison; no runtime claim',
                'bold_rule': 'Rhat closest to1, higher bulk/tail ESS, lower MCSE; compare displayed precision and bold both for a displayed tie. Method index0=NUTS,1=PMMH.',
                'bold_cells': bold_cells,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Wrote five PDF/PNG figures and paired diagnostics to', out)


if __name__ == '__main__':
    main()
