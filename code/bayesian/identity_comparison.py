"""Compare resumed identity-metric NUTS with the fixed diagonal PMMH control.

Both methods use iterations 501--1500; PMMH's first 500 are discarded for
comparison, not adaptive warmup. All draws enter diagnostics and densities.
The original warmup checkpoint must be an exact prefix of resumed NUTS.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

from report import NAMES, LABELS, acf
from paper_figures import COLORS, style, grid_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_batch(folder, method):
    configs, arrays, hashes = [], [], {}
    for c in range(4):
        d = folder/f'chain{c}'
        cfg = json.loads((d/'configuration.json').read_text())
        payload = (d/'checkpoint.npz').read_bytes()
        with np.load(io.BytesIO(payload)) as data:
            cp = {k: data[k].copy() for k in data.files}
        if cfg['method'] != method or len(cp['trace']) != 1500:
            raise ValueError('Expected four completed 1500-iteration control chains')
        completion = json.loads((d/'completion.json').read_text())
        if not completion['complete'] or completion['iteration'] != 1500:
            raise ValueError('Chain completion not verified')
        if not np.isfinite(cp['trace']).all():
            raise ValueError('Nonfinite chain')
        configs.append(cfg); arrays.append(cp)
        hashes[str(d/'checkpoint.npz')] = hashlib.sha256(payload).hexdigest()
        hashes[str(d/'configuration.json')] = sha(d/'configuration.json')
    if len({c['seed'] for c in configs}) != 4:
        raise ValueError('Repeated chain seeds')
    for field in ['method', 'warmup', 'particles', 'prior_scale', 'prior_center', 'prior_sd',
                  'pypomp_revision', 'names']:
        if any(c[field] != configs[0][field] for c in configs):
            raise ValueError(f'Inconsistent chains: {field}')
    center = np.asarray(configs[0]['prior_center'])
    scale = np.asarray(configs[0]['prior_sd'])/configs[0]['prior_scale']
    full = center + np.stack([a['trace'][:, 2:] for a in arrays])*scale
    return configs, arrays, full, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['nuts', 'pmmh', 'parent', 'output']:
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--paper', action='store_true',
                        help='Copy the three SI figures and diagnostic table into manuscript paths')
    args = parser.parse_args()
    nc, na, nf, nh = read_batch(args.nuts, 'nuts')
    pc, pa, pf, ph = read_batch(args.pmmh, 'pmrw')
    for field in ['particles', 'prior_scale', 'prior_center', 'prior_sd', 'names', 'pypomp_revision']:
        if nc[0][field] != pc[0][field]:
            raise ValueError(f'Different target: {field}')
    if nc[0]['warmup'] != 500 or pc[0]['warmup'] != 0:
        raise ValueError('Unexpected warmup conventions')
    if len({c['source_sha256']['model.py'] for c in nc+pc}) != 1:
        raise ValueError('Different model source')
    parent_hashes = {}
    for c, (cfg, cp) in enumerate(zip(nc, na)):
        parent = args.parent/f'chain{c}/checkpoint.npz'
        parent_hashes[str(parent)] = sha(parent)
        if cfg['parent_checkpoint_sha256'] != sha(parent):
            raise ValueError('Parent checkpoint hash mismatch')
        with np.load(parent) as old:
            if len(old['trace']) != 500 or not cfg['fixed_metric']:
                raise ValueError('Unexpected recovery state')
            for key in ['trace', 'info', 'step_size', 'iteration_seconds']:
                np.testing.assert_array_equal(cp[key][:500], old[key])
            np.testing.assert_array_equal(cp['chol'], old['chol'])
            np.testing.assert_array_equal(cp['step_size'][500:], np.full(1000, old['next_step_size']))
            np.testing.assert_array_equal(cp['next_step_size'], old['next_step_size'])
        np.testing.assert_array_equal(np.load(args.nuts/f'chain{c}/initial.npy'),
                                      np.load(args.pmmh/f'chain{c}/initial.npy'))
    draws = [nf[:, 500:], pf[:, 500:]]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'posterior_draws.npz', nuts=draws[0], pmmh=draws[1])
    rows = []
    for method, values in zip(['nuts', 'pmmh'], draws):
        for j, name in enumerate(NAMES):
            x = values[:, :, j]
            rows.append(dict(method=method, parameter=name, mean=float(x.mean()),
                             sd=float(x.std()), rhat=float(az.rhat(x, method='rank')),
                             ess_bulk=float(az.ess(x, method='bulk')),
                             ess_tail=float(az.ess(x, method='tail')),
                             mcse_mean=float(az.mcse(x, method='mean'))))
    table = pd.DataFrame(rows)
    table.to_csv(out/'diagnostics.csv', index=False)
    tex = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
           r'& \multicolumn{4}{c}{Identity-metric NUTS} & \multicolumn{4}{c}{Diagonal PMMH} \\',
           r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
           r'Coordinate & $\widehat R$ & Bulk ESS & Tail ESS & MCSE & $\widehat R$ & Bulk ESS & Tail ESS & MCSE \\',
           r'\midrule']
    for name, label in zip(NAMES, LABELS):
        pair = table[table.parameter == name].set_index('method')
        cells = []
        for method, other in [('nuts', 'pmmh'), ('pmmh', 'nuts')]:
            for field, fmt in [('rhat', '.3f'), ('ess_bulk', '.1f'),
                               ('ess_tail', '.1f'), ('mcse_mean', '.3g')]:
                value, reference = pair.loc[method, field], pair.loc[other, field]
                better = (abs(value-1) < abs(reference-1) if field == 'rhat'
                          else value < reference if field == 'mcse_mean'
                          else value > reference)
                cell = format(value, fmt)
                cells.append(r'\textbf{'+cell+'}' if better else cell)
        tex.append(label+' & '+' & '.join(cells)+r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (out/'diagnostics.tex').write_text('\n'.join(tex)+'\n')
    summary = {}
    for method, arrays in [('nuts', na), ('pmmh', pa)]:
        t = table[table.method == method]
        summary[method] = dict(max_rhat=float(t.rhat.max()),
            min_bulk_ess=float(t.ess_bulk.min()), min_tail_ess=float(t.ess_tail.min()),
            trend=t[t.parameter == 'beta_trend'].iloc[0].to_dict(),
            position_changed=[float(cp['info'][500:, 0].mean()) for cp in arrays])
    summary['nuts'].update(divergences=[int(cp['info'][500:, 6].sum()) for cp in na],
                          warmup_divergences=[int(cp['info'][:500, 6].sum()) for cp in na],
                          step_sizes=[float(cp['next_step_size']) for cp in na],
                          depth_limit_fraction=[float((cp['info'][500:, 8] >= nc[0]['max_depth']).mean()) for cp in na])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 10,
                         'axes.linewidth': .6, 'legend.fontsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    def save(fig, name):
        for ext in ['pdf', 'png']:
            fig.savefig(out/f'{name}.{ext}', dpi=220)
        plt.close(fig)
    chain_handles = [Line2D([], [], color=color, label=f'Chain {i+1}') for i, color in enumerate(COLORS)]
    prior_handle = Line2D([], [], color='.2', ls='--', label='Prior')
    names = ['Identity-metric NUTS', 'Diagonal PMMH']
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex='col', sharey='col', layout='constrained')
    grid = grid_for(draws[0][:, :, 3], draws[1][:, :, 3])
    for row, (method, values) in enumerate(zip(names, draws)):
        # Common absolute bandwidth for all eight displayed chain densities.
        bandwidth = np.sqrt(gaussian_kde(np.concatenate([x[:, :, 3].ravel() for x in draws])).covariance[0, 0])
        for c, chain in enumerate(values[:, :, 3]):
            axes[row, 0].plot(np.arange(501, 1501), chain, color=COLORS[c], lw=.55, alpha=.8)
            if np.std(chain, ddof=1) > 0:
                axes[row, 1].plot(grid, gaussian_kde(chain, bw_method=bandwidth/chain.std(ddof=1))(grid), color=COLORS[c])
            else:
                axes[row, 1].axvline(chain[0], color=COLORS[c])
            axes[row, 2].plot(np.arange(101), acf(chain, 100), color=COLORS[c])
        axes[row, 0].set_ylabel(method+'\n'+LABELS[3])
        axes[row, 1].plot(grid, norm.pdf(grid, nc[0]['prior_center'][3], nc[0]['prior_sd'][3]), color='.2', ls='--')
        axes[row, 1].set_ylabel('Density')
        axes[row, 2].set_ylabel('Autocorrelation')
        axes[row, 2].axhline(0, color='.5', lw=.5)
    for ax, label in zip(axes[1], ['Iteration', LABELS[3], 'Lag']):
        ax.set_xlabel(label)
    for letter, ax in zip('ABCDEF', axes.flat):
        style(ax); ax.set_title(letter, loc='left', fontweight='bold')
    fig.legend(handles=chain_handles+[prior_handle], loc='outside upper center', ncol=5, frameon=False)
    save(fig, 'trend_comparison')

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True, layout='constrained')
    for ax, method, full in zip(axes, names, [nf, pf]):
        for c in range(4):
            ax.plot(np.arange(1, 1501), full[c, :, 3], color=COLORS[c], lw=.55, alpha=.8)
        ax.axvspan(0, 500, color='.85', alpha=.5)
        ax.axvline(500, color='.4', ls='--', lw=.7)
        ax.set_ylabel(LABELS[3]); ax.set_title(method, loc='left'); style(ax)
    axes[0].text(.015, .96, 'Warmup', transform=axes[0].transAxes, va='top')
    axes[1].text(.015, .96, 'Discarded for comparison', transform=axes[1].transAxes, va='top')
    axes[1].set_xlabel('Iteration')
    fig.legend(handles=chain_handles, loc='outside upper center', ncol=4, frameon=False)
    save(fig, 'full_trend_traces')

    method_colors = ['#0072B2', '#D55E00']
    linestyles = ['-', '--']
    handles = [Line2D([], [], color=c, ls=ls, label=n) for c, ls, n in zip(method_colors, linestyles, names)]
    for kind in ['marginals', 'traces']:
        fig, axes = plt.subplots(6, 3, figsize=(10, 12), layout='constrained')
        for j, ax in enumerate(axes.flat):
            grid = grid_for(draws[0][:, :, j], draws[1][:, :, j])
            bandwidth = np.sqrt(gaussian_kde(np.concatenate([x[:, :, j].ravel() for x in draws])).covariance[0, 0])
            for values, color, ls in zip(draws, method_colors, linestyles):
                for chain in values[:, :, j]:
                    if kind == 'traces':
                        ax.plot(np.arange(501, 1501), chain, color=color, ls=ls, lw=.5, alpha=.65)
                    elif chain.std(ddof=1) > 0:
                        ax.plot(grid, gaussian_kde(chain, bw_method=bandwidth/chain.std(ddof=1))(grid), color=color, ls=ls, lw=.8, alpha=.8)
                    else:
                        ax.axvline(chain[0], color=color, ls=ls)
            if kind == 'marginals':
                ax.plot(grid, norm.pdf(grid, nc[0]['prior_center'][j], nc[0]['prior_sd'][j]), color='.2', ls=':', lw=.9)
            ax.set_title(LABELS[j]); style(ax)
        extras = [Line2D([], [], color='.2', ls=':', label='Prior')] if kind == 'marginals' else []
        fig.legend(handles=handles+extras, loc='outside upper center', ncol=3, frameon=False)
        if kind == 'marginals':
            fig.supylabel('Density')
        else:
            fig.supxlabel('Iteration')
        save(fig, 'all_parameter_'+kind)
    manifest = dict(checkpoints={**nh, **ph}, parent_checkpoints=parent_hashes,
        script_sha256=sha(Path(__file__)), window=[501, 1500], retained_per_chain=1000,
        chains_per_method=4, same_target=True, same_initial_points=True,
        parent_prefix_exact=True, metric_and_steps_frozen=True,
        thinning=1, axes='linear, shared within trend columns',
        density='Per-chain Gaussian KDE; common absolute Scott bandwidth from pooled methods/chains for each coordinate.',
        comparison='Equal iterations, not equal computing time. PMMH discard is not adaptation.',
        visual_review='pending', posterior_draws_sha256=sha(out/'posterior_draws.npz'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if args.paper:
        repo = Path(__file__).resolve().parents[2]
        for name in ['trend_comparison', 'all_parameter_traces', 'all_parameter_marginals']:
            shutil.copyfile(out/f'{name}.pdf', repo/'imgs/pmcmc'/f'identity_control_{name}.pdf')
        target = Path(__file__).resolve().parent/'results/paper/identity_diagnostics.tex'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(out/'diagnostics.tex', target)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
