"""Compare frozen reports differing only in prior scale or particle count."""
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

from compare import load_report
from report import NAMES, LABELS, style


def validate_pair(reference, alternative, kind):
    if kind not in {'prior', 'particles'}:
        raise ValueError('Unknown sensitivity comparison')
    if reference['method'] != alternative['method']:
        raise ValueError('Sensitivity reports must use the same method')
    if reference['model_sha256'] != alternative['model_sha256']:
        raise ValueError('Model source differs')
    left, right = reference['target'], alternative['target']
    allowed = {'prior_scale', 'prior_sd'} if kind == 'prior' else {'particles'}
    if {k: v for k, v in left.items() if k not in allowed} != {
            k: v for k, v in right.items() if k not in allowed}:
        raise ValueError('Target differs beyond the requested sensitivity setting')
    if kind == 'prior':
        if min(left['prior_scale'], right['prior_scale']) <= 0:
            raise ValueError('Prior scales must be positive')
        if left['prior_scale'] == right['prior_scale']:
            raise ValueError('Prior scale did not change')
        np.testing.assert_allclose(
            np.asarray(left['prior_sd']) / left['prior_scale'],
            np.asarray(right['prior_sd']) / right['prior_scale'], rtol=1e-12, atol=0)
    elif left['particles'] == right['particles']:
        raise ValueError('Particle count did not change')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--alternative', required=True, type=Path)
    parser.add_argument('--kind', required=True, choices=['prior', 'particles'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    method = json.loads((args.reference/'audit.json').read_text())['method']
    ra, rd, rt = load_report(args.reference, method)
    aa, ad, at = load_report(args.alternative, method)
    validate_pair(ra, aa, args.kind)
    args.output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(index=NAMES)
    for label, source in [('reference', rt), ('alternative', at)]:
        for field in ['mean', 'sd', 'q025', 'q975', 'rhat', 'ess_bulk', 'ess_tail', 'mcse_mean']:
            table[f'{label}_{field}'] = source[field].to_numpy()
    table['mean_difference'] = table.alternative_mean - table.reference_mean
    table['combined_mcse'] = np.hypot(table.reference_mcse_mean, table.alternative_mcse_mean)
    table['difference_over_mcse'] = table.mean_difference / table.combined_mcse
    table['difference_over_reference_sd'] = table.mean_difference / table.reference_sd
    table['sd_ratio'] = table.alternative_sd / table.reference_sd
    table.index.name = 'parameter'
    table.to_csv(args.output/'comparison.csv')
    adequate = bool(ra['numerical_convergence'] and aa['numerical_convergence'])
    field = 'prior_scale' if args.kind == 'prior' else 'particles'
    labels = [f'{field} = {audit["target"][field]:g}' for audit in [ra, aa]]
    audit = dict(kind=args.kind, reference=str(args.reference), alternative=str(args.alternative),
                 settings=[ra['target'][field], aa['target'][field]],
                 both_pass_numerical_checks=adequate,
                 interpretation='Descriptive comparison only. Mean shifts relative to MCSE are not equivalence tests. Unconverged-chain MCSE can be misleading. No insensitivity claim is made automatically.',
                 report_hashes={str(folder/name): hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                for folder in [args.reference, args.alternative]
                                for name in ['audit.json', 'diagnostics.csv', 'posterior_draws.npz']},
                 remaining_checks=['inspect all chain traces and marginals', 'interpret shifts with Monte Carlo uncertainty'])
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    colors = ['#0072B2', '#D55E00']
    plt.rcParams.update({'font.size':11, 'savefig.dpi':180})
    fig, axes = plt.subplots(6, 3, figsize=(11,15), layout='constrained')
    handles = [Line2D([], [], color=color, label=label) for color, label in zip(colors, labels)]
    handles.append(Line2D([], [], color='black', ls='--', label='Prior(s)'))
    fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False,
               title=None if adequate else 'Development samples: convergence checks have not passed')
    fig.supylabel('Density')
    for j, ax in enumerate(axes.flat):
        values = np.concatenate([rd[:,:,j].ravel(), ad[:,:,j].ravel()])
        low, high = values.min(), values.max()
        span = max(high-low, 1e-8)
        grid = np.linspace(low-.12*span, high+.12*span, 400)
        for draws, color in zip([rd, ad], colors):
            for chain in draws[:,:,j]:
                if np.std(chain) > 1e-10:
                    ax.plot(grid, gaussian_kde(chain)(grid), color=color, alpha=.65, lw=.8)
                else:
                    ax.axvline(chain[0], color=color, alpha=.65, lw=.8)
        priors = [(ra, colors[0]), (aa, colors[1])] if args.kind == 'prior' else [(ra, 'black')]
        for source, color in priors:
            target = source['target']
            ax.plot(grid, norm.pdf(grid, target['prior_center'][j], target['prior_sd'][j]),
                    color=color, ls='--', lw=1)
        ax.set_title(LABELS[j])
        style(ax)
    fig.savefig(args.output/'marginals.png')
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
