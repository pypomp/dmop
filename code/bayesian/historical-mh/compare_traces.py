"""Compare saved historical MH and current PMMH at 1,500 iterations.

This is a descriptive comparison of different configurations, not a controlled
test of the prior or a timing comparison. No sampling is performed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--control', type=Path, help='Completed historical-proposal-control-v1/pmrw directory')
parser.add_argument('--output', type=Path, default=ROOT / 'comparison')
args = parser.parse_args()
OUT = args.output
if OUT.exists():
    raise ValueError('Choose a new output directory to preserve frozen comparisons')
provenance = json.loads((ROOT / 'provenance.json').read_text())
historical = []
current = []
hashes = {}
current_configs = []
current_initials = []
for c in range(4):
    raw = (ROOT / f'theta_{c}.npy').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance['files'][f'theta_{c}.npy']['sha256']
    historical.append(np.load(io.BytesIO(raw))[:, 6])
    folder = ROOT.parent / f'corrected/results/gpu-comparison-v2/pmrw/chain{c}'
    cfg = json.loads((folder / 'configuration.json').read_text())
    current_configs.append(cfg)
    current_initials.append(np.load(folder / 'initial.npy'))
    raw = (folder / 'checkpoint.npz').read_bytes()
    z = np.load(io.BytesIO(raw))['trace'][:1500, 2:]
    assert z.shape == (1500, 18)
    current.append(cfg['prior_center'][3] + z[:, 3] * cfg['prior_sd'][3] / cfg['prior_scale'])
    hashes[str(folder.relative_to(ROOT.parent) / 'checkpoint.npz')] = hashlib.sha256(raw).hexdigest()
historical, current = np.array(historical), np.array(current)
datasets = [('Archived MH', historical), ('Current PMMH', current)]
frozen = dict(historical=historical, current=current)
if args.control:
    control = []
    for c in range(4):
        folder = args.control / f'chain{c}'
        cfg = json.loads((folder / 'configuration.json').read_text())
        for field in ['particles', 'prior_scale', 'prior_center', 'prior_sd',
                      'names', 'pypomp_revision', 'method', 'xla_flags']:
            assert cfg[field] == current_configs[c][field], field
        assert cfg['source_sha256']['model.py'] == current_configs[c]['source_sha256']['model.py']
        assert cfg['warmup'] == 0 and cfg['draws'] == 1500
        assert (folder / 'completion.json').exists(), 'Control has not completed'
        np.testing.assert_array_equal(np.load(folder / 'initial.npy'), current_initials[c])
        raw = (folder / 'checkpoint.npz').read_bytes()
        state = np.load(io.BytesIO(raw))
        z = state['trace'][:, 2:]
        assert z.shape == (1500, 18) and np.isfinite(z).all()
        np.testing.assert_array_equal(state['step_size'], np.full(1500, .02))
        np.testing.assert_array_equal(state['chol'], np.eye(18))
        control.append(cfg['prior_center'][3] + z[:, 3]*cfg['prior_sd'][3]/cfg['prior_scale'])
        hashes[str(folder / 'checkpoint.npz')] = hashlib.sha256(raw).hexdigest()
    frozen['control'] = np.array(control)
    datasets.append(('PMMH with original proposal', frozen['control']))
OUT.mkdir(parents=True, exist_ok=False)
np.savez_compressed(OUT / 'trend_draws.npz', **frozen)
colors = ['#00b9e8', '#d0ca33', '#fb65b3', '#62dcb5']
fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3.4*len(datasets)), layout='constrained')
lo = min(data.min() for _, data in datasets) - .08
hi = max(data.max() for _, data in datasets) + .08
grid = np.linspace(lo, hi, 500)
rows = []
for row, (label, data) in enumerate(datasets):
    for c, color in enumerate(colors):
        axes[row, 0].plot(np.arange(1, 1501), data[c], color=color, lw=.9, label=f'Chain {c+1}')
        axes[row, 1].plot(grid, gaussian_kde(data[c, 500:])(grid), color=color, lw=1.8)
        rows.append(dict(run=label, chain=c+1, start=float(data[c, 0]), end=float(data[c, -1]),
                         min=float(data[c].min()), max=float(data[c].max()),
                         rms_jump=float(np.sqrt(np.mean(np.diff(data[c])**2)))))
    axes[row, 0].axvline(500, color='black', ls='--', lw=1)
    axes[row, 0].set(xlabel='Iteration', ylabel=r'$100\beta_{\mathrm{trend}}$',
                     title=label+' — first 1,500 iterations', ylim=(lo, hi), xlim=(0, 1500))
    axes[row, 1].set(xlabel=r'$100\beta_{\mathrm{trend}}$', ylabel='Density',
                     title='Iterations 501–1,500', xlim=(lo, hi))
    for ax in axes[row]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.2, ls='--')
axes[0, 0].legend(ncol=2, fontsize=9)
maxdensity=max(ax.get_ylim()[1] for ax in axes[:, 1])
for ax in axes[:, 1]: ax.set_ylim(0, maxdensity)
fig.savefig(OUT / 'same_length_traces.png', dpi=180)
plt.close(fig)
(OUT / 'summary.json').write_text(json.dumps(dict(
    iteration_window=[1, 1500], density_window=[501, 1500],
    caveat='Different targets, priors, initial states and proposals; no causal attribution or timing comparison.',
    control_matches_current_target_and_starts=bool(args.control),
    checkpoint_sha256=hashes, rows=rows), indent=2)+'\n')
print(json.dumps(rows, indent=2))
