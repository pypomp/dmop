"""Compare saved historical MH and current PMMH at 1,500 iterations.

This is a descriptive comparison of different configurations, not a controlled
test of the prior or a timing comparison. No sampling is performed.
"""
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
OUT = ROOT / 'comparison'
OUT.mkdir(exist_ok=True)
provenance = json.loads((ROOT / 'provenance.json').read_text())
historical = []
current = []
hashes = {}
for c in range(4):
    raw = (ROOT / f'theta_{c}.npy').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance['files'][f'theta_{c}.npy']['sha256']
    historical.append(np.load(io.BytesIO(raw))[:, 6])
    folder = ROOT.parent / f'corrected/results/gpu-comparison-v2/pmrw/chain{c}'
    cfg = json.loads((folder / 'configuration.json').read_text())
    raw = (folder / 'checkpoint.npz').read_bytes()
    z = np.load(io.BytesIO(raw))['trace'][:1500, 2:]
    assert z.shape == (1500, 18)
    current.append(cfg['prior_center'][3] + z[:, 3] * cfg['prior_sd'][3] / cfg['prior_scale'])
    hashes[str(folder.relative_to(ROOT.parent) / 'checkpoint.npz')] = hashlib.sha256(raw).hexdigest()
historical, current = np.array(historical), np.array(current)
np.savez_compressed(OUT / 'trend_draws.npz', historical=historical, current=current)
colors = ['#00b9e8', '#d0ca33', '#fb65b3', '#62dcb5']
fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout='constrained')
lo = min(historical.min(), current.min()) - .08
hi = max(historical.max(), current.max()) + .08
grid = np.linspace(lo, hi, 500)
rows = []
for row, (label, data) in enumerate([('Archived MH', historical), ('Current PMMH', current)]):
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
    current_checkpoint_sha256=hashes, rows=rows), indent=2)+'\n')
print(json.dumps(rows, indent=2))
