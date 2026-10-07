"""Redraw manuscript Fig. 1A from the checked-in likelihood curves."""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dmop-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'curves.csv')
    parser.add_argument('--output', type=Path, default=HERE.parents[1] / 'imgs/095/mop.png')
    args = parser.parse_args()
    curves = np.genfromtxt(args.data, delimiter=',', names=True)
    plt.style.use(HERE / 'figure.mplstyle')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(curves['gamma'], curves['particle_filter'], linestyle='dashdot', label='True Log-Likelihood')
    ax.plot(curves['gamma'], curves['alpha_1'], label='Alpha=1 (similar to Poyiadjis, 2011)')
    ax.plot(curves['gamma'], curves['alpha_097'], label='Alpha=0.97 (Us)')
    ax.plot(curves['gamma'], curves['alpha_0'], label='Alpha=0 (similar to Naesseth, 2018)')
    ax.locator_params(axis='x', nbins=10)
    ax.set_xlabel('Recovery Rate')
    ax.set_ylabel('Log-Likelihood')
    ax.axvline(18, linestyle='--', color='black', label='Baseline Parameter')
    ax.legend()
    fig.tight_layout()
    ax.text(17.5, -3765, 'A', fontsize=36)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    main()
