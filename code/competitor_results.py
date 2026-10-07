"""Build SI tables from tracked Corenflos/Ditlevsen summaries.

Run from any directory: python code/competitor_results.py
No checkpoints, fitting, GPU, or sibling repositories are needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'imgs/competitors'
HIGH = ROOT / 'corenflos/results/particle_increase_j1000_final_100'
LOW = {
    ('Corenflos', 'Box'): 'corenflos/results/corenflos_j100_eps025_final_100',
    ('Ditlevsen-style', 'Box'): 'ditlevsen/results/block_smc_guided_j100_final_100',
    ('Corenflos', 'IF2'): 'corenflos/results/if2warm_ifad097_budget_j100_final_100',
    ('Ditlevsen-style', 'IF2'): 'ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100',
}


def read_runs(directory):
    evaluations = pd.read_csv(directory / 'final_evaluations.csv').set_index('start').sort_index()
    fits = pd.read_csv(directory / 'fit_summary.csv').set_index('start').sort_index()
    assert evaluations.index.is_unique and fits.index.is_unique
    assert evaluations.index.tolist() == fits.index.tolist() == list(range(100))
    assert evaluations.evaluation_status.eq('ok').all()
    assert evaluations.eval_particles.eq(5000).all()
    assert evaluations.eval_replicates.eq(36).all()
    assert np.isfinite(evaluations.euler20_loglik).all()
    return evaluations.euler20_loglik, fits.updates_completed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(ROOT / 'corenflos/results/ifad097_post_if2_reference/warm_start_evaluations.csv').set_index('start').sort_index().euler20_loglik
    assert baseline.index.tolist() == list(range(100))
    reference = pd.read_csv(ROOT / 'ditlevsen/results/reference/manuscript_likelihood.csv')
    ifad = reference.loc[reference.method.eq('IFAD-0.97') & reference.effort.eq('comparable'), 'euler_loglik']
    assert len(ifad) == 100 and np.isfinite(ifad).all()
    values, rows = {}, []
    for initialization in ('Box', 'IF2'):
        for method in ('Corenflos', 'Ditlevsen-style'):
            for particles in (100, 1000):
                variant = ('corenflos' if method == 'Corenflos' else 'ditlevsen') + ('_vanilla' if initialization == 'Box' else '_warm')
                directory = ROOT / LOW[method, initialization] if particles == 100 else HIGH / variant
                likelihood, updates = read_runs(directory)
                values[method, initialization, particles] = likelihood
                rows.append(dict(method=method, initialization=initialization, particles=particles,
                                 median=likelihood.median(), best=likelihood.max(), updates=updates.median(),
                                 source=str(directory.relative_to(ROOT))))
    likelihood, updates = read_runs(ROOT / 'ditlevsen/results/block_smc_guided_j5000_final_100')
    rows.insert(4, dict(method='Ditlevsen-style', initialization='Box', particles=5000,
                       median=likelihood.median(), best=likelihood.max(), updates=updates.median(),
                       source='ditlevsen/results/block_smc_guided_j5000_final_100'))
    rows.extend([
        dict(method='IF2 checkpoint', initialization='Box', particles=5000, median=baseline.median(), best=baseline.max(), updates=175, source='corenflos/results/ifad097_post_if2_reference/warm_start_evaluations.csv'),
        dict(method='IFAD-0.97', initialization='IF2', particles=5000, median=ifad.median(), best=ifad.max(), updates=400, source='ditlevsen/results/reference/manuscript_likelihood.csv'),
    ])
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    table = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Method & Start & $J$ & Median & Maximum & Updates \\', r'\midrule']
    for row in rows:
        table.append(f"{row['method']} & {row['initialization']} & {row['particles']:,} & {row['median']:.2f} & {row['best']:.2f} & {row['updates']:g} " + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'results_table.tex').write_text('\n'.join(table) + '\n')

    paired = []
    for row, particles in enumerate((100, 1000)):
        source = ROOT / 'corenflos/results/if2warm_ifad097_comparison' if particles == 100 else HIGH
        mismatch = pd.read_csv(source / 'objective_mismatch_if2warm_r20.csv')
        for col, method in enumerate(('Corenflos', 'Ditlevsen-style')):
            label = ('Corenflos' if col == 0 else 'Ditlevsen') + ' + IF2 warm start'
            frame = mismatch.loc[mismatch.source.eq(label)].set_index('start').sort_index()
            assert frame.index.tolist() == list(range(100))
            delta = values[method, 'IF2', particles] - baseline
            np.testing.assert_allclose(delta, frame.euler_delta, atol=1e-9)
            paired.append(dict(method=method, particles=particles, training_delta=frame.training_delta.median(),
                               euler_delta=delta.median(), improving=int(delta.gt(0).sum())))
    pd.DataFrame(paired).to_csv(OUT / 'paired_summary.csv', index=False)
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Method & $J$ & Fitting change & Euler-20 change & Improved \\', r'\midrule']
    for row in paired:
        table.append(f"{row['method']} & {row['particles']:,} & {row['training_delta']:.2f} & {row['euler_delta']:.2f} & {row['improving']}/100 " + r'\\')
    table += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'paired_table.tex').write_text('\n'.join(table) + '\n')



if __name__ == '__main__':
    main()
