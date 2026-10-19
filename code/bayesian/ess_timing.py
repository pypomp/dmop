"""Combine frozen main-comparison ESS estimates with recorded batch wall times.

Run from any directory after restoring the archived Bayesian results. Outputs
stay under results/paper. This is an observational timing comparison: GPU load
was not controlled, some chains mix poorly, and pilot preparation is excluded.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / 'results'
RUNS = {
    'nuts': ('gpu-forward-nuts-v2/review-nuts-2000',
             'gpu-forward-nuts-v2/nuts', 2000),
    'pmmh': ('gpu-comparison-v2/review-pmrw-2000',
             'gpu-comparison-v2/pmrw-long', 2000),
    'pmmh_long': ('gpu-comparison-v2/review-pmrw-30000',
                  'gpu-comparison-v2/pmrw-long', 30000),
}


def main():
    hashes, summary, frames, targets = {}, {}, {}, []

    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return data

    for name, (review, run, retained) in RUNS.items():
        review, run = ROOT / review, ROOT / run
        table = pd.read_csv(io.BytesIO(read(review / 'diagnostics.csv')))
        table = table.set_index('parameter')
        with np.load(io.BytesIO(read(review / 'posterior_draws.npz'))) as data:
            draws = data['draws']
        assert draws.shape == (4, retained, 18)
        batch_times, infos = [], []
        for c in range(4):
            folder = run / f'chain{c}'
            cfg = json.loads(read(folder / 'configuration.json'))
            assert cfg['warmup'] == 500
            assert 'whole batch' in cfg['timing_convention']
            targets.append({k: cfg[k] for k in
                            ('particles', 'prior_center', 'prior_sd', 'prior_scale',
                             'pypomp_revision')})
            targets[-1]['model_sha256'] = cfg['source_sha256']['model.py']
            assert list(table.index) == cfg['names']
            with np.load(io.BytesIO(read(folder / 'checkpoint.npz'))) as data:
                end = 500 + retained
                expected = (np.asarray(cfg['prior_center'])
                            + data['trace'][500:end, 2:]
                            * np.asarray(cfg['prior_sd']) / cfg['prior_scale'])
                np.testing.assert_allclose(draws[c], expected, rtol=1e-14, atol=1e-14)
                batch_times.append(data['iteration_seconds'][:end].copy())
                infos.append(data['info'][500:end].copy())
        for t in batch_times[1:]:
            np.testing.assert_array_equal(t, batch_times[0])
        seconds = float(batch_times[0][500:].sum())
        assert seconds > 0
        warmup_seconds = float(batch_times[0][:500].sum())
        summary[name] = dict(retained_per_chain=retained, sampling_seconds=seconds,
                             warmup_seconds=warmup_seconds,
                             seconds_per_batch=seconds / retained)
        for kind in ('bulk', 'tail'):
            table[f'{kind}_ess_per_second'] = table[f'ess_{kind}'] / seconds
            table[f'{kind}_ess_per_second_with_warmup'] = (
                table[f'ess_{kind}'] / (seconds + warmup_seconds))
        frames[name] = table[['rhat', 'ess_bulk', 'ess_tail'] +
                             [c for c in table if '_per_second' in c]]
        if name == 'nuts':
            info = np.stack(infos)
            summary[name].update(
                mean_integration_steps=float(info[:, :, 7].mean()),
                fraction_depth_5=float((info[:, :, 8] == 5).mean()),
                fraction_turning=float(info[:, :, 9].mean()))
    assert all(t == targets[0] for t in targets)
    out = pd.concat(frames, axis=1)
    out.columns = ['_'.join(c) for c in out.columns]
    for baseline in ('pmmh', 'pmmh_long'):
        for kind in ('bulk', 'tail'):
            ratio = (out[f'nuts_{kind}_ess_per_second'] /
                     out[f'{baseline}_{kind}_ess_per_second'])
            out[f'nuts_over_{baseline}_{kind}'] = ratio
            summary[f'nuts_over_{baseline}_{kind}'] = dict(
                minimum=float(ratio.min()), median=float(ratio.median()),
                maximum=float(ratio.max()), nuts_faster_coordinates=int((ratio > 1).sum()))
    dest = ROOT / 'paper'
    dest.mkdir(exist_ok=True)
    out.to_csv(dest / 'ess-timing.csv')
    summary['input_sha256'] = hashes
    summary['limitations'] = ('Observed batch timings; unequal GPU load. ESS estimates '
                              'are affected by incomplete mixing. Sampling excludes '
                              'warmup and preparation; warmup-inclusive columns still '
                              'exclude pilot runs and tuning screens.')
    (dest / 'ess-timing.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
