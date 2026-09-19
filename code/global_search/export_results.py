"""Export the global search results to version-independent tabular files.

The pickles in dmop_results/ and mif_results/ can only be read by the pypomp
version that wrote them, so the numbers behind the manuscript become
unrecoverable as soon as pypomp's internal classes are renamed. These exports
keep the values that the manuscript actually reports in plain tables that any
tool can read:

exports/final_estimates.csv   final log-likelihood and parameter estimate for
                              each of the 100 replicates of every run
exports/loglik_traces.csv.gz  log-likelihood at each iteration, for the
                              N_MONITORS=1 runs that monitor it
exports/timings.csv           wall clock seconds per algorithm phase

Full parameter traces are deliberately not exported; they run to hundreds of
megabytes, and both manuscript figures use only the final iteration.

Usage: python export_results.py
"""

import os
import pickle

import pandas as pd

RUNS = [
    ("IFAD-0.97", "dmop_results/dacca_results_rl4_alpha0.97_nm{nm}{sfx}.pkl"),
    ("IFAD-1", "dmop_results/dacca_results_rl4_alpha1.0_nm{nm}{sfx}.pkl"),
    ("IFAD-0", "dmop_results/dacca_results_rl4_alpha0.0_nm{nm}{sfx}.pkl"),
    ("IF2", "mif_results/dacca_results_rl4_nm{nm}{sfx}.pkl"),
]
EFFORTS = [("", "comparable"), ("_long", "extended")]
OUT = "exports"

final_rows = []
trace_rows = []
timing_rows = []
missing = []

for sfx, effort in EFFORTS:
    for model, pat in RUNS:
        for nm in (0, 1):
            path = pat.format(nm=nm, sfx=sfx)
            if not os.path.exists(path):
                missing.append(f"{path} (absent)")
                continue
            try:
                with open(path, "rb") as fh:
                    obj = pickle.load(fh)
            except Exception as e:
                # A pickle written by an older pypomp cannot be read back, which
                # is precisely what these exports exist to prevent. Report it
                # rather than quietly dropping the run from the tables.
                missing.append(f"{path} (unreadable: {type(e).__name__})")
                continue
            tag = {"model": model, "effort": effort, "n_monitors": nm}

            # Final estimates: the pre-pruning pfilter evaluation of every
            # replicate, which is what the manuscript's tables and the
            # raincloud and density figures are built from.
            res = obj.results(index=-2).copy()
            res = res.rename(columns={"theta_idx": "rep"})
            final_rows.append(res.assign(**tag))

            # Wall clock per phase.
            t = obj.time().reset_index(drop=True)
            timing_rows.append(t.rename(columns={"time": "seconds"}).assign(**tag))

            # Log-likelihood traces. With N_MONITORS=0 the per-iteration
            # log-likelihood is not evaluated, so those runs carry no signal.
            if nm == 1:
                tr = obj.traces()
                keep = [c for c in ("theta_idx", "iteration", "method", "logLik", "se")
                        if c in tr.columns]
                tr = tr[keep].rename(columns={"theta_idx": "rep"})
                trace_rows.append(tr.assign(**tag))

os.makedirs(OUT, exist_ok=True)


def _write(rows, name, **kw):
    if not rows:
        print(f"{name}: nothing to write")
        return
    df = pd.concat(rows, ignore_index=True)
    lead = [c for c in ("model", "effort", "n_monitors") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False, **kw)
    print(f"{name}: {len(df):,} rows, {os.path.getsize(path) / 1e6:.1f} MB")


_write(final_rows, "final_estimates.csv")
_write(trace_rows, "loglik_traces.csv.gz", compression="gzip")
_write(timing_rows, "timings.csv")

if missing:
    print(f"\n{len(missing)} run(s) not present:")
    for m in missing:
        print(f"  {m}")
