"""Recover MC errors for the exact archived main-text IFAD/IF2 estimates.

Reads the same monitored result objects as ditlevsen.reference_results;
the separate unmonitored runs supply fitting times only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ditlevsen.reference_results import _paths, _read_pickle, _full_pfilter_entry

ROOT = Path(__file__).resolve().parents[2]


def main():
    reference = pd.read_csv(ROOT/"ditlevsen/results/reference/manuscript_likelihood.csv")
    trace = pd.read_csv(ROOT/"ditlevsen/results/reference/manuscript_trace_summary.csv")
    rows = []
    for method in ("IFAD-0.97", "IF2"):
        path, _, historical = _paths(method, "comparable")
        result = _full_pfilter_entry(_read_pickle(path, historical=historical))
        raw = np.asarray(result.logLiks, dtype=float)
        weights = np.exp(raw-raw.max(1, keepdims=True))
        ll = raw.max(1)+np.log(weights.mean(1))
        se = weights.std(1, ddof=1)/np.sqrt(raw.shape[1])/weights.mean(1)
        expected = reference.loc[reference.method.eq(method) & reference.effort.eq("comparable")].sort_values("replicate")
        np.testing.assert_allclose(ll, expected.euler_loglik)
        seconds = trace.loc[trace.method.eq(method) & trace.effort.eq("comparable"), "elapsed_seconds"].max()
        rows.extend({"model": "dhaka", "method": "IFAD" if method == "IFAD-0.97" else method,
                     "start": i, "loglik": float(value), "mcse": float(se[i]),
                     "seconds": float(seconds), "status": "complete", "source": path,
                     "source_revision": "7dc0a6a^" if historical else "working archive",
                     "eval_reps": raw.shape[1]} for i, value in enumerate(ll))
    pd.DataFrame(rows).to_csv(Path(__file__).parent/"results/dhaka_reference.csv", index=False)


if __name__ == "__main__":
    main()
