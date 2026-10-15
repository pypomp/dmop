"""Audit the archived Figure 6 chains; do not refit or claim a common target."""
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = [r"\log\gamma", r"\log m", r"\log\rho", r"\log\epsilon",
          r"\log\omega", r"\operatorname{logit}c", r"100\beta_{\mathrm{trend}}",
          r"\log\sigma", r"\log\tau"] + [rf"\beta_{i}" for i in range(1, 7)] + [rf"\omega_{i}" for i in range(1, 7)]


def main():
    provenance = json.loads((HERE / "provenance.json").read_text())
    for name, digest in provenance["files"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived input changed: {name}")
    chains = np.stack([np.load(HERE / "data" / f"theta_{i}.npy") for i in range(4)])
    assert chains.shape == (4, 1500, 21)
    assert np.isneginf(chains[:, :, 2]).all() and np.isposinf(chains[:, :, 5]).all()
    draws = chains[:, provenance["burnin"]:]
    rows, single = [], []
    for j, label in enumerate(LABELS):
        if j in provenance["fixed_columns"]:
            continue
        x = draws[:, :, j].astype(float)
        assert np.isfinite(x).all()
        per_chain = []
        for c in range(4):
            row = {"coordinate": j, "label": label, "chain": c,
                   "mean": float(x[c].mean()), "sd": float(x[c].std(ddof=1)),
                   "ess_bulk": float(az.ess(x[c:c+1], method="bulk")),
                   "ess_tail": float(az.ess(x[c:c+1], method="tail")),
                   "mcse_mean": float(az.mcse(x[c:c+1], method="mean"))}
            single.append(row)
            per_chain.append(row)
        rows.append({"coordinate": j, "label": label,
                     "rhat_descriptive": float(az.rhat(x, method="rank")),
                     "ess_bulk_min": min(r["ess_bulk"] for r in per_chain),
                     "ess_tail_min": min(r["ess_tail"] for r in per_chain),
                     "mcse_mean_max": max(r["mcse_mean"] for r in per_chain)})
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(single).to_csv(out / "per_chain.csv", index=False)
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Coordinate & $\widehat R$ & Bulk ESS & Tail ESS & MCSE \\", r"\midrule"]
    for r in rows:
        tex.append(f"${r['label']}$ & {r['rhat_descriptive']:.2f} & {r['ess_bulk_min']:.1f} & {r['ess_tail_min']:.1f} & {r['mcse_mean_max']:.3f} " + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "diagnostics.tex").write_text("\n".join(tex) + "\n")
    record = {"arviz": az.__version__, "numpy": np.__version__,
              "draws": list(draws.shape), "diagnosed_coordinates": len(rows),
              "rhat_range": [min(r["rhat_descriptive"] for r in rows), max(r["rhat_descriptive"] for r in rows)],
              "common_target": False,
              "rhat_interpretation": "descriptive between-run disagreement; priors differ by chain",
              "ess_mcse_interpretation": "within-chain estimates, requiring stationarity; no pooling",
              "unavailable": ["prior clouds and their covariance matrices", "divergences", "tree-depth saturation", "timings and ESS per second", "original tracing-time random seeds"],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
