"""Generate imgs/precise_table.tex from the result pickles.

The table was previously assembled by hand from the rendered report, which is
easy to get subtly wrong. The conventions it follows, preserved here:

* "Best log-likelihood" is the post-pruning evaluation (index=-1), a fresh
  unbiased evaluation of the single selected best replicate.
* "Median log-likelihood" is the median over all 100 replicates before
  pruning (index=-2).
* Both come from the N_MONITORS=1 runs; the times come from the N_MONITORS=0
  runs, which omit the per-iteration particle filter calls so that the
  comparison between methods is not distorted by monitoring overhead.

Usage: python make_table.py
"""

import pickle

import pandas as pd

RUNS = [
    ("IFAD-0.97", "dmop_results/dacca_results_rl4_alpha0.97_nm{nm}{sfx}.pkl"),
    ("IFAD-1", "dmop_results/dacca_results_rl4_alpha1.0_nm{nm}{sfx}.pkl"),
    ("IF2", "mif_results/dacca_results_rl4_nm{nm}{sfx}.pkl"),
    ("IFAD-0", "dmop_results/dacca_results_rl4_alpha0.0_nm{nm}{sfx}.pkl"),
]
BLOCKS = [("", "Comparable computational effort"), ("_long", "Extended computational effort")]
OUT = "../../imgs/precise_table.tex"


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


lines = [
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r" & Best log-likelihood & Median log-likelihood & IF2 time (s) & DMOP time (s) \\",
]

for sfx, heading in BLOCKS:
    lines += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\textit{{{heading}}}}} \\", r"\midrule"]
    rows = []
    for model, pat in RUNS:
        obj1 = _load(pat.format(nm=1, sfx=sfx))
        best = obj1.results(index=-1)["logLik"].max()
        median = obj1.results(index=-2)["logLik"].median()

        times = _load(pat.format(nm=0, sfx=sfx)).time()
        if2 = times.loc[times["method"] == "mif", "time"]
        dmop = times.loc[times["method"] == "train", "time"]
        rows.append(
            {
                "model": model,
                "best": best,
                "median": median,
                "if2": f"{round(if2.iloc[0]):d}" if len(if2) else "-",
                "dmop": f"{round(dmop.iloc[0]):d}" if len(dmop) else "-",
            }
        )
    for r in sorted(rows, key=lambda r: -r["best"]):
        lines.append(
            f"{r['model']} & {r['best']:.2f} & {r['median']:.2f} "
            f"& {r['if2']} & {r['dmop']} \\\\"
        )

lines += [r"\bottomrule", r"\end{tabular}"]

with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nwrote {OUT}")
pd.options.display.width = 120
