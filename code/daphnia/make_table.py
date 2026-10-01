"""Generate imgs/daphnia/daphnia_table.tex from the portable result tables.

Maximum and median use all 50 original final evaluations. Re-evaluated uses
fresh particle filters at each method's original maximizing parameter vector.
Times sum the 25 batches and exclude particle filter evaluations.

Usage: python make_table.py
"""

import argparse
from pathlib import Path

import pandas as pd

RUNS = ["IFAD-0.97", "IFAD-1", "IFAD-0", "MPIF"]


def main():
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=directory / "exports")
    parser.add_argument("--output", type=Path,
                        default=directory / "../../imgs/daphnia/daphnia_table.tex")
    args = parser.parse_args()
    final = pd.read_csv(args.exports / "final_estimates.csv", float_precision="round_trip")
    timing = pd.read_csv(args.exports / "timings.csv", float_precision="round_trip")
    timing = timing.loc[timing["method"].isin(["mif", "train"])]
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & Maximum & Re-evaluated & Median & MPIF time (s) & DMOP time (s) \\", r"\midrule"]
    for model in RUNS:
        values = final.loc[final["model"] == model]
        best = values.sort_values(["logLik", "rep"], ascending=[False, True]).iloc[0]
        fresh = values.loc[values["reevaluated_logLik"].notna()]
        if len(fresh) != 1 or fresh.iloc[0]["rep"] != best["rep"]:
            raise ValueError(f"{model}: expected one re-evaluation at the best estimate")
        fresh = fresh.iloc[0]
        seconds = timing.loc[timing["model"] == model].groupby("method")["seconds"].sum()
        mif = f"{round(seconds['mif']):,d}"
        dmop = f"{round(seconds['train']):,d}" if "train" in seconds else "-"
        lines.append(
            f"{model} & ${best['logLik']:.2f}\\;({best['MCSE']:.2f})$ "
            f"& ${fresh['reevaluated_logLik']:.2f}\\;({fresh['reevaluated_MCSE']:.2f})$ "
            f"& ${values['logLik'].median():.2f}$ & {mif} & {dmop} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
