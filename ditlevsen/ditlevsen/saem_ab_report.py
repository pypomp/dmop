"""Report a complete paired Dhaka score/SAEM experiment on linear axes."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

LABELS = {"score": "Score updates", "saem": "Numerical SAEM"}
COLORS = {"score": "#767676", "saem": "#31688e"}


def paired_mcse(a, b):
    """Delta-method MC error for log(mean L_b)-log(mean L_a), with paired draws."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired likelihood replicates required")
    residual = np.exp(b-logsumexp(b))*len(b)-np.exp(a-logsumexp(a))*len(a)
    return float(residual.std(ddof=1)/np.sqrt(len(a)))


def generate(root):
    # More than one worker can finish at once; only one writes the report.
    with (root/"report.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return _generate(root)


def _generate(root):
    config = json.loads((root/"protocol.json").read_text())
    directories = [root/regime/f"start{i:03d}" for regime in ("cold", "warm")
                   for i in range(config["starts"])]
    if any(not (p/"status.json").exists() or not json.loads((p/"status.json").read_text())["complete"]
           for p in directories):
        print("A/B report pending the remaining complete pairs", flush=True)
        return False
    rows, pairs, inputs = [], [], []
    successful = {"completed", "time-budget", "maximum-iterations"}
    for directory in directories:
        frame = pd.read_csv(directory/"evaluation.csv").set_index("method")
        if set(frame.index) != {"initial", "score", "saem"} or len(frame) != 3:
            raise ValueError(f"Incomplete evaluations in {directory}")
        if not np.isfinite(frame.loglik).all():
            raise ValueError(f"Unresolved evaluation in {directory}")
        draws = pd.read_csv(directory/"evaluation_replicates.csv")
        raw = draws.pivot(index="replicate", columns="method", values="loglik").sort_index()
        if len(raw) != config["eval_replicates"] or not np.isfinite(raw.to_numpy()).all():
            raise ValueError(f"Invalid evaluation replicates in {directory}")
        inputs.extend([directory/"evaluation.csv", directory/"evaluation_replicates.csv"])
        first = frame.loc["initial"]
        pairs.append(dict(regime=first.regime, start=int(first.start),
            score=float(frame.loc["score", "loglik"]), saem=float(frame.loc["saem", "loglik"]),
            difference=float(frame.loc["saem", "loglik"]-frame.loc["score", "loglik"]),
            mcse=paired_mcse(raw.score.to_numpy(), raw.saem.to_numpy())))
        for method in ("score", "saem"):
            meta_path = directory/f"{method}.json"
            meta = json.loads(meta_path.read_text())
            inputs.append(meta_path)
            rows.append(dict(regime=first.regime, start=int(first.start), method=method,
                loglik=float(frame.loc[method, "loglik"]), mcse=float(frame.loc[method, "mcse"]),
                change=float(frame.loc[method, "loglik"]-first.loglik),
                updates=meta["updates"], seconds=meta["seconds"],
                failed=meta["status"] not in successful, status=meta["status"]))
    final, paired = pd.DataFrame(rows), pd.DataFrame(pairs)
    output = root/"report"
    output.mkdir(exist_ok=True)
    final.to_csv(output/"final.csv", index=False)
    paired.to_csv(output/"paired.csv", index=False)
    summary = final.groupby(["regime", "method"], sort=False).agg(
        starts=("start", "size"), median=("loglik", "median"), maximum=("loglik", "max"),
        median_change=("change", "median"), median_mcse=("mcse", "median"),
        median_updates=("updates", "median"), median_seconds=("seconds", "median"), failed=("failed", "sum"))
    summary.to_csv(output/"summary.csv")
    paired.groupby("regime").agg(starts=("start", "size"), median_difference=("difference", "median"),
        mean_difference=("difference", "mean"), saem_higher=("difference", lambda x: int((x > 0).sum())),
        median_mcse=("mcse", "median")).to_csv(output/"paired_summary.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), layout="constrained")
    for j, regime in enumerate(("cold", "warm")):
        subset = paired.loc[paired.regime.eq(regime)].sort_values("start")
        ax = axes[0, j]
        for row in subset.itertuples():
            ax.plot([0, 1], [row.score, row.saem], color=".8", lw=.7, zorder=1)
        for x, method in enumerate(("score", "saem")):
            f = final.loc[final.regime.eq(regime) & final.method.eq(method)].sort_values("start")
            good = ~f.failed
            ax.scatter(np.full(good.sum(), x), f.loc[good, "loglik"], s=22,
                       color=COLORS[method], alpha=.8, zorder=2)
            ax.scatter(np.full((~good).sum(), x), f.loc[~good, "loglik"], s=35,
                       color=COLORS[method], marker="x", zorder=3)
        ax.set_xticks([0, 1], list(LABELS.values()))
        ax.set_xlim(-.3, 1.3)
        ax.set_ylabel("Final log-likelihood")
        ax.set_title(f"({'AB'[j]}) "+("Global starts" if regime == "cold" else "IF2 warm starts"), loc="left")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(axis="y", color=".92")
        ax = axes[1, j]
        ax.errorbar(subset.start, subset.difference, yerr=2*subset.mcse,
                    fmt="o", color=COLORS["saem"], ms=4, capsize=2, lw=1.)
        ax.axhline(0, color=".4", lw=.8, ls="--")
        ax.set_xlabel("Starting point")
        ax.set_ylabel("Log-likelihood: SAEM minus score")
        ax.set_title(f"({'CD'[j]}) Paired differences", loc="left")
        ax.grid(axis="y", color=".92")
    for ext in ("pdf", "png"):
        fig.savefig(output/f"comparison.{ext}", dpi=220)
    plt.close(fig)
    table = [r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Initialization & Optimizer & Median $\ell$ & Median change & Updates & Time (s) & Failed \\", r"\midrule"]
    for (regime, method), row in summary.iterrows():
        table.append(f"{'Global' if regime == 'cold' else 'IF2'} & {LABELS[method]} & "
            f"{row['median']:.2f} & {row.median_change:.2f} & {row.median_updates:.0f} & "
            f"{row.median_seconds:.1f} & {row.failed:.0f} "+r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (output/"table.tex").write_text("\n".join(table)+"\n")
    (output/"README.md").write_text(
        "# Dhaka optimizer A/B comparison\n\n"
        f"{config['starts']} paired starts per initialization regime; "
        f"{config['particles']} fitting particles and {config['budget_seconds']:g} seconds per fit. "
        "Both methods use the same monthly Gaussian transition and guided smoother. "
        "Score updates retain the archived settings, including burn-in 30; numerical SAEM "
        f"uses burn-in {config['saem_burnin']} and at most {config['mstep_iterations']} L-BFGS iterations per M-step. "
        "This compares the two optimizer packages, including their schedules.\n\n"
        "All likelihood axes are linear. Lines in the top panels join estimates from the same start; "
        "crosses mark prematurely stopped fits, retaining their last eligible estimate. "
        "The bottom panels show SAEM minus score, with twice the paired Monte Carlo standard error. "
        "Evaluation draws are shared within each pair and independent of fitting. "
        "No fitting output is selected using these evaluations.\n\n"
        "Median change subtracts each fit's own initial log-likelihood. Updates counts completed "
        "updates at the selected estimate. Time records the fitting call, including an overrun "
        "needed to detect the deadline; over-budget estimates are discarded. Compilation, final "
        "evaluation, and the cost of the precomputed IF2 starting points are excluded.\n")
    (output/"provenance.json").write_text(json.dumps({
        "inputs": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "protocol": config, "axis_scale": "linear"}, indent=2)+"\n")
    print(summary.to_string(), flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    generate(parser.parse_args().root)
