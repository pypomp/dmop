"""Build cross-model panels and a table from completed, independently evaluated fits.

The default requires all 20 starts of each new comparison and the archived
100-start Dhaka study. It never fits models or chooses optimization outputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["oscillator", "linear", "spx", "daphnia", "dhaka"]
TITLES = {"oscillator": "Harmonic oscillator", "linear": "Linear Gaussian",
          "spx": "SPX", "daphnia": "Daphnia", "dhaka": "Dhaka"}
METHODS = ["IFAD", "IF2", "DS19", "CTDD21"]
COLORS = {"IFAD": "#31688e", "IF2": "#b59c00", "DS19": "#292929", "CTDD21": "#d95f02"}


def new_results(root, model, expected=20):
    folders = sorted(root.glob(f"final-{model}*"))
    if not folders:
        raise ValueError(f"No final results for {model}")
    frames, timings, sources, reference = [], [], [], None
    for folder in folders:
        config = json.loads((folder / "configuration.json").read_text())
        status = json.loads((folder / "status.json").read_text())
        if config["purpose"] != "final" or not status["complete"]:
            raise ValueError(f"Unfinished or pilot run: {folder}")
        frames.append(pd.read_csv(folder / "checkpoints.csv"))
        timings.append(pd.read_csv(folder / "timings.csv"))
        sources.extend([folder / "configuration.json", folder / "checkpoints.csv", folder / "timings.csv"])
        if model in ("linear", "oscillator"):
            analytic = json.loads((folder / "analytic_reference.json").read_text())
            if not analytic["converged"]:
                raise ValueError(f"Unconverged analytic reference: {folder}")
            if reference is not None and abs(reference-analytic["loglik"]) > 1e-5:
                raise ValueError("Analytic references disagree")
            reference = analytic["loglik"]
    trace = pd.concat(frames, ignore_index=True).replace({"method": {"MPIF": "IF2"}})
    timing = pd.concat(timings, ignore_index=True).replace({"method": {"MPIF": "IF2"}})
    final = trace.loc[trace.final].copy()
    for method in METHODS:
        ids = sorted(final.loc[final.method.eq(method), "start"])
        if ids != list(range(expected)):
            raise ValueError(f"Missing/duplicate starts for {model}/{method}: {ids}")
    if timing.duplicated(["start", "method"]).any() or len(timing) != 4*expected:
        raise ValueError(f"Invalid timing records for {model}")
    if not np.isfinite(final.loglik).all():
        raise ValueError(f"Unresolved final likelihood evaluations in {model}; report and resolve before plotting")
    final = final.drop(columns="seconds").merge(
        timing[["start", "method", "seconds"]], on=["start", "method"], validate="one_to_one")
    if reference is None:
        reference = float(final.loglik.max())
    elif final.loglik.max() > reference + 1e-4:
        raise ValueError(f"Fit exceeds analytic reference for {model}")
    final["model"] = model
    progress = []
    for method, frame in trace.groupby("method", sort=False):
        grid = np.linspace(0., frame.seconds.max(), 201)
        values = []
        for _, run in frame.groupby("start"):
            run = run.sort_values("seconds").drop_duplicates("seconds", keep="last")
            values.append(np.interp(grid, run.seconds, run.loglik))
        q10, median, q90 = np.quantile(values, [.1, .5, .9], axis=0)
        progress.append(pd.DataFrame({"model": model, "method": method, "seconds": grid,
                                      "median": median, "q10": q10, "q90": q90}))
    return final, pd.concat(progress), reference, sources


def dhaka_results():
    folder = ROOT / "imgs/competitors/corenflos100_ditlevsen1000"
    reference_folder = ROOT / "ditlevsen/results/reference"
    sources = [folder/"final_all_runs.csv", folder/"optimization_summary.csv",
               reference_folder/"manuscript_likelihood.csv", reference_folder/"manuscript_trace_summary.csv",
               ROOT/"code/benchmarks/results/dhaka_reference.csv"]
    finals = pd.read_csv(sources[0])
    selected = {"DS19 + IF2 warm start (J=1,000)": "DS19",
                "CTDD21 + IF2 warm start (J=100)": "CTDD21"}
    final = finals.loc[finals.method.isin(selected)].rename(
        columns={"euler20_loglik": "loglik", "euler20_se": "mcse"}).copy()
    final["method"] = final.method.map(selected)
    final["seconds"] = 942.8173823356628  # Declared output-selection budget, including warm start.
    if not final.evaluation_status.eq("ok").all():
        raise ValueError("Unresolved archived Dhaka evaluation")
    final["status"] = "complete"
    for method, path in (("CTDD21", ROOT/"corenflos/results/if2warm_ifad097_budget_j100_final_100/fit_summary.csv"),
                         ("DS19", ROOT/"corenflos/results/particle_increase_j1000_final_100/ditlevsen_warm/fit_summary.csv")):
        fits = pd.read_csv(path).set_index("start")
        sources.append(path)
        selected_rows = final.method.eq(method)
        reasons = final.loc[selected_rows, "start"].map(fits.termination_reason)
        final.loc[selected_rows, "status"] = np.where(reasons.isin(["time-budget", "maximum-iterations"]),
                                                       "complete", reasons)
    baseline = pd.read_csv(sources[4])
    columns = ["start", "method", "loglik", "mcse", "seconds", "status"]
    final = pd.concat([final[columns], baseline[columns]], ignore_index=True).assign(model="dhaka")
    for method in METHODS:
        if sorted(final.loc[final.method.eq(method), "start"]) != list(range(100)):
            raise ValueError(f"Incomplete Dhaka archive: {method}")
    # Verify baseline values against the archive used by the existing S11 figures.
    old = pd.read_csv(sources[2])
    for method, name in (("IFAD", "IFAD-0.97"), ("IF2", "IF2")):
        a = final.loc[final.method.eq(method)].sort_values("start").loglik
        b = old.loc[old.method.eq(name) & old.effort.eq("comparable")].sort_values("replicate").euler_loglik
        np.testing.assert_allclose(a, b)
    trace = pd.read_csv(sources[1])
    trace = trace.loc[trace.method.isin(selected)].copy()
    trace["method"] = trace.method.map(selected)
    old_trace = pd.read_csv(sources[3])
    old_trace = old_trace.loc[old_trace.effort.eq("comparable")
                             & old_trace.method.isin(["IFAD-0.97", "IF2"])].copy()
    old_trace["method"] = old_trace.method.replace({"IFAD-0.97": "IFAD"})
    trace = pd.concat([trace, old_trace]).rename(columns={"elapsed_seconds": "seconds"})
    # The archive stores q10 and maximum, not q90. Plot median curves only.
    trace["q90"] = np.nan
    trace["model"] = "dhaka"
    return final, trace, float(final.loglik.max()), sources


def axes_for(models):
    if len(models) == 5:
        fig = plt.figure(figsize=(11.4, 6.4), layout="constrained")
        grid = fig.add_gridspec(2, 6)
        axes = [fig.add_subplot(grid[0, 2*i:2*i+2]) for i in range(3)]
        axes += [fig.add_subplot(grid[1, 3*i:3*i+3]) for i in range(2)]
    else:
        fig, axes = plt.subplots(1, len(models), figsize=(3.8*len(models), 3.4), squeeze=False,
                                 layout="constrained")
        axes = list(axes.flat)
    return fig, axes


def save(fig, output, name):
    for extension in ("pdf", "png"):
        fig.savefig(output/f"{name}.{extension}", dpi=220,
                    metadata={"CreationDate": None} if extension == "pdf" else None)
    plt.close(fig)


def generate(results, output, models, expected):
    all_final, all_progress, references, sources = [], [], {}, []
    for model in models:
        final, progress, ref, paths = dhaka_results() if model == "dhaka" else new_results(results, model, expected)
        all_final.append(final)
        all_progress.append(progress)
        references[model] = ref
        sources.extend(paths)
    final, progress = pd.concat(all_final), pd.concat(all_progress)
    output.mkdir(parents=True, exist_ok=True)
    final.to_csv(output/"final_results.csv", index=False)
    progress.to_csv(output/"progress_summary.csv", index=False)
    (output/"provenance.json").write_text(json.dumps({"references": references,
        "reference_definition": "analytic maximum for Gaussian models; best displayed final estimate otherwise",
        "inputs": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                   hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2)+"\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "pdf.fonttype": 42})
    fig, axes = axes_for(models)
    for i, (ax, model) in enumerate(zip(axes, models)):
        for j, method in enumerate(METHODS):
            frame = final.loc[final.model.eq(model) & final.method.eq(method)]
            deficits = references[model]-frame.loglik.to_numpy()
            bp = ax.boxplot([deficits], positions=[j], widths=.38, patch_artist=True,
                            showfliers=False, medianprops={"color": "black", "linewidth": 1.4})
            bp["boxes"][0].set(facecolor=COLORS[method], alpha=.22)
            jitter = np.random.default_rng(719+i*4+j).uniform(-.14, .14, len(frame))
            ax.scatter(j+jitter, deficits, s=13 if len(frame) == 20 else 7,
                       color=COLORS[method], alpha=.7, linewidths=0, zorder=3)
        ax.axhline(0., color=".65", lw=.7, ls="--")
        ax.set_yscale("symlog", linthresh=1.)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_xticks(range(4), ["IFAD", "MPIF" if model == "daphnia" else "IF2", "DS19", "CTDD21"])
        ax.set_ylabel("Log-likelihood deficit")
        ax.set_title(f"({chr(65+i)}) {TITLES[model]}", loc="left")
        ax.grid(axis="y", color=".92", zorder=0)
    save(fig, output, "final_likelihoods")

    fig, axes = axes_for(models)
    for i, (ax, model) in enumerate(zip(axes, models)):
        for method in METHODS:
            frame = progress.loc[progress.model.eq(model) & progress.method.eq(method)].sort_values("seconds")
            ax.plot(frame.seconds, references[model]-frame["median"], color=COLORS[method], lw=1.6)
        ax.set_yscale("symlog", linthresh=1.)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.axhline(0., color=".65", lw=.7, ls="--")
        ax.set_xlabel("Fitting time (s)")
        ax.set_ylabel("Median log-likelihood deficit")
        ax.set_title(f"({chr(65+i)}) {TITLES[model]}", loc="left")
        ax.grid(color=".92")
    fig.legend(handles=[Line2D([], [], color=COLORS[m], lw=2, label="IF2 / MPIF" if m == "IF2" else m)
                        for m in METHODS], loc="outside lower center", ncols=4, frameon=False)
    save(fig, output, "optimization")

    summary = []
    table = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & Median & Maximum & IQR & MCSE & Time (s) & Failed \\", r"\midrule"]
    for model in models:
        n = int(final.loc[final.model.eq(model)].groupby("method").size().iloc[0])
        table.append(r"\multicolumn{7}{l}{\textit{"+TITLES[model]+f" ({n} starts)"+r"}} \\")
        for method in METHODS:
            f = final.loc[final.model.eq(model) & final.method.eq(method)]
            row = {"model": model, "method": method, "starts": len(f), "median": f.loglik.median(),
                   "maximum": f.loglik.max(), "iqr": f.loglik.quantile(.75)-f.loglik.quantile(.25),
                   "median_mcse": f.mcse.median(), "median_seconds": f.seconds.median(),
                   "failed": int(f.status.ne("complete").sum())}
            summary.append(row)
            label = "MPIF" if model == "daphnia" and method == "IF2" else method
            table.append(f"{label} & {row['median']:.2f} & {row['maximum']:.2f} & {row['iqr']:.2f} & "
                         f"{row['median_mcse']:.2f} & {row['median_seconds']:.1f} & {row['failed']} "+r"\\")
        if model != models[-1]:
            table.append(r"\addlinespace")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (output/"results_table.tex").write_text("\n".join(table)+"\n")
    pd.DataFrame(summary).to_csv(output/"summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent/"results")
    parser.add_argument("--output", type=Path, default=ROOT/"imgs/benchmarks")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--expected-starts", type=int, default=20)
    args = parser.parse_args()
    generate(args.results, args.output, args.models, args.expected_starts)
