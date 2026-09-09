"""Manuscript-style figures for the SMC pseudo-score benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from plotnine import (
    aes,
    annotate,
    coord_cartesian,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_density,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_vline,
    geom_violin,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)


METHOD_ORDER = ["IFAD-0", "IFAD-0.97", "IFAD-1", "IF2"]
METHOD_COLORS = {
    "IFAD-0": "#440154",
    "IFAD-0.97": "#31688e",
    "IFAD-1": "#35b779",
    "IF2": "#fde725",
    "Ditlevsen": "#000000",
}
BEST_KNOWN = -3744.17
TRACE_LOWER_LIMIT = -3900
PARAMETERS = [
    "sigma",
    "tau",
    "gamma",
    "epsilon",
    "S_0",
    "I_0",
    "R1_0",
    "bs3",
    "omegas5",
]
PARAMETER_LABELS = {
    "sigma": r"Process noise ($\sigma$)",
    "tau": r"Measurement noise ($\tau$)",
    "gamma": r"Recovery rate ($\gamma$)",
    "epsilon": r"Immunity loss rate ($\epsilon$)",
    "S_0": r"Initial susceptible ($S_0$)",
    "I_0": r"Initial infected ($I_0$)",
    "R1_0": r"Initial recovered ($R_{1,0}$)",
    "bs3": r"Seasonality spline ($\beta_3$)",
    "omega5": r"Reservoir spline ($\omega_5$)",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _raincloud_panel(
    frame: pd.DataFrame,
    order: list[str],
    letter: str,
    filename: Path,
    *,
    lower_cutoff: float | None = -3780.0,
) -> None:
    keep = np.isfinite(frame["logLik"])
    if lower_cutoff is not None:
        keep &= frame["logLik"] >= lower_cutoff
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError(f"no finite likelihoods for panel {letter}")
    frame["Model"] = pd.Categorical(frame["Model"], categories=order, ordered=True)
    positions = {name: float(index) for index, name in enumerate(order)}
    frame["Model_num"] = frame["Model"].map(positions).astype(float)
    frame["Model_violin_x"] = frame["Model_num"] + 0.05
    frame["Model_boxplot_x"] = frame["Model_num"] - 0.10
    rng = np.random.RandomState(42)
    frame["Model_jitter_x"] = (
        frame["Model_num"] - 0.30 + rng.uniform(-0.04, 0.04, len(frame))
    )
    counts = frame.groupby("Model", observed=True).size()
    distribution_methods = counts.loc[counts >= 2].index
    distribution = frame.loc[frame["Model"].isin(distribution_methods)]
    minimum = float(frame["logLik"].min())
    maximum = float(frame["logLik"].max())
    span = maximum - minimum
    if span <= 100.0:
        tick_step = 10
    elif span <= 500.0:
        tick_step = 100
    elif span <= 2000.0:
        tick_step = 250
    else:
        tick_step = 500
    tick_minimum = int(np.floor(minimum / tick_step) * tick_step)
    tick_maximum = int(np.ceil(maximum / tick_step) * tick_step)
    ticks = list(range(tick_minimum, tick_maximum + tick_step, tick_step))
    plot = (
        ggplot(frame, aes(y="logLik", fill="Model", color="Model"))
        + geom_violin(
            aes(x="Model_violin_x", group="Model"),
            data=distribution,
            style="right",
            width=1.0,
            alpha=0.6,
            show_legend=False,
        )
        + geom_boxplot(
            aes(x="Model_boxplot_x", group="Model"),
            data=distribution,
            width=0.10,
            alpha=0.6,
            color="black",
            outlier_alpha=0,
            size=0.8,
            show_legend=False,
        )
        + geom_point(
            aes(x="Model_jitter_x"), alpha=0.6, size=1.5, show_legend=False
        )
        + geom_hline(
            yintercept=maximum,
            color="red",
            linetype="dotted",
            size=1.2,
            alpha=0.9,
        )
        + annotate(
            "text",
            x=len(order) - 0.6,
            y=minimum + max(2.0, 0.03 * span),
            label=letter,
            size=28,
            fontweight="bold",
            color="black",
        )
        + coord_flip()
        + scale_x_continuous(
            breaks=list(positions.values()),
            labels=order,
            limits=(-0.5, len(order) - 0.5),
        )
        + scale_y_continuous(breaks=ticks)
        + scale_color_manual(values=METHOD_COLORS)
        + scale_fill_manual(values=METHOD_COLORS)
        + labs(x="", y="Log-Likelihood")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(size=14, color="black"),
            axis_text_y=element_text(size=14, color="black"),
            axis_title_x=element_text(size=15, color="black"),
            axis_title_y=element_text(size=15, color="black"),
            panel_grid_major_y=element_blank(),
            panel_grid_minor_y=element_blank(),
            panel_grid_major_x=element_line(color="#e5e5e5", size=0.8),
            panel_grid_minor_x=element_blank(),
        )
    )
    plot.save(filename, width=7, height=3.8, dpi=220, verbose=False)


def plot_likelihoods(
    evaluations: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
    nstep: int,
) -> None:
    comparable_all = reference.loc[
        reference["effort"].eq("comparable"),
        ["method", "euler_loglik"],
    ].rename(columns={"method": "Model", "euler_loglik": "logLik"})
    ds = evaluations.loc[
        evaluations["inference_nstep"].eq(nstep), ["euler20_loglik"]
    ].rename(columns={"euler20_loglik": "logLik"})
    ds["Model"] = "Ditlevsen"
    panel_a = pd.concat([comparable_all, ds], ignore_index=True)
    extended_all = reference.loc[
        reference["effort"].eq("extended"),
        ["method", "euler_loglik"],
    ].rename(columns={"method": "Model", "euler_loglik": "logLik"})

    left = output / f"likelihood_comparable_r{nstep:02d}.png"
    right = output / "likelihood_extended_reference.png"
    _raincloud_panel(panel_a, [*METHOD_ORDER, "Ditlevsen"], "A", left)
    _raincloud_panel(extended_all, METHOD_ORDER, "B", right)
    with Image.open(left) as left_image, Image.open(right) as right_image:
        gap = 20
        combined = Image.new(
            "RGB",
            (
                left_image.width + gap + right_image.width,
                max(left_image.height, right_image.height),
            ),
            "white",
        )
        combined.paste(left_image.convert("RGB"), (0, 0))
        combined.paste(right_image.convert("RGB"), (left_image.width + gap, 0))
        combined.save(output / f"likelihood_comparison_r{nstep:02d}.png", dpi=(220, 220))

    comparable_display = comparable_all.loc[
        comparable_all["logLik"] >= TRACE_LOWER_LIMIT
    ]
    extended_display = extended_all.loc[
        extended_all["logLik"] >= TRACE_LOWER_LIMIT
    ]
    full_panel_a = pd.concat([comparable_display, ds], ignore_index=True)
    full_left = output / f"likelihood_comparable_full_r{nstep:02d}.png"
    full_right = output / "likelihood_extended_reference_full.png"
    _raincloud_panel(
        full_panel_a,
        [*METHOD_ORDER, "Ditlevsen"],
        "A",
        full_left,
        lower_cutoff=None,
    )
    _raincloud_panel(
        extended_display,
        METHOD_ORDER,
        "B",
        full_right,
        lower_cutoff=None,
    )
    with Image.open(full_left) as left_image, Image.open(
        full_right
    ) as right_image:
        gap = 20
        combined = Image.new(
            "RGB",
            (
                left_image.width + gap + right_image.width,
                max(left_image.height, right_image.height),
            ),
            "white",
        )
        combined.paste(left_image.convert("RGB"), (0, 0))
        combined.paste(
            right_image.convert("RGB"), (left_image.width + gap, 0)
        )
        combined.save(
            output / f"likelihood_comparison_full_r{nstep:02d}.png",
            dpi=(220, 220),
        )


def plot_parameters(
    fit_summary: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
    nstep: int,
) -> None:
    original = reference.loc[
        reference["effort"].eq("extended"), ["method", *PARAMETERS]
    ].rename(columns={"method": "source"})
    ditlevsen = fit_summary.loc[
        fit_summary["inference_nstep"].eq(nstep), [*PARAMETERS]
    ].copy()
    ditlevsen["source"] = "Ditlevsen"
    wide = pd.concat([original, ditlevsen], ignore_index=True)
    long = wide.melt(
        id_vars="source",
        value_vars=PARAMETERS,
        var_name="quantity",
        value_name="param_value",
    )
    long = long.loc[np.isfinite(long["param_value"])].copy()
    long["quantity"] = long["quantity"].replace({"omegas5": "omega5"})
    source_order = [*METHOD_ORDER, "Ditlevsen"]
    long["source"] = pd.Categorical(
        long["source"], categories=source_order, ordered=True
    )
    long["quantity_label"] = long["quantity"].map(PARAMETER_LABELS)
    long["quantity_label"] = pd.Categorical(
        long["quantity_label"],
        categories=list(PARAMETER_LABELS.values()),
        ordered=True,
    )
    plot = (
        ggplot(long, aes(x="param_value", fill="source", color="source"))
        + geom_density(alpha=0.4)
        + facet_wrap("quantity_label", scales="free", ncol=3)
        + scale_x_continuous(labels=lambda values: [f"{value:g}" for value in values])
        + scale_y_continuous(labels=lambda values: [f"{value:g}" for value in values])
        + scale_color_manual(values=METHOD_COLORS, breaks=source_order)
        + scale_fill_manual(values=METHOD_COLORS, breaks=source_order)
        + theme_minimal()
        + theme(
            axis_text_x=element_text(size=10, color="black"),
            axis_text_y=element_text(size=10, color="black"),
            axis_title_x=element_text(size=12, color="black"),
            axis_title_y=element_text(size=12, color="black"),
            strip_text=element_text(size=11, fontweight="bold", color="black"),
            legend_text=element_text(size=10),
            legend_title=element_blank(),
            legend_position="bottom",
            panel_spacing=0.02,
        )
        + labs(x="Parameter Value Estimate", y="Density")
    )
    plot.save(
        output / f"parameter_comparison_r{nstep:02d}.png",
        width=9.5,
        height=6.0,
        dpi=220,
        verbose=False,
    )


def _ditlevsen_trace_summary(
    traces: pd.DataFrame, nstep: int, budget: float
) -> pd.DataFrame:
    subset = traces.loc[
        traces["inference_nstep"].eq(nstep)
        & np.isfinite(traces["euler20_loglik"])
    ].copy()
    grid = np.arange(0.0, budget + 0.1, 100.0)
    rows: list[dict[str, float]] = []
    for start, group in subset.groupby("start"):
        group = group.sort_values("elapsed_seconds")
        for elapsed in grid:
            distance = np.abs(group["elapsed_seconds"].to_numpy() - elapsed)
            row = group.iloc[int(np.argmin(distance))]
            rows.append(
                {
                    "start": float(start),
                    "elapsed_seconds": elapsed,
                    "euler20_loglik": float(row["euler20_loglik"]),
                }
            )
    carried = pd.DataFrame(rows)
    summary = (
        carried.groupby("elapsed_seconds")["euler20_loglik"]
        .agg(
            median="median",
            q10=lambda values: values.quantile(0.10),
            maximum="max",
        )
        .reset_index()
    )
    summary["source"] = "Ditlevsen"
    return summary


def plot_trace(
    traces: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
    nstep: int,
    budget: float,
) -> None:
    original = reference.loc[reference["effort"].eq("comparable")].copy()
    original["source"] = original["method"]
    transition = (
        original.loc[
            original["stage"].eq("train"), ["method", "elapsed_seconds"]
        ]
        .sort_values("elapsed_seconds")
        .groupby("method", as_index=False)
        .first()
        .rename(columns={"method": "source"})
    )
    ds = _ditlevsen_trace_summary(traces, nstep, budget)
    frame = pd.concat(
        [
            original[["elapsed_seconds", "median", "q10", "maximum", "source"]],
            ds,
        ],
        ignore_index=True,
    )
    source_order = [*METHOD_ORDER, "Ditlevsen"]
    frame["source"] = pd.Categorical(
        frame["source"], categories=source_order, ordered=True
    )
    non_if2 = frame.loc[frame["source"].ne("IF2")]
    if2 = frame.loc[frame["source"].eq("IF2")]
    plot = (
        ggplot(frame, aes(x="elapsed_seconds", y="median", color="source"))
        + geom_ribbon(
            aes(ymin="q10", ymax="maximum", fill="source"),
            alpha=0.10,
            color=None,
            show_legend=False,
        )
        + geom_line(size=1.2)
        + geom_line(aes(y="q10"), data=non_if2, alpha=0.2, size=0.6, show_legend=False)
        + geom_line(aes(y="maximum"), data=non_if2, alpha=0.2, size=0.6, show_legend=False)
        + geom_line(aes(y="q10"), data=if2, alpha=0.35, size=0.6, show_legend=False)
        + geom_line(aes(y="maximum"), data=if2, alpha=0.35, size=0.6, show_legend=False)
        + geom_vline(
            aes(xintercept="elapsed_seconds", color="source"),
            data=transition,
            inherit_aes=False,
            linetype="dotted",
            size=1.0,
            show_legend=False,
        )
        + geom_hline(
            yintercept=BEST_KNOWN,
            color="red",
            linetype="dotted",
            size=1.2,
            alpha=0.9,
        )
        + scale_color_manual(values=METHOD_COLORS, breaks=source_order)
        + scale_fill_manual(values=METHOD_COLORS, breaks=source_order)
        + labs(x="Elapsed Time (Seconds)", y="Log-Likelihood")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(size=14, color="black"),
            axis_text_y=element_text(size=14, color="black"),
            axis_title_x=element_text(size=15, color="black"),
            axis_title_y=element_text(size=15, color="black"),
            legend_text=element_text(size=12, color="black"),
            legend_title=element_blank(),
            legend_position="right",
            panel_grid_major=element_line(color="#e5e5e5", size=0.8),
            panel_grid_minor=element_blank(),
        )
    )
    (plot + coord_cartesian(xlim=(0, budget), ylim=(TRACE_LOWER_LIMIT, None))).save(
        output / f"optimization_elapsed_r{nstep:02d}.png",
        width=7,
        height=3.8,
        dpi=220,
        verbose=False,
    )
    (plot + coord_cartesian(xlim=(0, budget), ylim=(-5600.0, None))).save(
        output / f"optimization_elapsed_full_r{nstep:02d}.png",
        width=7,
        height=3.8,
        dpi=220,
        verbose=False,
    )


def plot_substeps(evaluations: pd.DataFrame, output: Path) -> None:
    frame = evaluations.loc[np.isfinite(evaluations["euler20_loglik"])].copy()
    sns.set_theme(style="whitegrid", context="paper")
    figure, axis = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    sns.boxplot(
        data=frame,
        x="inference_nstep",
        y="euler20_loglik",
        color="white",
        width=0.25,
        fliersize=0,
        linewidth=1.2,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="inference_nstep",
        y="euler20_loglik",
        color="black",
        alpha=0.55,
        size=4,
        jitter=0.12,
        ax=axis,
    )
    axis.axhline(BEST_KNOWN, color="red", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Process substeps per month")
    axis.set_ylabel("Log-Likelihood")
    figure.savefig(output / "substep_likelihood.png", dpi=220)
    plt.close(figure)


def write_captions(output: Path, count: int, budget: float) -> None:
    text = rf"""% Captions for the PNG benchmark figures.

\caption{{Raincloud plots of the Dacca searches. \textbf{{A}}, comparable
computational effort, with {count} Ditlevsen fits limited to {budget:g} seconds.
\textbf{{B}}, the manuscript's extended-effort reference runs; Ditlevsen was
not run at the extended budget. Each Ditlevsen search contributes the iterate
with the highest block pseudo-log-likelihood. Log likelihoods below $-3780$
are omitted.}}

\caption{{Expanded raincloud comparison showing all {count} Ditlevsen pilot
fits and manuscript fits above $-3900$.}}

\caption{{Final estimates for nine Dacca parameters. The original methods use
the manuscript's extended-effort runs; each Ditlevsen density contains {count}
bounding-box starts limited to {budget:g} seconds and selected by block
pseudo-log-likelihood.}}

\caption{{Optimization progress against elapsed time at comparable
computational effort. The manuscript's lower display limit is relaxed from
$-3800$ to $-3900$ to show the Ditlevsen pilot. Runs that terminate early are
held at their final estimate.}}

\caption{{The same elapsed-time comparison with a $-5600$ lower limit, showing
the Ditlevsen median and its 10th-percentile-to-maximum envelope.}}

\caption{{Independently evaluated Euler-20 log-likelihoods after fitting with
5, 10, or 20 process substeps per month.}}
"""
    (output / "figure_captions.tex").write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/smc_benchmark_10"))
    parser.add_argument("--reference", type=Path, default=Path("results/reference"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.data / "figures"
    output.mkdir(parents=True, exist_ok=True)
    selected_path = args.data / "selected_evaluations.csv"
    if selected_path.exists():
        evaluations = _read_csv(selected_path)
        fit_summary = evaluations
    else:
        evaluations = _read_csv(args.data / "final_evaluations.csv")
        fit_summary = _read_csv(args.data / "fit_summary.csv")
    traces = _read_csv(args.data / "optimization_euler_traces.csv")
    likelihood = _read_csv(args.reference / "manuscript_likelihood.csv")
    parameters = _read_csv(args.reference / "manuscript_parameters.csv")
    reference_traces = _read_csv(args.reference / "manuscript_trace_summary.csv")
    configuration = pd.read_json(args.data / "configuration.json", typ="series")
    nsteps = sorted(evaluations["inference_nstep"].unique())
    for nstep in nsteps:
        plot_likelihoods(evaluations, likelihood, output, int(nstep))
        plot_parameters(fit_summary, parameters, output, int(nstep))
        plot_trace(
            traces,
            reference_traces,
            output,
            int(nstep),
            float(configuration["maximum_elapsed_seconds"]),
        )
    plot_substeps(evaluations, output)
    write_captions(
        output,
        int(configuration["starts"]),
        float(configuration["maximum_elapsed_seconds"]),
    )


if __name__ == "__main__":
    main()
