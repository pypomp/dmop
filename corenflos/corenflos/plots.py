"""DMOP manuscript-style comparisons including Corenflos and Ditlevsen."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dmop-matplotlib")

import pandas as pd
import numpy as np
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
    geom_violin,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)


REFERENCE_METHODS = ["IFAD-0", "IFAD-0.97", "IFAD-1", "IF2"]
METHODS = [*REFERENCE_METHODS, "Ditlevsen", "Corenflos"]
COLORS = {
    "IFAD-0": "#440154",
    "IFAD-0.97": "#31688e",
    "IFAD-1": "#35b779",
    "IF2": "#fde725",
    "Ditlevsen": "#000000",
    "Corenflos": "#d95f02",
}
BEST_KNOWN = -3744.17
TRACE_LOWER_LIMIT = -4300.0
LIKELIHOOD_LOWER_LIMIT = -4300.0
REFERENCE_LIKELIHOOD_LOWER_LIMIT = -3800.0
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


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _raincloud(
    frame: pd.DataFrame,
    order: list[str],
    letter: str,
    destination: Path,
    *,
    lower_cutoff: float,
) -> None:
    frame = frame.loc[
        np.isfinite(frame["logLik"]) & frame["logLik"].ge(lower_cutoff)
    ].copy()
    if frame.empty:
        raise ValueError(f"no likelihoods above cutoff for panel {letter}")
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
    distribution = frame.loc[frame["Model"].isin(counts.loc[counts >= 2].index)]
    minimum = float(frame["logLik"].min())
    maximum = float(frame["logLik"].max())
    span = maximum - minimum
    if span <= 100:
        tick_step = 10
    elif span <= 1000:
        tick_step = 100
    elif span <= 2000:
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
        + geom_point(aes(x="Model_jitter_x"), alpha=0.6, size=1.5, show_legend=False)
        + geom_hline(
            yintercept=BEST_KNOWN,
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
        + scale_color_manual(values=COLORS)
        + scale_fill_manual(values=COLORS)
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
    plot.save(destination, width=7, height=3.8, dpi=220, verbose=False)


def plot_likelihoods(
    corenflos: pd.DataFrame,
    ditlevsen: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
) -> None:
    comparable = reference.loc[
        reference["effort"].eq("comparable"), ["method", "euler_loglik"]
    ].rename(columns={"method": "Model", "euler_loglik": "logLik"})
    ds = ditlevsen[["euler20_loglik"]].rename(columns={"euler20_loglik": "logLik"})
    ds["Model"] = "Ditlevsen"
    cf = corenflos[["euler20_loglik"]].rename(columns={"euler20_loglik": "logLik"})
    cf["Model"] = "Corenflos"
    panel_a = pd.concat((comparable, ds, cf), ignore_index=True)
    extended = reference.loc[
        reference["effort"].eq("extended"), ["method", "euler_loglik"]
    ].rename(columns={"method": "Model", "euler_loglik": "logLik"})
    left = output / "likelihood_comparable_r20.png"
    right = output / "likelihood_extended_reference.png"
    _raincloud(panel_a, METHODS, "A", left, lower_cutoff=LIKELIHOOD_LOWER_LIMIT)
    _raincloud(
        extended,
        REFERENCE_METHODS,
        "B",
        right,
        lower_cutoff=REFERENCE_LIKELIHOOD_LOWER_LIMIT,
    )
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
        combined.save(output / "likelihood_comparison_r20.png", dpi=(220, 220))
    left.unlink()
    right.unlink()


def plot_parameters(
    corenflos: pd.DataFrame,
    ditlevsen: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
) -> None:
    original = reference.loc[
        reference["effort"].eq("extended"), ["method", *PARAMETERS]
    ].rename(columns={"method": "source"})
    ds = ditlevsen[PARAMETERS].copy()
    ds["source"] = "Ditlevsen"
    cf = corenflos[PARAMETERS].copy()
    cf["source"] = "Corenflos"
    long = pd.concat((original, ds, cf), ignore_index=True).melt(
        id_vars="source",
        value_vars=PARAMETERS,
        var_name="quantity",
        value_name="param_value",
    )
    long = long.loc[np.isfinite(long["param_value"])].copy()
    long["quantity"] = long["quantity"].replace({"omegas5": "omega5"})
    long["source"] = pd.Categorical(long["source"], categories=METHODS, ordered=True)
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
        + scale_color_manual(values=COLORS, breaks=METHODS)
        + scale_fill_manual(values=COLORS, breaks=METHODS)
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
        output / "parameter_comparison_r20.png",
        width=9.5,
        height=6.0,
        dpi=220,
        verbose=False,
    )


def _trace_summary(traces: pd.DataFrame, label: str) -> pd.DataFrame:
    traces = traces.loc[np.isfinite(traces["euler20_loglik"])].copy()
    if traces.empty:
        return pd.DataFrame(
            columns=["elapsed_seconds", "median", "q10", "maximum", "source"]
        )
    maximum_iteration = int(traces["iteration"].max())
    iterations = np.arange(maximum_iteration + 1)
    carried = []
    for start, group in traces.groupby("start"):
        one = (
            group.sort_values("iteration")
            .drop_duplicates("iteration", keep="last")
            .set_index("iteration")[["elapsed_seconds", "euler20_loglik"]]
            .reindex(iterations)
            .ffill()
        )
        one["start"] = start
        one["iteration"] = iterations
        carried.append(one.reset_index(drop=True))
    frame = pd.concat(carried, ignore_index=True)
    summary = (
        frame.groupby("iteration")
        .agg(
            elapsed_seconds=("elapsed_seconds", "median"),
            median=("euler20_loglik", "median"),
            q10=("euler20_loglik", lambda values: values.quantile(0.10)),
            maximum=("euler20_loglik", "max"),
        )
        .reset_index()
    )
    summary["source"] = label
    return summary


def plot_trace(
    corenflos: pd.DataFrame,
    ditlevsen: pd.DataFrame,
    reference: pd.DataFrame,
    output: Path,
    budget: float,
) -> None:
    original = reference.loc[reference["effort"].eq("extended")].copy()
    original["source"] = original["method"]
    transitions = (
        original.loc[original["stage"].eq("train"), ["method", "elapsed_seconds"]]
        .sort_values("elapsed_seconds")
        .groupby("method", as_index=False)
        .first()
        .rename(columns={"method": "source"})
    )
    frame = pd.concat(
        (
            original[["elapsed_seconds", "median", "q10", "maximum", "source"]],
            _trace_summary(ditlevsen, "Ditlevsen"),
            _trace_summary(corenflos, "Corenflos"),
        ),
        ignore_index=True,
    )
    frame["source"] = pd.Categorical(frame["source"], categories=METHODS, ordered=True)
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
        + geom_line(
            aes(y="maximum"), data=non_if2, alpha=0.2, size=0.6, show_legend=False
        )
        + geom_line(aes(y="q10"), data=if2, alpha=0.35, size=0.6, show_legend=False)
        + geom_line(aes(y="maximum"), data=if2, alpha=0.35, size=0.6, show_legend=False)
        + geom_vline(
            aes(xintercept="elapsed_seconds", color="source"),
            data=transitions,
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
        + scale_color_manual(values=COLORS, breaks=METHODS)
        + scale_fill_manual(values=COLORS, breaks=METHODS)
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
        output / "optimization_elapsed_r20.png",
        width=7,
        height=3.8,
        dpi=220,
        verbose=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--ditlevsen",
        type=Path,
        default=Path("../ditlevsen/results/block_smc_guided_j100_final_100"),
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("../ditlevsen/results/reference")
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.data / "figures"
    output.mkdir(parents=True, exist_ok=True)
    corenflos_evaluations = _read(args.data / "final_evaluations.csv")
    corenflos_fits = _read(args.data / "fit_summary.csv")
    corenflos_traces = _read(args.data / "optimization_euler_traces.csv")
    ditlevsen_evaluations = _read(args.ditlevsen / "final_evaluations.csv")
    ditlevsen_fits = _read(args.ditlevsen / "fit_summary.csv")
    ditlevsen_traces = _read(args.ditlevsen / "optimization_euler_traces.csv")
    likelihood_reference = _read(args.reference / "manuscript_likelihood.csv")
    parameter_reference = _read(args.reference / "manuscript_parameters.csv")
    trace_reference = _read(args.reference / "manuscript_trace_summary.csv")
    configuration = pd.read_json(args.data / "configuration.json", typ="series")

    plot_likelihoods(
        corenflos_evaluations,
        ditlevsen_evaluations,
        likelihood_reference,
        output,
    )
    plot_parameters(
        corenflos_fits,
        ditlevsen_fits,
        parameter_reference,
        output,
    )
    plot_trace(
        corenflos_traces,
        ditlevsen_traces,
        trace_reference,
        output,
        float(configuration["maximum_elapsed_seconds"]),
    )


if __name__ == "__main__":
    main()
