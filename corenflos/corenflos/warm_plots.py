"""IF2-warm-start comparison figures for Corenflos and Ditlevsen."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dmop-matplotlib")

import numpy as np
import pandas as pd
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

from ditlevsen.benchmark import _parameter_columns
from ditlevsen.warm_starts import load_starts_file

from .plots import BEST_KNOWN, PARAMETER_LABELS, PARAMETERS, _trace_summary


OFFSET = 92.16042757034302
TOTAL = 942.8173823356628
METHODS = [
    "Corenflos",
    "Corenflos + IF2",
    "Ditlevsen",
    "Ditlevsen + IF2",
    "IF2 checkpoint",
    "IFAD-0.97",
]
COLORS = {
    "Corenflos": "#e6ab78",
    "Corenflos + IF2": "#d95f02",
    "Ditlevsen": "#969696",
    "Ditlevsen + IF2": "#000000",
    "IF2 checkpoint": "#e6c700",
    "IF2 warm start": "#e6c700",
    "IFAD-0.97": "#31688e",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _likelihood_frame(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    sources = (
        (args.corenflos_cold / "final_evaluations.csv", "Corenflos"),
        (args.corenflos_warm / "final_evaluations.csv", "Corenflos + IF2"),
        (args.ditlevsen_cold / "final_evaluations.csv", "Ditlevsen"),
        (args.ditlevsen_warm / "final_evaluations.csv", "Ditlevsen + IF2"),
        (args.warm_start / "warm_start_evaluations.csv", "IF2 checkpoint"),
    )
    for path, label in sources:
        frame = _read(path)[["euler20_loglik"]].rename(
            columns={"euler20_loglik": "logLik"}
        )
        frame["Model"] = label
        rows.append(frame)
    reference = _read(args.reference / "manuscript_likelihood.csv")
    ifad = reference.loc[
        reference["method"].eq("IFAD-0.97") & reference["effort"].eq("comparable"),
        ["euler_loglik"],
    ].rename(columns={"euler_loglik": "logLik"})
    ifad["Model"] = "IFAD-0.97"
    rows.append(ifad)
    return pd.concat(rows, ignore_index=True)


def plot_likelihood(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.loc[np.isfinite(frame["logLik"]) & frame["logLik"].ge(-4300)].copy()
    frame["Model"] = pd.Categorical(frame["Model"], categories=METHODS, ordered=True)
    positions = {method: float(index) for index, method in enumerate(METHODS)}
    frame["model_number"] = frame["Model"].map(positions).astype(float)
    frame["violin_number"] = frame["model_number"] + 0.05
    frame["box_number"] = frame["model_number"] - 0.10
    rng = np.random.RandomState(42)
    frame["point_number"] = (
        frame["model_number"] - 0.30 + rng.uniform(-0.04, 0.04, len(frame))
    )
    counts = frame.groupby("Model", observed=True).size()
    distribution = frame.loc[frame["Model"].isin(counts.loc[counts >= 2].index)]
    plot = (
        ggplot(frame, aes(y="logLik", fill="Model", color="Model"))
        + geom_violin(
            aes(x="violin_number", group="Model"),
            data=distribution,
            style="right",
            width=1.0,
            alpha=0.6,
            show_legend=False,
        )
        + geom_boxplot(
            aes(x="box_number", group="Model"),
            data=distribution,
            width=0.10,
            alpha=0.6,
            color="black",
            outlier_alpha=0,
            size=0.8,
            show_legend=False,
        )
        + geom_point(aes(x="point_number"), alpha=0.6, size=1.5, show_legend=False)
        + geom_hline(
            yintercept=BEST_KNOWN,
            color="red",
            linetype="dotted",
            size=1.2,
        )
        + coord_flip()
        + scale_x_continuous(
            breaks=list(positions.values()),
            labels=METHODS,
            limits=(-0.5, len(METHODS) - 0.5),
        )
        + scale_y_continuous(breaks=list(range(-4300, -3699, 100)))
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
    plot.save(
        output / "likelihood_if2warm_comparison_r20.png",
        width=8.2,
        height=4.1,
        dpi=220,
        verbose=False,
    )


def _parameter_frame(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    fit_sources = (
        (args.corenflos_cold / "fit_summary.csv", "Corenflos"),
        (args.corenflos_warm / "fit_summary.csv", "Corenflos + IF2"),
        (args.ditlevsen_cold / "fit_summary.csv", "Ditlevsen"),
        (args.ditlevsen_warm / "fit_summary.csv", "Ditlevsen + IF2"),
    )
    for path, label in fit_sources:
        frame = _read(path)[PARAMETERS].copy()
        frame["source"] = label
        rows.append(frame)
    starts = load_starts_file(args.starts_file, 100)
    checkpoint = pd.DataFrame([_parameter_columns(start) for start in starts])[
        PARAMETERS
    ]
    checkpoint["source"] = "IF2 checkpoint"
    rows.append(checkpoint)
    reference = _read(args.reference / "manuscript_parameters.csv")
    ifad = reference.loc[
        reference["method"].eq("IFAD-0.97") & reference["effort"].eq("comparable"),
        PARAMETERS,
    ].copy()
    ifad["source"] = "IFAD-0.97"
    rows.append(ifad)
    return pd.concat(rows, ignore_index=True)


def plot_parameters(frame: pd.DataFrame, output: Path) -> None:
    long = frame.melt(
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
        + geom_density(alpha=0.30)
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
            legend_text=element_text(size=9),
            legend_title=element_blank(),
            legend_position="bottom",
            panel_spacing=0.02,
        )
        + labs(x="Parameter Value Estimate", y="Density")
    )
    plot.save(
        output / "parameter_if2warm_comparison_r20.png",
        width=10.2,
        height=6.2,
        dpi=220,
        verbose=False,
    )


def _optimization_frame(args: argparse.Namespace) -> pd.DataFrame:
    reference = _read(args.reference / "manuscript_trace_summary.csv")
    ifad = reference.loc[
        reference["method"].eq("IFAD-0.97") & reference["effort"].eq("comparable")
    ].copy()
    ifad["source"] = np.where(ifad["stage"].eq("mif"), "IF2 warm start", "IFAD-0.97")
    columns = ["elapsed_seconds", "median", "q10", "maximum", "source"]
    corenflos = _trace_summary(
        _read(args.corenflos_warm / "optimization_euler_traces.csv"),
        "Corenflos + IF2",
    )
    ditlevsen = _trace_summary(
        _read(args.ditlevsen_warm / "optimization_euler_traces.csv"),
        "Ditlevsen + IF2",
    )
    checkpoint = _read(
        args.warm_start / "warm_start_evaluations.csv"
    )["euler20_loglik"]
    origins = pd.DataFrame(
        [
            {
                "elapsed_seconds": OFFSET,
                "median": float(checkpoint.median()),
                "q10": float(checkpoint.quantile(0.10)),
                "maximum": float(checkpoint.max()),
                "source": source,
            }
            for source in ("Corenflos + IF2", "Ditlevsen + IF2")
        ]
    )
    return pd.concat(
        (ifad[columns], origins[columns], corenflos[columns], ditlevsen[columns]),
        ignore_index=True,
    )


def plot_optimization(frame: pd.DataFrame, output: Path) -> None:
    order = ["IF2 warm start", "IFAD-0.97", "Ditlevsen + IF2", "Corenflos + IF2"]
    frame["source"] = pd.Categorical(frame["source"], categories=order, ordered=True)
    corenflos = frame.loc[frame["source"].eq("Corenflos + IF2")]
    plot = (
        ggplot(frame, aes(x="elapsed_seconds", y="median", color="source"))
        + geom_ribbon(
            aes(ymin="q10", ymax="maximum", fill="source"),
            alpha=0.10,
            color=None,
            show_legend=False,
        )
        + geom_line(size=1.2)
        + geom_line(aes(y="q10"), alpha=0.2, size=0.6, show_legend=False)
        + geom_line(aes(y="maximum"), alpha=0.2, size=0.6, show_legend=False)
        + geom_line(
            data=corenflos,
            color=COLORS["Corenflos + IF2"],
            size=1.8,
            show_legend=False,
        )
        + geom_vline(xintercept=OFFSET, color="black", linetype="dotted", size=1.0)
        + geom_hline(
            yintercept=BEST_KNOWN,
            color="red",
            linetype="dotted",
            size=1.2,
        )
        + scale_color_manual(values=COLORS, breaks=order)
        + scale_fill_manual(values=COLORS, breaks=order)
        + labs(x="Elapsed Time (Seconds)", y="Log-Likelihood")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(size=14, color="black"),
            axis_text_y=element_text(size=14, color="black"),
            axis_title_x=element_text(size=15, color="black"),
            axis_title_y=element_text(size=15, color="black"),
            legend_text=element_text(size=11, color="black"),
            legend_title=element_blank(),
            legend_position="right",
            panel_grid_major=element_line(color="#e5e5e5", size=0.8),
            panel_grid_minor=element_blank(),
        )
    )
    (plot + coord_cartesian(xlim=(0, TOTAL), ylim=(-4300.0, None))).save(
        output / "optimization_if2warm_elapsed_r20.png",
        width=7.5,
        height=4.0,
        dpi=220,
        verbose=False,
    )
    (plot + coord_cartesian(xlim=(0, TOTAL), ylim=(-8000.0, None))).save(
        output / "optimization_if2warm_elapsed_full_r20.png",
        width=7.5,
        height=4.0,
        dpi=220,
        verbose=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corenflos-warm",
        type=Path,
        default=Path("results/if2warm_ifad097_budget_j100_final_100"),
    )
    parser.add_argument(
        "--ditlevsen-warm",
        type=Path,
        default=Path(
            "../ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100"
        ),
    )
    parser.add_argument(
        "--corenflos-cold",
        type=Path,
        default=Path("results/corenflos_j100_eps025_final_100"),
    )
    parser.add_argument(
        "--ditlevsen-cold",
        type=Path,
        default=Path("../ditlevsen/results/block_smc_guided_j100_final_100"),
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path("results/ifad097_post_if2_reference"),
    )
    parser.add_argument(
        "--starts-file",
        type=Path,
        default=Path("../ditlevsen/results/reference/ifad097_comparable_post_if2.npz"),
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("../ditlevsen/results/reference")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/if2warm_ifad097_comparison/figures"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_likelihood(_likelihood_frame(args), args.output)
    plot_parameters(_parameter_frame(args), args.output)
    plot_optimization(_optimization_frame(args), args.output)


if __name__ == "__main__":
    main()
