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
    coord_cartesian,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_col,
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
    scale_linetype_manual,
    scale_shape_manual,
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
# The continuation medians and envelopes span approximately -3814 to -3741.
TRACE_LIMITS = (-3820.0, -3735.0)
OVERVIEW_LIMITS = (-4800.0, -3735.0)
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
DISPLAY_LABELS = {
    "Corenflos + IF2": "Corenflos + IF2 warm start",
    "Ditlevsen + IF2": "Ditlevsen + IF2 warm start",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save(plot: ggplot, output: Path, name: str, width: float, height: float,
          save_pdf: bool = False) -> None:
    for extension in (("png", "pdf") if save_pdf else ("png",)):
        plot.save(output / f"{name}.{extension}", width=width, height=height,
                  dpi=220, verbose=False)


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


def plot_likelihood(frame: pd.DataFrame, output: Path, *, display_labels=None,
                    minimum=-4300, methods=None, limits=None,
                    name="likelihood_if2warm_comparison_r20", save_pdf=False,
                    best_known=BEST_KNOWN) -> None:
    labels = DISPLAY_LABELS if display_labels is None else display_labels
    methods = METHODS if methods is None else methods
    frame = frame.loc[np.isfinite(frame["logLik"]) & frame["Model"].isin(methods)].copy()
    if minimum is not None:
        frame = frame.loc[frame["logLik"].ge(minimum)].copy()
    frame["Model"] = pd.Categorical(frame["Model"], categories=methods, ordered=True)
    positions = {method: float(index) for index, method in enumerate(methods)}
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
        + coord_flip(ylim=limits)
        + scale_x_continuous(
            breaks=list(positions.values()),
            labels=[labels.get(method, method) for method in methods],
            limits=(-0.5, len(methods) - 0.5),
        )
        + (scale_y_continuous(breaks=list(range(-4300, -3699, 100)))
           if minimum == -4300 else scale_y_continuous())
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
    if best_known is not None:
        plot += geom_hline(yintercept=best_known, color="red", linetype="dotted", size=1.2)
    _save(plot, output, name, 10.2 if display_labels else 8.2, 4.1, save_pdf)


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


def plot_parameters(frame: pd.DataFrame, output: Path, *, display_labels=None,
                    save_pdf=False) -> None:
    labels = DISPLAY_LABELS if display_labels is None else display_labels
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
        + scale_color_manual(
            values=COLORS,
            breaks=METHODS,
            labels=[labels.get(method, method) for method in METHODS],
        )
        + scale_fill_manual(
            values=COLORS,
            breaks=METHODS,
            labels=[labels.get(method, method) for method in METHODS],
        )
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
    _save(plot, output, "parameter_if2warm_comparison_r20", 10.2, 6.2, save_pdf)


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
    corenflos_cold = _trace_summary(
        _read(args.corenflos_cold / "optimization_euler_traces.csv"),
        "Corenflos",
    )
    ditlevsen_cold = _trace_summary(
        _read(args.ditlevsen_cold / "optimization_euler_traces.csv"),
        "Ditlevsen",
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
        (
            ifad[columns],
            origins[columns],
            corenflos[columns],
            ditlevsen[columns],
            corenflos_cold[columns],
            ditlevsen_cold[columns],
        ),
        ignore_index=True,
    )


def _optimization_plot(frame: pd.DataFrame, order: list[str], *,
                       display_labels=None, best_known=BEST_KNOWN) -> ggplot:
    frame = frame.loc[frame["source"].isin(order)].copy()
    frame["source"] = pd.Categorical(frame["source"], categories=order, ordered=True)
    linetypes = {source: "solid" for source in order}
    linetypes["Corenflos + IF2"] = "dashed"
    display_labels = DISPLAY_LABELS if display_labels is None else display_labels
    labels = [display_labels.get(source, source) for source in order]
    plot = (
        ggplot(frame, aes(x="elapsed_seconds", y="median", color="source"))
        + geom_ribbon(
            aes(ymin="q10", ymax="maximum", fill="source"),
            alpha=0.10,
            color=None,
            show_legend=False,
        )
        + geom_line(aes(linetype="source"), size=1.0)
        + geom_line(aes(y="q10"), alpha=0.2, size=0.6, show_legend=False)
        + geom_line(aes(y="maximum"), alpha=0.2, size=0.6, show_legend=False)
        + geom_vline(xintercept=OFFSET, color="black", linetype="dotted", size=1.0)
        + scale_color_manual(values=COLORS, breaks=order, labels=labels)
        + scale_fill_manual(values=COLORS, breaks=order, labels=labels)
        + scale_linetype_manual(values=linetypes, breaks=order, labels=labels)
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
    if best_known is not None:
        plot += geom_hline(yintercept=best_known, color="red", linetype="dotted", size=1.2)
    return plot


def plot_optimization(frame: pd.DataFrame, output: Path, *, display_labels=None,
                      save_pdf=False, best_known=BEST_KNOWN) -> None:
    order = ["IF2 warm start", "IFAD-0.97", "Ditlevsen + IF2", "Corenflos + IF2"]
    focused = (
        _optimization_plot(frame, order, display_labels=display_labels, best_known=best_known)
        + scale_y_continuous(breaks=list(range(-3820, -3734, 10)))
        + coord_cartesian(xlim=(0, TOTAL), ylim=TRACE_LIMITS)
    )
    _save(focused, output, "optimization_if2warm_elapsed_r20",
          10.0 if display_labels else 7.5, 4.0, save_pdf)
    overview = (
        _optimization_plot(frame, order + ["Ditlevsen", "Corenflos"],
                           display_labels=display_labels, best_known=best_known)
        + geom_line(
            aes(y="maximum"),
            data=frame.loc[frame["source"].isin(["Ditlevsen", "Corenflos"])],
            alpha=0.65,
            size=0.6,
            show_legend=False,
        )
        + scale_y_continuous(breaks=list(range(-4800, -3799, 200)))
        + coord_cartesian(xlim=(0, TOTAL), ylim=OVERVIEW_LIMITS)
    )
    _save(overview, output, "optimization_if2warm_elapsed_full_r20",
          10.0 if display_labels else 7.5, 4.0, save_pdf)


def _objective_mismatch_frame(args: argparse.Namespace) -> pd.DataFrame:
    baseline = _read(args.warm_start / "warm_start_evaluations.csv")
    baseline = baseline[["start", "euler20_loglik"]].rename(
        columns={"euler20_loglik": "initial_euler_loglik"}
    )
    rows = []
    for directory, source in (
        (args.corenflos_warm, "Corenflos + IF2"),
        (args.ditlevsen_warm, "Ditlevsen + IF2"),
    ):
        fits = _read(directory / "fit_summary.csv")
        training = _read(directory / "optimization_training_traces.csv")
        final = _read(directory / "final_evaluations.csv")
        if set(fits["start"]) != set(baseline["start"]) or set(final["start"]) != set(
            baseline["start"]
        ):
            raise ValueError(f"{source}: mismatched warm-start identities")
        selected = fits[["start", "output_selected_iteration"]].rename(
            columns={"output_selected_iteration": "iteration"}
        ).merge(
            training[["start", "iteration", "pseudo_loglik"]],
            on=["start", "iteration"],
            how="left",
            validate="one_to_one",
        ).rename(columns={"pseudo_loglik": "selected_training_loglik"})
        initial = training.loc[training["iteration"].eq(0), ["start", "pseudo_loglik"]]
        initial = initial.rename(columns={"pseudo_loglik": "initial_training_loglik"})
        final = final[["start", "euler20_loglik"]].rename(
            columns={"euler20_loglik": "selected_euler_loglik"}
        )
        matched = (
            selected.merge(initial, on="start", how="left", validate="one_to_one")
            .merge(baseline, on="start", how="left", validate="one_to_one")
            .merge(final, on="start", how="left", validate="one_to_one")
            .rename(columns={"iteration": "output_selected_iteration"})
            .sort_values("start")
        )
        value_columns = [
            "initial_training_loglik", "selected_training_loglik",
            "initial_euler_loglik", "selected_euler_loglik",
        ]
        if not np.isfinite(matched[value_columns].to_numpy()).all():
            raise ValueError(f"{source}: missing/non-finite paired objective values")
        matched["training_delta"] = (
            matched["selected_training_loglik"] - matched["initial_training_loglik"]
        )
        matched["euler_delta"] = (
            matched["selected_euler_loglik"] - matched["initial_euler_loglik"]
        )
        matched["source"] = DISPLAY_LABELS[source]
        rows.append(matched)
    return pd.concat(rows, ignore_index=True)


def plot_objective_mismatch(frame: pd.DataFrame, output: Path, *,
                            display_labels=None, save_pdf=False) -> None:
    labels = DISPLAY_LABELS if display_labels is None else display_labels
    sources = ("Corenflos + IF2", "Ditlevsen + IF2")
    order = [labels[source] for source in sources]
    frame = frame.copy()
    frame["source"] = frame["source"].replace(
        {DISPLAY_LABELS[source]: labels[source] for source in sources}
    )
    frame["source"] = pd.Categorical(frame["source"], categories=order, ordered=True)
    medians = frame.groupby("source", observed=True)[["training_delta", "euler_delta"]].median()
    summary = medians.reset_index().melt(
        id_vars="source", var_name="objective", value_name="median_change"
    )
    objective_labels = {
        "training_delta": "Logged training", "euler_delta": "Euler-20",
    }
    summary["objective"] = pd.Categorical(
        summary["objective"].map(objective_labels),
        categories=list(objective_labels.values()),
        ordered=True,
    )
    styling = theme(
        axis_text_x=element_text(size=12, color="black"),
        axis_text_y=element_text(size=12, color="black"),
        axis_title_x=element_text(size=14, color="black"),
        axis_title_y=element_text(size=14, color="black"),
        strip_text=element_text(size=12, color="black"),
        legend_text=element_text(size=11, color="black"),
        legend_title=element_blank(),
        legend_position="bottom",
        panel_grid_major=element_line(color="#e5e5e5", size=0.8),
        panel_grid_minor=element_blank(),
    )
    bars = (
        ggplot(summary, aes(x="objective", y="median_change", fill="objective"))
        + geom_col(width=0.6, show_legend=False)
        + geom_hline(yintercept=0.0, color="black", size=0.6)
        + facet_wrap("source", nrow=1)
        + scale_fill_manual(values={"Logged training": "#969696", "Euler-20": "#31688e"})
        + labs(x="", y="Median paired log-likelihood change")
        + theme_minimal()
        + styling
    )
    _save(bars, output, "objective_mismatch_if2warm_medians_r20", 9.4, 4.3, save_pdf)
    frame["marker"] = "Warm starts"
    medians = medians.reset_index()
    medians["marker"] = "Marginal medians"
    scatter = (
        ggplot(frame, aes(x="training_delta", y="euler_delta", color="source"))
        + geom_vline(xintercept=0.0, color="#666666", linetype="dashed", size=0.6)
        + geom_hline(yintercept=0.0, color="#666666", linetype="dashed", size=0.6)
        + geom_point(aes(shape="marker"), size=2.0, alpha=0.65)
        + geom_point(
            aes(shape="marker"), data=medians,
            size=4.0, color="black", fill="white", stroke=1.1,
        )
        + facet_wrap("source", nrow=1)
        + scale_color_manual(
            values={labels[source]: COLORS[source] for source in sources},
            guide=None,
        )
        + scale_shape_manual(values={"Warm starts": "o", "Marginal medians": "D"})
        + labs(x="Logged training log-likelihood change", y="Euler-20 log-likelihood change")
        + theme_minimal()
        + styling
        + theme(panel_spacing_x=0.055)
    )
    _save(scatter, output, "objective_mismatch_if2warm_paired_r20", 9.4, 4.5, save_pdf)


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
    mismatch = _objective_mismatch_frame(args)
    mismatch.to_csv(args.output.parent / "objective_mismatch_if2warm_r20.csv", index=False)
    plot_objective_mismatch(mismatch, args.output)


if __name__ == "__main__":
    main()
