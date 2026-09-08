"""Plot the Dacca validation results using the manuscript's figure style."""

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


def _read_inputs(validation: Path, reference: Path) -> dict[str, pd.DataFrame]:
    return {
        "optimizer": pd.read_csv(validation / "preflight_optimizer.csv"),
        "schedule": pd.read_csv(validation / "preflight_paper_schedule.csv"),
        "viability": pd.read_csv(validation / "filter_viability.csv"),
        "likelihood": pd.read_csv(reference / "manuscript_likelihood.csv"),
        "parameters": pd.read_csv(reference / "manuscript_parameters.csv"),
        "traces": pd.read_csv(reference / "manuscript_trace_summary.csv"),
    }


def _likelihood_panel(
    data: dict[str, pd.DataFrame], effort: str, letter: str, filename: Path
) -> None:
    """Render one Figure 3 panel."""

    reference = data["likelihood"].loc[
        data["likelihood"]["effort"].eq(effort),
        ["method", "euler_loglik"],
    ].rename(columns={"method": "Model", "euler_loglik": "logLik"})
    reference = reference.loc[reference["logLik"] >= -3780].copy()

    if effort == "comparable":
        ditlevsen = data["optimizer"][["final_euler20_loglik"]].rename(
            columns={"final_euler20_loglik": "logLik"}
        )
        ditlevsen["Model"] = "Ditlevsen"
    else:
        ditlevsen = pd.DataFrame(
            {
                "Model": ["Ditlevsen"],
                "logLik": [float(data["schedule"]["euler20_loglik"].iloc[-1])],
            }
        )
    frame = pd.concat([reference, ditlevsen], ignore_index=True)
    order = [*METHOD_ORDER, "Ditlevsen"]
    frame["Model"] = pd.Categorical(frame["Model"], categories=order, ordered=True)
    positions = {name: float(index) for index, name in enumerate(order)}
    frame["Model_num"] = frame["Model"].map(positions).astype(float)
    frame["Model_violin_x"] = frame["Model_num"] + 0.05
    frame["Model_boxplot_x"] = frame["Model_num"] - 0.10
    rng = np.random.RandomState(42)
    frame["Model_jitter_x"] = (
        frame["Model_num"] - 0.30 + rng.uniform(-0.04, 0.04, len(frame))
    )
    minimum = float(frame["logLik"].min())
    maximum = float(frame["logLik"].max())
    span = maximum - minimum
    tick_step = 10 if span <= 120 else 50 if span <= 400 else 200
    tick_minimum = int(np.floor(minimum / tick_step) * tick_step)
    tick_maximum = int(np.floor(maximum / tick_step) * tick_step)
    tick_breaks = list(range(tick_minimum, tick_maximum + tick_step, tick_step))
    distribution_frame = frame.loc[
        frame["Model"].ne("Ditlevsen") | (effort == "comparable")
    ]
    plot = (
        ggplot(frame, aes(y="logLik", fill="Model", color="Model"))
        + geom_violin(
            aes(x="Model_violin_x", group="Model"),
            data=distribution_frame,
            style="right",
            width=1.0,
            alpha=0.6,
            show_legend=False,
        )
        + geom_boxplot(
            aes(x="Model_boxplot_x", group="Model"),
            data=distribution_frame,
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
            x=4.4,
            y=minimum + max(2.0, 0.03 * span),
            label=letter,
            size=28,
            fontweight="bold",
            color="black",
        )
        + coord_flip()
        + scale_x_continuous(
            breaks=list(positions.values()), labels=order, limits=(-0.5, 4.5)
        )
        + scale_y_continuous(breaks=tick_breaks)
        + scale_color_manual(values=METHOD_COLORS)
        + scale_fill_manual(values=METHOD_COLORS)
        + labs(
            title="",
            subtitle="",
            x="",
            y="Log-Likelihood",
        )
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
        filename,
        width=7,
        height=3.8,
        dpi=220,
        verbose=False,
    )


def plot_likelihood_preflight(data: dict[str, pd.DataFrame], output: Path) -> None:
    """Repeat both panels of Figure 3 with Ditlevsen added."""

    comparable = output / "preflight_likelihood_comparable.png"
    extended = output / "preflight_likelihood_extended.png"
    _likelihood_panel(data, "comparable", "A", comparable)
    _likelihood_panel(data, "extended", "B", extended)
    with Image.open(comparable) as left, Image.open(extended) as right:
        gap = 20
        combined = Image.new(
            "RGB", (left.width + gap + right.width, max(left.height, right.height)), "white"
        )
        combined.paste(left.convert("RGB"), (0, 0))
        combined.paste(right.convert("RGB"), (left.width + gap, 0))
        combined.save(output / "preflight_likelihood_comparison.png", dpi=(220, 220))


def plot_parameter_preflight(data: dict[str, pd.DataFrame], output: Path) -> None:
    """Repeat the Figure 4 grammar with the Ditlevsen estimate added."""

    parameters = [
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
    reference = data["parameters"].loc[
        data["parameters"]["effort"].eq("extended"),
        ["method", *parameters],
    ].rename(columns={"method": "source"})
    long = reference.melt(
        id_vars="source",
        value_vars=parameters,
        var_name="quantity",
        value_name="param_value",
    )
    long = long.loc[np.isfinite(long["param_value"])].copy()
    long["quantity"] = long["quantity"].replace({"omegas5": "omega5"})
    source_order = [*METHOD_ORDER, "Ditlevsen"]
    long["source"] = pd.Categorical(
        long["source"], categories=source_order, ordered=True
    )

    final = data["schedule"].iloc[-1]
    ds_values = {
        name.replace("omegas5", "omega5"): float(final[name])
        for name in parameters
    }
    ds = pd.DataFrame(
        {
            "quantity": list(ds_values),
            "param_value": list(ds_values.values()),
            "source": ["Ditlevsen"] * len(ds_values),
        }
    )
    ds["source"] = pd.Categorical(
        ds["source"], categories=source_order, ordered=True
    )
    labels = {
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
    order = list(labels)
    for frame in (long, ds):
        frame["quantity_label"] = frame["quantity"].map(labels)
        frame["quantity_label"] = pd.Categorical(
            frame["quantity_label"],
            categories=[labels[name] for name in order],
            ordered=True,
        )

    plot = (
        ggplot(long, aes(x="param_value", fill="source", color="source"))
        + geom_density(alpha=0.4)
        + geom_vline(
            aes(xintercept="param_value"),
            data=ds,
            inherit_aes=False,
            color="black",
            linetype="dashed",
            size=0.8,
            show_legend=False,
        )
        + facet_wrap("quantity_label", scales="free", ncol=3)
        + scale_x_continuous(labels=lambda values: [f"{value:g}" for value in values])
        + scale_y_continuous(labels=lambda values: [f"{value:g}" for value in values])
        + scale_color_manual(values=METHOD_COLORS, breaks=METHOD_ORDER)
        + scale_fill_manual(values=METHOD_COLORS, breaks=METHOD_ORDER)
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
        + labs(
            x="Parameter Value Estimate",
            y="Density",
        )
    )
    plot.save(
        output / "preflight_parameter_comparison.png",
        width=9.5,
        height=6.0,
        dpi=220,
        verbose=False,
    )


def plot_elapsed_trace(data: dict[str, pd.DataFrame], output: Path) -> None:
    """Manuscript trace grammar with the DS Euler-20 preflight over elapsed time."""

    reference = data["traces"].loc[
        data["traces"]["effort"].eq("extended")
    ].copy()
    transition = (
        reference.loc[reference["stage"].eq("train")]
        .sort_values("elapsed_seconds")
        .groupby("method", as_index=False)
        .first()
        .rename(columns={"method": "source"})
    )
    reference["source"] = reference["method"]
    ds = data["schedule"].copy()
    ds = ds.assign(
        source="Ditlevsen",
        median=ds["euler20_loglik"],
        q10=ds["euler20_loglik"] - 2.0 * ds["euler20_se"],
        maximum=ds["euler20_loglik"] + 2.0 * ds["euler20_se"],
    )
    trace = pd.concat(
        [
            reference[["elapsed_seconds", "median", "q10", "maximum", "source"]],
            ds[["elapsed_seconds", "median", "q10", "maximum", "source"]],
        ],
        ignore_index=True,
    )
    source_order = [*METHOD_ORDER, "Ditlevsen"]
    trace["source"] = pd.Categorical(
        trace["source"], categories=source_order, ordered=True
    )
    reference_non_if2 = trace.loc[
        trace["source"].isin(["IFAD-0", "IFAD-0.97", "IFAD-1"])
    ]
    reference_if2 = trace.loc[trace["source"].eq("IF2")]
    ditlevsen_trace = trace.loc[trace["source"].eq("Ditlevsen")]

    plot = (
        ggplot(trace, aes(x="elapsed_seconds", y="median", color="source"))
        + geom_line(size=1.2)
        + geom_line(
            aes(y="q10"), data=reference_non_if2,
            alpha=0.2, size=0.6, show_legend=False
        )
        + geom_line(
            aes(y="maximum"), data=reference_non_if2,
            alpha=0.2, size=0.6, show_legend=False
        )
        + geom_line(
            aes(y="q10"), data=reference_if2,
            alpha=0.35, size=0.6, show_legend=False
        )
        + geom_line(
            aes(y="maximum"), data=reference_if2,
            alpha=0.35, size=0.6, show_legend=False
        )
        + geom_line(
            aes(y="q10"), data=ditlevsen_trace,
            alpha=0.25, size=0.6, show_legend=False
        )
        + geom_line(
            aes(y="maximum"), data=ditlevsen_trace,
            alpha=0.25, size=0.6, show_legend=False
        )
        + geom_ribbon(
            aes(ymin="q10", ymax="maximum", fill="source"),
            alpha=0.10,
            color=None,
            show_legend=False,
        )
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
        + labs(
            x="Elapsed Time (Seconds)",
            y="Log-Likelihood",
        )
        + scale_color_manual(values=METHOD_COLORS, breaks=source_order)
        + scale_fill_manual(values=METHOD_COLORS, breaks=source_order)
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
        + coord_cartesian(xlim=(0, 3600), ylim=(-3800, None))
    )
    plot.save(
        output / "preflight_optimization_elapsed.png",
        width=7,
        height=3.8,
        dpi=220,
        verbose=False,
    )


def plot_substep_diagnostics(data: dict[str, pd.DataFrame], output: Path) -> None:
    """Show viability and filtering health as substeps per month increase."""

    frame = data["viability"].copy()
    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for start, group in frame.groupby("start"):
        group = group.sort_values("inference_nstep")
        axes[0].plot(
            group["inference_nstep"],
            group["maximum_invalid_fraction"],
            marker="o",
            linewidth=1.5,
            label=f"Box start {start}",
        )
        viable = group.loc[~group["collapsed"]]
        axes[1].plot(
            viable["inference_nstep"],
            viable["minimum_ess"],
            marker="o",
            linewidth=1.5,
            label=f"Box start {start}",
        )
    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    for axis in axes:
        axis.set_xticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20"])
        axis.set_xlabel("Process substeps per month")
    axes[0].set_ylabel("Maximum invalid-particle fraction")
    axes[0].set_ylim(-0.04, 1.04)
    axes[1].set_ylabel("Minimum ESS (of 128)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[0].text(0.02, 0.96, "A", transform=axes[0].transAxes, va="top", fontsize=20, fontweight="bold")
    axes[1].text(0.02, 0.96, "B", transform=axes[1].transAxes, va="top", fontsize=20, fontweight="bold")
    figure.savefig(
        output / "preflight_substep_diagnostics.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, default=Path("results/validation"))
    parser.add_argument("--reference", type=Path, default=Path("results/reference"))
    parser.add_argument("--output", type=Path, default=Path("results/validation"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = _read_inputs(args.validation, args.reference)
    plot_likelihood_preflight(data, args.output)
    plot_parameter_preflight(data, args.output)
    plot_elapsed_trace(data, args.output)
    plot_substep_diagnostics(data, args.output)


if __name__ == "__main__":
    main()
