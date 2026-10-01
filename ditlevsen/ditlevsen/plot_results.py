"""Create DMOP-manuscript-style figures from the corrected benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

from .model import DEFAULT_PARAMETERS


PALETTE = {
    1: "#54278f",
    2: "#2b8cbe",
    3: "#66c2a4",
    4: "#238b45",
    5: "#cc79a7",
    10: "#d55e00",
    20: "#000000",
}
METHOD_PALETTE = {
    "IFAD-0": "#54278f",
    "IFAD-0.97": "#2b8cbe",
    "IFAD-1": "#2ca25f",
    "IF2": "#f0c808",
}
IFAD_BEST = -3744.17
DISPLAY_FLOOR = -3800.0


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=220)
    plt.close(figure)


def plot_likelihood_comparison(
    frame: pd.DataFrame,
    manuscript: pd.DataFrame,
    output: Path,
) -> None:
    """Extend Figure 3 using the manuscript's original plotting grammar."""

    reference = manuscript.loc[manuscript["effort"].eq("comparable")].copy()
    reference = reference.rename(
        columns={"method": "Model", "euler_loglik": "logLik"}
    )
    fits = frame.loc[
        frame["method"].eq("DS-local+nugget QML")
        & frame["inference_nstep"].isin([5, 10, 20])
        & frame["euler_loglik"].notna()
    ].copy()
    fits["Model"] = fits["inference_nstep"].map(
        lambda value: f"DS-QML-{int(value)}"
    )
    fits = fits.rename(columns={"euler_loglik": "logLik"})
    raincloud = pd.concat(
        [reference[["Model", "logLik"]], fits[["Model", "logLik"]]],
        ignore_index=True,
    )

    # This is the exact manuscript display filter, including its -3780 cutoff.
    raincloud = raincloud.loc[raincloud["logLik"] >= -3780].copy()
    model_order = [
        "IFAD-0",
        "IFAD-0.97",
        "IFAD-1",
        "IF2",
        "DS-QML-5",
        "DS-QML-10",
        "DS-QML-20",
    ]
    colors = {
        **METHOD_PALETTE,
        **{f"DS-QML-{r}": PALETTE[r] for r in (5, 10, 20)},
    }
    raincloud["Model"] = pd.Categorical(
        raincloud["Model"], categories=model_order, ordered=True
    )
    model_map = {name: float(index) for index, name in enumerate(model_order)}
    raincloud["Model_num"] = raincloud["Model"].map(model_map).astype(float)
    raincloud["Model_violin_x"] = raincloud["Model_num"] + 0.05
    raincloud["Model_boxplot_x"] = raincloud["Model_num"] - 0.1
    rng = np.random.default_rng(42)
    raincloud["Model_jitter_x"] = (
        raincloud["Model_num"]
        - 0.30
        + rng.uniform(-0.04, 0.04, size=len(raincloud))
    )
    minimum_tick = int(np.floor(raincloud["logLik"].min() / 10) * 10)
    maximum_tick = int(np.ceil(raincloud["logLik"].max() / 10) * 10)
    ticks = list(range(minimum_tick, maximum_tick + 10, 10))
    plot = (
        ggplot(raincloud, aes(y="logLik", fill="Model", color="Model"))
        + geom_violin(
            aes(x="Model_violin_x", group="Model"),
            style="right",
            width=1.0,
            alpha=0.6,
            show_legend=False,
        )
        + geom_boxplot(
            aes(x="Model_boxplot_x", group="Model"),
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
            yintercept=raincloud["logLik"].max(),
            color="red",
            linetype="dotted",
            size=1.2,
            alpha=0.9,
        )
        + annotate(
            "text",
            x=6.4,
            y=raincloud["logLik"].min() + 2,
            label="A",
            size=28,
            fontweight="bold",
            color="black",
        )
        + coord_flip()
        + scale_x_continuous(
            breaks=list(model_map.values()), labels=model_order
        )
        + scale_y_continuous(breaks=ticks)
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + labs(title="", subtitle="", x="", y="Log-Likelihood")
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
        output / "method_likelihood_comparison.png",
        width=7,
        height=6.0,
        dpi=220,
        verbose=False,
    )


def plot_discretization(data: Path, output: Path) -> None:
    mixing = pd.read_csv(data / "backward_mixing.csv")
    brackets = pd.read_csv(data / "bracket_rank.csv")
    transition = pd.read_csv(data / "transition_rank.csv")
    nstep_column = (
        "inference_nstep" if "inference_nstep" in transition else "nstep"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    axis = axes[0]
    axis.plot(
        brackets["bracket_depth"],
        brackets["rank"],
        marker="o",
        color="#2b8cbe",
    )
    axis.axvspan(-0.1, 1.1, color="#f28e2b", alpha=0.17)
    axis.text(0.04, 0.90, "order-1.5\nfirst bracket", transform=axis.transAxes, fontsize=8)
    axis.set_xticks(brackets["bracket_depth"].astype(int))
    axis.set_yticks(range(1, 7))
    axis.set_xlabel("Maximum drift-bracket depth")
    axis.set_ylabel("Rank of normalized bracket matrix")
    axis.set_title("A   Dacca bracket depth", loc="left")

    axis = axes[1]
    for order, label, color in (
        (2, "DS first-bracket covariance", "#54278f"),
        (6, "six-column completion", "#2ca25f"),
    ):
        subset = transition.loc[transition["order"] == order]
        axis.plot(
            subset[nstep_column],
            subset["rank"],
            marker="o",
            label=label,
            color=color,
        )
    axis.set_xscale("log", base=2)
    values = sorted(transition[nstep_column].unique())
    axis.set_xticks(values, [str(int(value)) for value in values])
    axis.set_ylim(0.7, 6.3)
    axis.set_yticks(range(1, 7))
    axis.set_xlabel("Substeps per month")
    axis.set_ylabel("Numerical covariance rank")
    axis.set_title("B   Local Gaussian covariance", loc="left")
    axis.legend(frameon=False, fontsize=7)

    axis = axes[2]
    x = mixing["nstep"]
    axis.plot(
        x,
        mixing["unit_backward_ess_mean"],
        marker="o",
        color="#54278f",
        label="ordinary one-step BS",
    )
    axis.plot(
        x,
        mixing["bridge_backward_ess_mean"],
        marker="o",
        color="#2ca25f",
        label="fixed-month lookahead",
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks(x, [str(int(value)) for value in x])
    axis.set_xlabel("Substeps per month")
    axis.set_ylabel("Backward-weight ESS (of 128)")
    axis.set_title("C   Backward mixing", loc="left")
    axis.legend(frameon=False, fontsize=7, loc="center left")
    _save(figure, output, "discretization_diagnostics")


def plot_stability(data: Path, output: Path) -> None:
    frame = pd.read_csv(data / "taylor_stability.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    axis = axes[0]
    for nstep, subset in frame.groupby("inference_nstep"):
        early = subset.loc[subset["month"] <= 12]
        values = np.minimum(
            early["maximum_absolute_filtered_state"].to_numpy(), 1e8
        )
        axis.plot(
            early["month"],
            values,
            color=PALETTE[int(nstep)],
            marker="o",
            markersize=3,
            label=f"r={int(nstep)}",
        )
    axis.axhline(1.2, color="0.3", linestyle=":", linewidth=1)
    axis.set_yscale("log")
    axis.set_xlabel("Observation month")
    axis.set_ylabel("Maximum absolute filtered state")
    axis.set_title("A   Coarse Taylor-transition stability", loc="left")
    axis.legend(frameon=False, ncol=2)

    axis = axes[1]
    horizon = int(frame["month"].max()) + 1
    nsteps = sorted(int(value) for value in frame["inference_nstep"].unique())
    first_box = []
    first_nonfinite = []
    for nstep in nsteps:
        subset = frame.loc[frame["inference_nstep"] == nstep]
        invalid = subset.loc[~subset["state_in_box"]]
        nonfinite = subset.loc[~subset["finite_loglik_increment"]]
        first_box.append(int(invalid["month"].iloc[0]) if len(invalid) else horizon)
        first_nonfinite.append(
            int(nonfinite["month"].iloc[0]) if len(nonfinite) else horizon
        )
    axis.scatter(
        nsteps,
        first_box,
        color="#2b8cbe",
        marker="o",
        s=45,
        label="first state-box violation",
    )
    axis.scatter(
        nsteps,
        first_nonfinite,
        color="#d62728",
        marker="x",
        s=55,
        linewidth=2,
        label="first non-finite increment",
    )
    axis.axhline(horizon, color="0.5", linestyle=":", linewidth=1)
    axis.text(
        0.98,
        0.95,
        "no failure",
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(nsteps, [str(value) for value in nsteps])
    axis.set_xlabel("Ditlevsen substeps per month")
    axis.set_ylabel("First affected month")
    axis.set_title("B   Full-series viability", loc="left")
    axis.legend(frameon=False, fontsize=8, loc="center right")
    _save(figure, output, "taylor_stability")


def plot_mixing_sensitivity(data: Path, output: Path) -> None:
    frame = pd.read_csv(data / "backward_mixing_sensitivity.csv")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    for axis, (spread, subset) in zip(
        axes,
        frame.groupby("parent_spread", sort=True),
        strict=True,
    ):
        axis.plot(
            subset["nstep"],
            subset["unit_backward_ess_mean"],
            marker="o",
            color="#54278f",
            label="ordinary BS",
        )
        axis.plot(
            subset["nstep"],
            subset["bridge_backward_ess_mean"],
            marker="o",
            color="#2ca25f",
            label="fixed-month lookahead",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(subset["nstep"], [str(int(x)) for x in subset["nstep"]])
        axis.set_title(f"Parent spread = {spread:g}")
        axis.set_xlabel("Substeps per month")
    axes[0].set_ylabel("Backward-weight ESS (of 128)")
    axes[-1].legend(frameon=False, fontsize=8)
    _save(figure, output, "mixing_sensitivity")


def plot_completion_comparison(
    local_likelihood: pd.DataFrame,
    krylov_likelihood: pd.DataFrame,
    output: Path,
) -> None:
    local = local_likelihood.loc[
        local_likelihood["method"].eq("DS-local+nugget QML")
        & local_likelihood["start"].eq(0)
        & local_likelihood["euler_loglik"].notna(),
        ["inference_nstep", "euler_loglik", "euler_se"],
    ].copy()
    krylov = krylov_likelihood.loc[
        krylov_likelihood["method"].eq("DS-Krylov QML")
        & krylov_likelihood["euler_loglik"].notna(),
        ["inference_nstep", "euler_loglik", "euler_se"],
    ].copy()
    merged = local.merge(
        krylov,
        on="inference_nstep",
        suffixes=("_local", "_krylov"),
    ).sort_values("inference_nstep")
    if merged.empty:
        return

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    axis = axes[0]
    axis.errorbar(
        merged["inference_nstep"],
        merged["euler_loglik_local"],
        yerr=merged["euler_se_local"],
        color="#54278f",
        marker="o",
        capsize=3,
        label="first-bracket + nugget",
    )
    axis.errorbar(
        merged["inference_nstep"],
        merged["euler_loglik_krylov"],
        yerr=merged["euler_se_krylov"],
        color="#2ca25f",
        marker="s",
        capsize=3,
        label="six-column completion",
    )
    axis.axhline(IFAD_BEST, color="red", linestyle=":", linewidth=2)
    axis.set_xscale("log", base=2)
    nsteps = merged["inference_nstep"].astype(int).tolist()
    axis.set_xticks(nsteps, [str(value) for value in nsteps])
    axis.set_xlabel("Ditlevsen substeps per month")
    axis.set_ylabel("Euler-20 log-likelihood")
    axis.set_title("A   Higher-bracket sensitivity", loc="left")
    axis.legend(frameon=False)

    difference = merged["euler_loglik_krylov"] - merged["euler_loglik_local"]
    axis = axes[1]
    axis.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    axis.bar(
        merged["inference_nstep"].astype(str),
        difference,
        color=[PALETTE[int(value)] for value in merged["inference_nstep"]],
    )
    axis.set_xlabel("Ditlevsen substeps per month")
    axis.set_ylabel("Krylov minus first-bracket log-likelihood")
    axis.set_title("B   Change in evaluation target", loc="left")
    _save(figure, output, "completion_comparison")


def plot_optimization_comparison(
    ds_traces: pd.DataFrame,
    manuscript_traces: pd.DataFrame,
    output: Path,
) -> None:
    """Extend the manuscript trace using its exact single-panel grammar."""

    reference = manuscript_traces.loc[
        manuscript_traces["effort"].eq("extended")
    ].copy()
    reference = reference.rename(
        columns={
            "method": "source",
            "elapsed_seconds": "time_scaled",
            "maximum": "qmax",
        }
    )
    ds = ds_traces.loc[
        ds_traces["start"].eq(0)
        & ds_traces["inference_nstep"].isin([5, 10, 20])
        & ds_traces["euler_loglik"].notna()
    ].copy()
    ds["source"] = ds["inference_nstep"].map(
        lambda value: f"DS-QML-{int(value)}"
    )
    ds["time_scaled"] = ds["elapsed_seconds"]
    ds["median"] = ds["euler_loglik"]
    ds["q10"] = ds["euler_loglik"]
    ds["qmax"] = ds["euler_loglik"]
    ds["stage"] = "qml"
    columns = ["source", "time_scaled", "median", "q10", "qmax", "stage"]
    statistics = pd.concat([reference[columns], ds[columns]], ignore_index=True)
    source_order = [
        "IFAD-0",
        "IFAD-0.97",
        "IFAD-1",
        "IF2",
        "DS-QML-5",
        "DS-QML-10",
        "DS-QML-20",
    ]
    colors = {
        **METHOD_PALETTE,
        **{f"DS-QML-{r}": PALETTE[r] for r in (5, 10, 20)},
    }
    statistics["source"] = pd.Categorical(
        statistics["source"], categories=source_order, ordered=True
    )
    transitions = reference.loc[reference["stage"].eq("train")].copy()
    transitions = transitions.loc[
        transitions.groupby("source", observed=True)["time_scaled"].idxmin()
    ]
    manuscript_likelihood = pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "results/reference/manuscript_likelihood.csv"
    )
    best_loglik = float(
        manuscript_likelihood.loc[
            manuscript_likelihood["effort"].eq("extended"), "euler_loglik"
        ].max()
    )
    non_if2 = statistics.loc[statistics["source"] != "IF2"]
    if2 = statistics.loc[statistics["source"] == "IF2"]
    plot = (
        ggplot(statistics, aes(x="time_scaled", y="median", color="source"))
        + geom_line(size=1.2)
        + geom_line(
            aes(y="q10", color="source"),
            data=non_if2,
            alpha=0.2,
            size=0.6,
            show_legend=False,
        )
        + geom_line(
            aes(y="qmax", color="source"),
            data=non_if2,
            alpha=0.2,
            size=0.6,
            show_legend=False,
        )
        + geom_line(
            aes(y="q10", color="source"),
            data=if2,
            alpha=0.35,
            size=0.6,
            show_legend=False,
        )
        + geom_line(
            aes(y="qmax", color="source"),
            data=if2,
            alpha=0.35,
            size=0.6,
            show_legend=False,
        )
        + geom_ribbon(
            aes(ymin="q10", ymax="qmax", fill="source"),
            data=non_if2,
            alpha=0.1,
            color=None,
        )
        + geom_ribbon(
            aes(ymin="q10", ymax="qmax", fill="source"),
            data=if2,
            alpha=0.1,
            color=None,
        )
        + geom_hline(
            yintercept=best_loglik,
            color="red",
            linetype="dotted",
            size=1.2,
            alpha=0.9,
        )
        + geom_vline(
            aes(xintercept="time_scaled", color="source"),
            data=transitions,
            linetype="dotted",
            size=1.0,
            show_legend=False,
        )
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + labs(title="", subtitle="", x="Elapsed Time (Seconds)", y="Log-Likelihood")
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
        + coord_cartesian(ylim=(-3800, None))
    )
    plot.save(
        output / "optimization_comparison.png",
        width=7,
        height=3.8,
        dpi=220,
        verbose=False,
    )


def plot_parameter_comparison(
    summary: pd.DataFrame,
    manuscript: pd.DataFrame,
    output: Path,
) -> None:
    """Extend Figure 4 using the manuscript's exact density-facet grammar."""

    successful = summary.loc[
        summary["converged"] & summary["inference_nstep"].isin([5, 10, 20])
    ].copy()
    reference = manuscript.loc[manuscript["effort"].eq("extended")].copy()
    reference = reference.rename(columns={"method": "source"})
    successful["source"] = successful["inference_nstep"].map(
        lambda value: f"DS-QML-{int(value)}"
    )
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
    combined = pd.concat(
        [reference[["source", *parameters]], successful[["source", *parameters]]],
        ignore_index=True,
    )
    long = combined.melt(
        id_vars="source",
        value_vars=parameters,
        var_name="quantity",
        value_name="param_value",
    )
    long["quantity"] = long["quantity"].replace({"omegas5": "omega5"})
    key_quantities = [
        "sigma",
        "tau",
        "gamma",
        "epsilon",
        "S_0",
        "I_0",
        "R1_0",
        "bs3",
        "omega5",
    ]
    label_map = {
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
    long["quantity_label"] = long["quantity"].map(label_map)
    long["quantity_label"] = pd.Categorical(
        long["quantity_label"],
        categories=[label_map[name] for name in key_quantities],
        ordered=True,
    )
    source_order = [
        "IFAD-0",
        "IFAD-0.97",
        "IFAD-1",
        "IF2",
        "DS-QML-5",
        "DS-QML-10",
        "DS-QML-20",
    ]
    long["source"] = pd.Categorical(
        long["source"], categories=source_order, ordered=True
    )
    colors = {
        **METHOD_PALETTE,
        **{f"DS-QML-{r}": PALETTE[r] for r in (5, 10, 20)},
    }
    plot = (
        ggplot(long, aes(x="param_value", fill="source", color="source"))
        + geom_density(alpha=0.4)
        + facet_wrap("quantity_label", scales="free", ncol=3)
        + scale_x_continuous(labels=lambda values: [f"{value:g}" for value in values])
        + scale_y_continuous(labels=lambda values: [f"{value:g}" for value in values])
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
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
        output / "parameter_comparison.png",
        width=9.5,
        height=6.0,
        dpi=220,
        verbose=False,
    )


def write_summaries(
    likelihood: pd.DataFrame,
    summary: pd.DataFrame,
    data: Path,
    output: Path,
    *,
    krylov_likelihood: pd.DataFrame | None = None,
    manuscript_likelihood: pd.DataFrame | None = None,
    target_traces: pd.DataFrame | None = None,
) -> None:
    fits = likelihood.loc[likelihood["method"].str.contains("QML", na=False)]
    rows = []
    for nstep in sorted(summary["inference_nstep"].unique()):
        candidate = fits.loc[fits["inference_nstep"] == nstep]
        finite = candidate["euler_loglik"].dropna()
        fit_subset = summary.loc[summary["inference_nstep"] == nstep]
        rows.append(
            {
                "substeps": int(nstep),
                "attempts": int(len(fit_subset)),
                "successful_fits": int(fit_subset["converged"].sum()),
                "best_euler20_loglik": float(finite.max()) if len(finite) else None,
                "median_euler20_loglik": float(finite.median()) if len(finite) else None,
            }
        )
    mixing = pd.read_csv(data / "backward_mixing.csv")
    findings = {
        "evaluation_model": "Dacca Euler model with 20 substeps per month",
        "ifad_097_paper_best": IFAD_BEST,
        "published_parameter_point": float(
            likelihood.loc[
                likelihood["method"] == "Published parameter point",
                "euler_loglik",
            ].iloc[0]
        ),
        "ditlevsen_fits": rows,
        "ordinary_backward_ess_at_20": float(
            mixing.loc[mixing["nstep"] == 20, "unit_backward_ess_mean"].iloc[0]
        ),
        "bridge_backward_ess_at_20": float(
            mixing.loc[mixing["nstep"] == 20, "bridge_backward_ess_mean"].iloc[0]
        ),
        "substep_trend_interpretation": (
            "The DS-QML Euler-20 scores at r=5,10,20 are effectively flat, not "
            "monotonically improving or worsening. This deterministic wrapper has "
            "no sampled genealogy, so it does not test the expected DS-SAEM "
            "path-degeneracy trend. Ordinary backward-weight ESS does degrade."
        ),
    }
    if manuscript_likelihood is not None:
        comparable = manuscript_likelihood.loc[
            manuscript_likelihood["effort"].eq("comparable")
        ]
        findings["manuscript_comparable_effort"] = [
            {
                "method": method,
                "searches": int(len(subset)),
                "best_euler20_loglik": float(subset["euler_loglik"].max()),
                "median_euler20_loglik": float(subset["euler_loglik"].median()),
            }
            for method, subset in comparable.groupby("method", sort=False)
        ]
    if target_traces is not None:
        timed_rows = []
        for nstep, subset in target_traces.loc[
            target_traces["start"].eq(0)
        ].groupby("inference_nstep"):
            subset = subset.sort_values("iteration")
            timed_rows.append(
                {
                    "substeps": int(nstep),
                    "elapsed_seconds": float(subset["elapsed_seconds"].iloc[-1]),
                    "initial_euler20_loglik": float(subset["euler_loglik"].iloc[0]),
                    "final_euler20_loglik": float(subset["euler_loglik"].iloc[-1]),
                    "change_euler20_loglik": float(
                        subset["euler_loglik"].iloc[-1]
                        - subset["euler_loglik"].iloc[0]
                    ),
                }
            )
        findings["timed_published_start_trace"] = timed_rows
    if krylov_likelihood is not None:
        completed = krylov_likelihood.loc[
            krylov_likelihood["method"].eq("DS-Krylov QML")
            & krylov_likelihood["euler_loglik"].notna()
        ].sort_values("inference_nstep")
        findings["krylov_published_start"] = [
            {
                "substeps": int(row["inference_nstep"]),
                "euler20_loglik": float(row["euler_loglik"]),
                "euler20_se": float(row["euler_se"]),
            }
            for _, row in completed.iterrows()
        ]
    (output / "findings.json").write_text(json.dumps(findings, indent=2) + "\n")

    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"DS substeps & Attempts & Successful & Best Euler-20 & Median Euler-20 \\",
        r"\midrule",
    ]
    for row in rows:
        best = "--" if row["best_euler20_loglik"] is None else f"{row['best_euler20_loglik']:.2f}"
        median = "--" if row["median_euler20_loglik"] is None else f"{row['median_euler20_loglik']:.2f}"
        lines.append(
            f"{row['substeps']} & {row['attempts']} & {row['successful_fits']} & {best} & {median} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{3}}{{l}}{{Published parameter point (reevaluated)}} & {findings['published_parameter_point']:.2f} & -- \\",
            r"\multicolumn{3}{l}{IFAD-0.97 (paper)} & -3744.17 & -3745.77 \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output / "summary_table.tex").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/data_primary"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--krylov-data", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("results/reference"))
    parser.add_argument("--timed-data", type=Path, default=Path("results/data_timed"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()
    likelihood = pd.read_csv(args.data / "euler_loglik.csv")
    summary = pd.read_csv(args.data / "fit_summary.csv")
    manuscript_likelihood = pd.read_csv(
        args.reference / "manuscript_likelihood.csv"
    )
    manuscript_parameters = pd.read_csv(
        args.reference / "manuscript_parameters.csv"
    )
    manuscript_traces = pd.read_csv(
        args.reference / "manuscript_trace_summary.csv"
    )
    target_traces = pd.read_csv(args.timed_data / "target_optimization_traces.csv")
    plot_likelihood_comparison(likelihood, manuscript_likelihood, args.output)
    plot_discretization(args.data, args.output)
    plot_stability(args.data, args.output)
    plot_mixing_sensitivity(args.data, args.output)
    plot_optimization_comparison(target_traces, manuscript_traces, args.output)
    plot_parameter_comparison(summary, manuscript_parameters, args.output)
    krylov_likelihood = None
    if args.krylov_data is not None:
        krylov_likelihood = pd.read_csv(args.krylov_data / "euler_loglik.csv")
        plot_completion_comparison(
            likelihood,
            krylov_likelihood,
            args.output,
        )
    write_summaries(
        likelihood,
        summary,
        args.data,
        args.output,
        krylov_likelihood=krylov_likelihood,
        manuscript_likelihood=manuscript_likelihood,
        target_traces=target_traces,
    )


if __name__ == "__main__":
    main()
