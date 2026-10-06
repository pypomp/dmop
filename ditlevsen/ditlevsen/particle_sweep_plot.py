"""Plot the wall-clock particle-count and learning-rate screen."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BEST_KNOWN = -3744.17


def plot_particle_sweep(summary: Path, output: Path) -> None:
    frame = pd.read_csv(summary).sort_values(["fit_particles", "learning_rate"])
    if frame.empty:
        raise ValueError("particle sweep summary is empty")
    frame["rate_rank"] = frame.groupby("fit_particles")["learning_rate"].rank(
        method="dense"
    )
    frame["schedule"] = frame["rate_rank"].map(
        {1.0: "Lower learning rate", 2.0: "Higher learning rate"}
    )

    sns.set_theme(style="whitegrid", context="paper")
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    palette = {
        "Lower learning rate": "#31688e",
        "Higher learning rate": "#35b779",
    }
    for schedule, subset in frame.groupby("schedule", sort=False):
        color = palette[schedule]
        axes[0].plot(
            subset["fit_particles"],
            subset["euler20_loglik"],
            marker="o",
            linewidth=2,
            color=color,
            label=schedule,
        )
        axes[1].plot(
            subset["fit_particles"],
            subset["updates_completed"],
            marker="o",
            linewidth=2,
            color=color,
            label=schedule,
        )
    axes[0].axhline(BEST_KNOWN, color="red", linestyle=":", linewidth=2)
    axes[0].set_ylabel("Euler-20 Log-Likelihood")
    axes[1].set_ylabel("Score Updates in 400 Seconds")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(sorted(frame["fit_particles"].unique()))
        axis.set_xticklabels(
            [f"{value:,}" for value in sorted(frame["fit_particles"].unique())]
        )
        axis.set_xlabel("Fitting Particles")
    axes[0].legend(frameon=False)
    axes[1].legend().remove()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_particle_sweep(args.summary, args.output)


if __name__ == "__main__":
    main()
