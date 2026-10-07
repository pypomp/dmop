"""Compare fixed particle settings, separately for standalone and IF2 starts.

From dmop/corenflos:
  PYTHONPATH=.:../ditlevsen:../../pypomp python -m corenflos.settings_plots

Reads existing results only; does not fit, evaluate, or select individual runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from plotnine import coord_cartesian

from . import warm_plots as w


ROOT = Path(__file__).resolve().parents[2]
HIGH = ROOT / "corenflos/results/particle_increase_j1000_final_100"
LOW = {
    "corenflos_cold": ROOT / "corenflos/results/corenflos_j100_eps025_final_100",
    "corenflos_warm": ROOT / "corenflos/results/if2warm_ifad097_budget_j100_final_100",
    "ditlevsen_cold": ROOT / "ditlevsen/results/block_smc_guided_j100_final_100",
    "ditlevsen_warm": ROOT / "ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100",
}
METHOD_KEYS = {
    "corenflos_cold": "Corenflos", "corenflos_warm": "Corenflos + IF2",
    "ditlevsen_cold": "Ditlevsen", "ditlevsen_warm": "Ditlevsen + IF2",
}


def sources_for(corenflos_particles=100, ditlevsen_particles=1000):
    directories, labels = {}, {}
    for key, method in METHOD_KEYS.items():
        particles = corenflos_particles if key.startswith("corenflos") else ditlevsen_particles
        if particles not in (100, 1000):
            raise ValueError("Complete standalone and warm-start arms require J=100 or J=1000")
        variant = key.replace("_cold", "_vanilla")
        directories[key] = LOW[key] if particles == 100 else HIGH / variant
        labels[method] = f"{w.DISPLAY_LABELS.get(method, method)} (J={particles:,})"
    labels["IF2 warm start"] = "IF2 warm start"
    return SimpleNamespace(
        **directories,
        warm_start=ROOT / "corenflos/results/ifad097_post_if2_reference",
        starts_file=ROOT / "ditlevsen/results/reference/ifad097_comparable_post_if2.npz",
        reference=ROOT / "ditlevsen/results/reference",
    ), labels


def validate_runs(directory, particles):
    config = json.loads((directory / "configuration.json").read_text())
    if config["particles"] != particles or config["starts"] != 100:
        raise ValueError(f"Unexpected fitting settings: {directory}")
    fits = pd.read_csv(directory / "fit_summary.csv").set_index("start").sort_index()
    final = pd.read_csv(directory / "final_evaluations.csv").set_index("start").sort_index()
    for frame in (fits, final):
        if not frame.index.is_unique or frame.index.tolist() != list(range(100)):
            raise ValueError(f"Expected all 100 unique starts: {directory}")
    if not (final.evaluation_status.eq("ok").all()
            and final.eval_particles.eq(5000).all()
            and final.eval_replicates.eq(36).all()
            and np.isfinite(final.euler20_loglik).all()):
        raise ValueError(f"Invalid independent final evaluations: {directory}")
    return config, fits, final


def generate(output, corenflos_particles=100, ditlevsen_particles=1000):
    args, labels = sources_for(corenflos_particles, ditlevsen_particles)
    settings, configurations, final_rows = [], {}, []
    for key, method in METHOD_KEYS.items():
        particles = corenflos_particles if key.startswith("corenflos") else ditlevsen_particles
        directory = getattr(args, key)
        config, fits, final = validate_runs(directory, particles)
        configurations[labels[method]] = config
        settings.append(dict(
            method=labels[method], initialization="IF2" if key.endswith("warm") else "standalone",
            particles=particles, learning_rate=config["learning_rate"],
            fitting_budget_seconds=config["maximum_elapsed_seconds"],
            output_selection_seconds=config["output_selection_seconds"],
            starts=len(final), median_euler_loglik=final.euler20_loglik.median(),
            median_updates=fits.updates_completed.median(), source=str(directory.relative_to(ROOT)),
        ))
        final_rows.append(final.reset_index().assign(method=labels[method]))

    likelihood = w._likelihood_frame(args)
    if len(likelihood) != 600 or not np.isfinite(likelihood.logLik).all():
        raise ValueError("Expected 100 finite final values for each of six methods")
    optimization = w._optimization_frame(args)
    parameters = w._parameter_frame(args)
    mismatch = w._objective_mismatch_frame(args)
    output.mkdir(parents=True, exist_ok=True)

    # The overview is a coordinate zoom, not a data filter: distributions are
    # computed with all starts. A separate uncropped view displays every output.
    w.plot_likelihood(likelihood, output, display_labels=labels, minimum=None,
                      limits=w.OVERVIEW_LIMITS, save_pdf=True, best_known=None)
    w.plot_likelihood(likelihood, output, display_labels=labels, minimum=None,
                      name="likelihood_all_outputs_r20", save_pdf=True, best_known=None)
    w.plot_parameters(parameters, output, display_labels=labels, save_pdf=True)
    w.plot_optimization(optimization, output, display_labels=labels, save_pdf=True,
                        best_known=None)
    uncropped = (
        w._optimization_plot(optimization, ["IF2 warm start", "IFAD-0.97",
                             "Ditlevsen + IF2", "Corenflos + IF2", "Ditlevsen", "Corenflos"],
                             display_labels=labels, best_known=None)
        + coord_cartesian(xlim=(0, float(optimization.elapsed_seconds.max())))
    )
    w._save(uncropped, output, "optimization_all_values_r20", 10.0, 4.0, True)
    w.plot_objective_mismatch(mismatch, output, display_labels=labels, save_pdf=True)

    pd.DataFrame(settings).to_csv(output / "settings.csv", index=False)
    pd.concat(final_rows, ignore_index=True).to_csv(output / "final_all_runs.csv", index=False)
    optimization.assign(method=optimization.source.map(lambda value: labels.get(value, value))).to_csv(
        output / "optimization_summary.csv", index=False)
    mismatch["source"] = mismatch.source.replace(
        {w.DISPLAY_LABELS[key]: labels[key] for key in ("Corenflos + IF2", "Ditlevsen + IF2")})
    mismatch.to_csv(output / "objective_mismatch.csv", index=False)
    (output / "settings.json").write_text(json.dumps(configurations, indent=2) + "\n")
    print(pd.DataFrame(settings).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corenflos-particles", type=int, choices=(100, 1000), default=100)
    parser.add_argument("--ditlevsen-particles", type=int, choices=(100, 1000), default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"imgs/competitors/corenflos{args.corenflos_particles}_ditlevsen{args.ditlevsen_particles}"
    generate(output, args.corenflos_particles, args.ditlevsen_particles)


if __name__ == "__main__":
    main()
