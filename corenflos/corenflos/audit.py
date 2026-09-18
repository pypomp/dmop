"""Completion checks for a Corenflos Dacca benchmark directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ditlevsen.benchmark import make_starts


FIGURES = {
    "likelihood_comparison_r20.png",
    "parameter_comparison_r20.png",
    "optimization_elapsed_r20.png",
    "optimization_elapsed_full_r20.png",
}


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AssertionError(f"missing {path}")
    return pd.read_csv(path)


def audit(
    directory: Path,
    expected_starts: int,
    expected_maximum_invalid_fraction: float = 1.0,
) -> dict[str, float | int]:
    configuration_path = directory / "configuration.json"
    if not configuration_path.exists():
        raise AssertionError(f"missing {configuration_path}")
    configuration = json.loads(configuration_path.read_text())
    _assert_equal(int(configuration["starts"]), expected_starts, "configured starts")
    _assert_equal(int(configuration["nstep"]), 20, "inference Euler substeps")
    _assert_equal(int(configuration["particles"]), 100, "fitting particles")
    _assert_equal(float(configuration["epsilon"]), 0.25, "transport epsilon")
    _assert_equal(
        float(configuration["maximum_acceptable_invalid_fraction"]),
        expected_maximum_invalid_fraction,
        "maximum acceptable invalid fraction",
    )
    _assert_equal(float(configuration["maximum_elapsed_seconds"]), 800.0, "fit budget")
    _assert_equal(
        float(configuration["output_selection_seconds"]), 700.0, "selection cutoff"
    )

    checkpoints = sorted((directory / "checkpoints").glob("fit_start*.npz"))
    _assert_equal(len(checkpoints), expected_starts, "checkpoint count")
    expected_parameters = make_starts(expected_starts, int(configuration["seed"]))
    for index, path in enumerate(checkpoints):
        with np.load(path, allow_pickle=False) as fit:
            np.testing.assert_allclose(
                fit["start"], expected_parameters[index], rtol=0.0, atol=0.0
            )
            parameter_trace = fit["parameter_trace"]
            elapsed = fit["elapsed_trace"]
            _assert_equal(
                parameter_trace.shape[1], 23, f"start {index} parameter width"
            )
            _assert_equal(
                parameter_trace.shape[0], elapsed.shape[0], f"start {index} trace rows"
            )
            if np.any(np.diff(elapsed) < 0.0):
                raise AssertionError(f"start {index}: elapsed time is not monotone")
            selected = int(fit["output_selected_iteration"])
            if not 0 <= selected < len(elapsed):
                raise AssertionError(f"start {index}: selected iteration out of range")

    fits = _read(directory / "fit_summary.csv")
    finals = _read(directory / "final_evaluations.csv")
    training = _read(directory / "optimization_training_traces.csv")
    target = _read(directory / "optimization_euler_traces.csv")
    expected_indices = set(range(expected_starts))
    _assert_equal(set(fits["start"].astype(int)), expected_indices, "fit starts")
    _assert_equal(set(finals["start"].astype(int)), expected_indices, "final starts")
    _assert_equal(int(finals["eval_particles"].nunique()), 1, "final J variants")
    _assert_equal(int(finals["eval_particles"].iloc[0]), 5000, "final J")
    _assert_equal(int(finals["eval_replicates"].iloc[0]), 36, "final replicates")
    _assert_equal(int(finals["evaluation_nstep"].iloc[0]), 20, "final Euler substeps")
    if not finals["evaluation_status"].eq("ok").all():
        raise AssertionError("one or more final evaluations failed")
    if not np.isfinite(finals["euler20_loglik"]).all():
        raise AssertionError("one or more final log likelihoods are non-finite")

    _assert_equal(int(target["eval_particles"].iloc[0]), 5000, "trace J")
    _assert_equal(int(target["eval_replicates"].iloc[0]), 1, "trace replicates")
    if not target["evaluation_status"].eq("ok").all():
        raise AssertionError("one or more trace evaluations failed")
    for start in range(expected_starts):
        training_iterations = np.sort(
            training.loc[training["start"].eq(start), "iteration"].astype(int).unique()
        )
        target_iterations = np.sort(
            target.loc[target["start"].eq(start), "iteration"].astype(int).unique()
        )
        expected_iterations = np.arange(training_iterations[-1] + 1)
        if not np.array_equal(training_iterations, expected_iterations):
            raise AssertionError(f"start {start}: training trace skips an update")
        if not np.array_equal(target_iterations, expected_iterations):
            raise AssertionError(f"start {start}: Euler trace skips an update")

    figure_directory = directory / "figures"
    found_figures = {path.name for path in figure_directory.glob("*.png")}
    missing = FIGURES - found_figures
    if missing:
        raise AssertionError(f"missing figures: {sorted(missing)}")
    pdfs = list(figure_directory.glob("*.pdf"))
    if pdfs:
        raise AssertionError(f"PDF figures are not allowed: {pdfs}")

    result: dict[str, float | int] = {
        "starts": expected_starts,
        "training_trace_rows": len(training),
        "euler_trace_rows": len(target),
        "median_euler20_loglik": float(finals["euler20_loglik"].median()),
        "best_euler20_loglik": float(finals["euler20_loglik"].max()),
        "worst_euler20_loglik": float(finals["euler20_loglik"].min()),
        "time_budget_fits": int(fits["termination_reason"].eq("time-budget").sum()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument(
        "--maximum-acceptable-invalid-fraction", type=float, default=1.0
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.data,
                args.starts,
                args.maximum_acceptable_invalid_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
