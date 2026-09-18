"""Completion audit for the IFAD-0.97 IF2-warm-start comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ditlevsen.warm_starts import extract_ifad_mif_starts


OFFSET = 244.44243359565735
REMAINING = 614.6859633922577
TOTAL = 859.128396987915
FIGURES = {
    "likelihood_if2warm_comparison_r20.png",
    "parameter_if2warm_comparison_r20.png",
    "optimization_if2warm_elapsed_r20.png",
    "optimization_if2warm_elapsed_full_r20.png",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AssertionError(f"missing {path}")
    return pd.read_csv(path)


def _configuration(path: Path) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"missing {path}")
    return json.loads(path.read_text())


def _check_trace_iterations(
    training: pd.DataFrame, target: pd.DataFrame, starts: int
) -> None:
    for start in range(starts):
        expected = np.sort(
            training.loc[training["start"].eq(start), "iteration"].astype(int).unique()
        )
        observed = np.sort(
            target.loc[target["start"].eq(start), "iteration"].astype(int).unique()
        )
        contiguous = np.arange(expected[-1] + 1)
        if not np.array_equal(expected, contiguous):
            raise AssertionError(f"training trace skips start {start}")
        if not np.array_equal(observed, contiguous):
            raise AssertionError(f"Euler trace skips start {start}")


def _check_method(
    directory: Path,
    starts: np.ndarray,
    *,
    corenflos: bool,
) -> dict[str, float | int]:
    config = _configuration(directory / "configuration.json")
    if int(config["starts"]) != starts.shape[0]:
        raise AssertionError("wrong start count")
    if int(config["particles"]) != 100:
        raise AssertionError("fitting particle count is not 100")
    if float(config["elapsed_time_offset_seconds"]) != OFFSET:
        raise AssertionError("wrong IF2 elapsed-time offset")
    if float(config["maximum_elapsed_seconds"]) != REMAINING:
        raise AssertionError("wrong remaining elapsed-time budget")
    if float(config["output_selection_seconds"]) != TOTAL:
        raise AssertionError("wrong total selection time")
    update_key = "maximum_updates" if corenflos else "iterations"
    if int(config[update_key]) != 175:
        raise AssertionError("wrong gradient-update cap")
    nstep = int(config["nstep"] if corenflos else config["nsteps"][0])
    if nstep != 20:
        raise AssertionError("inference is not Euler-20")

    pattern = "fit_start*.npz" if corenflos else "fit_r20_start*.npz"
    checkpoints = sorted((directory / "checkpoints").glob(pattern))
    if len(checkpoints) != starts.shape[0]:
        raise AssertionError(
            f"expected {starts.shape[0]} checkpoints, found {len(checkpoints)}"
        )
    for start, path in enumerate(checkpoints):
        with np.load(path, allow_pickle=False) as fit:
            np.testing.assert_array_equal(fit["start"], starts[start])
            np.testing.assert_allclose(
                fit["elapsed_trace"] - fit["optimizer_elapsed_trace"],
                OFFSET,
                rtol=0.0,
                atol=1e-10,
            )
            if float(fit["elapsed_time_offset_seconds"]) != OFFSET:
                raise AssertionError(f"start {start}: wrong checkpoint offset")

    training = _read(directory / "optimization_training_traces.csv")
    target = _read(directory / "optimization_euler_traces.csv")
    finals = _read(directory / "final_evaluations.csv")
    _check_trace_iterations(training, target, starts.shape[0])
    if len(training) != len(target):
        raise AssertionError("training and Euler trace row counts differ")
    if int(finals["start"].nunique()) != starts.shape[0]:
        raise AssertionError("final evaluation start count differs")
    if not finals["evaluation_status"].eq("ok").all():
        raise AssertionError("one or more final evaluations failed")
    if not target["evaluation_status"].eq("ok").all():
        raise AssertionError("one or more trace evaluations failed")
    if (
        not finals["eval_particles"].eq(5000).all()
        or not finals["eval_replicates"].eq(36).all()
    ):
        raise AssertionError("final evaluation effort differs")
    if (
        not target["eval_particles"].eq(5000).all()
        or not target["eval_replicates"].eq(1).all()
    ):
        raise AssertionError("trace evaluation effort differs")
    likelihood_column = "euler20_loglik"
    return {
        "trace_rows": int(len(target)),
        "median": float(finals[likelihood_column].median()),
        "best": float(finals[likelihood_column].max()),
        "at_least_-4300": int(finals[likelihood_column].ge(-4300.0).sum()),
    }


def audit(
    corenflos: Path,
    ditlevsen: Path,
    warm_start: Path,
    figures: Path,
) -> dict[str, object]:
    expected_starts, metadata = extract_ifad_mif_starts()
    with np.load(warm_start, allow_pickle=False) as archive:
        starts = np.asarray(archive["starts"], dtype=float)
    np.testing.assert_array_equal(starts, expected_starts)
    if metadata["if2_elapsed_seconds"] != OFFSET:
        raise AssertionError("reference MIF runtime changed")
    if metadata["remaining_elapsed_seconds"] != REMAINING:
        raise AssertionError("reference training runtime changed")
    if metadata["total_elapsed_seconds"] != TOTAL:
        raise AssertionError("reference total runtime changed")

    baseline = _read(
        corenflos.parent / "ifad097_post_if2_reference" / "warm_start_evaluations.csv"
    )
    if (
        len(baseline) != starts.shape[0]
        or not baseline["evaluation_status"].eq("ok").all()
    ):
        raise AssertionError("IF2 checkpoint evaluations are incomplete")
    if (
        not baseline["eval_particles"].eq(5000).all()
        or not baseline["eval_replicates"].eq(36).all()
    ):
        raise AssertionError("IF2 checkpoint evaluation effort differs")

    found = {path.name for path in figures.glob("*.png")}
    missing = FIGURES - found
    if missing:
        raise AssertionError(f"missing figures: {sorted(missing)}")
    pdfs = list(figures.glob("*.pdf"))
    if pdfs:
        raise AssertionError(f"PDF figures are not allowed: {pdfs}")

    return {
        "starts": int(starts.shape[0]),
        "if2_elapsed_seconds": OFFSET,
        "remaining_elapsed_seconds": REMAINING,
        "total_elapsed_seconds": TOTAL,
        "IF2 checkpoint": {
            "median": float(baseline["euler20_loglik"].median()),
            "best": float(baseline["euler20_loglik"].max()),
        },
        "Corenflos + IF2": _check_method(corenflos, starts, corenflos=True),
        "Ditlevsen + IF2": _check_method(ditlevsen, starts, corenflos=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corenflos",
        type=Path,
        default=Path("results/if2warm_ifad097_budget_j100_final_100"),
    )
    parser.add_argument(
        "--ditlevsen",
        type=Path,
        default=Path(
            "../ditlevsen/results/block_smc_guided_j100_if2warm_ifad097_budget_final_100"
        ),
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path("../ditlevsen/results/reference/ifad097_comparable_post_if2.npz"),
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("results/if2warm_ifad097_comparison/figures"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.corenflos, args.ditlevsen, args.warm_start, args.figures),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
