"""Extract the post-IF2 checkpoints used by the IFAD manuscript runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .model import ESTIMATED_PARAMETER_NAMES, encode_parameters
from .reference_results import _entries, _paths, _read_pickle


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def extract_ifad_mif_starts(
    method: str = "IFAD-0.97", effort: str = "comparable"
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact parameter vectors at IFAD's post-MIF boundary.

    The monitored pickle supplies the 100 parameter vectors. The unmonitored
    pickle supplies the runtimes used for the manuscript's elapsed-time axis.
    Initial-value weights are normalized by ``encode_parameters``, matching
    the Dacca initializer's effective state composition.
    """

    if method == "IF2":
        raise ValueError("IF2 has no subsequent IFAD gradient stage")
    if effort not in ("comparable", "extended"):
        raise ValueError("effort must be 'comparable' or 'extended'")
    monitored_path, timing_path, historical = _paths(method, effort)
    monitored = _read_pickle(monitored_path, historical=historical)
    timing = _read_pickle(timing_path, historical=historical)
    monitored_mif = [entry for entry in _entries(monitored) if entry.method == "mif"]
    monitored_train = [
        entry for entry in _entries(monitored) if entry.method == "train"
    ]
    if len(monitored_mif) != 1:
        raise ValueError(
            f"expected one monitored MIF entry, found {len(monitored_mif)}"
        )
    if len(monitored_train) != 1:
        raise ValueError(
            f"expected one monitored training entry, found {len(monitored_train)}"
        )
    timing_entries = [
        entry for entry in _entries(timing) if entry.method in ("mif", "train")
    ]
    timing_mif = [entry for entry in timing_entries if entry.method == "mif"]
    timing_train = [entry for entry in timing_entries if entry.method == "train"]
    if len(timing_mif) != 1 or len(timing_train) != 1:
        raise ValueError("expected one unmonitored MIF and one training entry")

    mif_entry = monitored_mif[0]
    train_entry = monitored_train[0]

    # ``Result.theta`` stores the parameters supplied *to* that operation.
    # The MIF entry therefore contains the global-search inputs, while the
    # following training entry contains the post-MIF checkpoints that IFAD
    # actually uses.  Cross-check those inputs against the last MIF trace so
    # this distinction cannot silently regress again.
    starts = np.stack([encode_parameters(theta) for theta in train_entry.theta])
    # Remove the softmax null shift without changing the effective IVP.
    starts[:, 18:23] -= np.max(starts[:, 18:23], axis=1, keepdims=True)

    final_iteration = int(np.max(mif_entry.traces_da.coords["iteration"].values))
    final_trace = mif_entry.traces_da.sel(iteration=final_iteration)
    trace_variables = [str(value) for value in final_trace.coords["variable"].values]
    trace_rows = np.asarray(final_trace.values, dtype=float)
    trace_starts = np.stack(
        [
            encode_parameters(dict(zip(trace_variables, row, strict=True)))
            for row in trace_rows
        ]
    )
    trace_starts[:, 18:23] -= np.max(
        trace_starts[:, 18:23], axis=1, keepdims=True
    )
    np.testing.assert_allclose(starts, trace_starts, rtol=2e-6, atol=2e-6)

    if starts.ndim != 2 or starts.shape[1] != len(ESTIMATED_PARAMETER_NAMES):
        raise ValueError(f"unexpected checkpoint shape {starts.shape}")
    if not np.isfinite(starts).all():
        raise ValueError("post-MIF checkpoint contains non-finite values")

    mif_seconds = float(timing_mif[0].execution_time)
    train_seconds = float(timing_train[0].execution_time)
    metadata: dict[str, Any] = {
        "source_method": method,
        "source_effort": effort,
        "source_monitored_pickle": monitored_path,
        "source_timing_pickle": timing_path,
        "checkpoint_source": "input parameters of monitored IFAD training stage",
        "checkpoint_verified_against_mif_iteration": final_iteration,
        "timing_basis": "unmonitored component runtime",
        "starts": int(starts.shape[0]),
        "parameters": list(ESTIMATED_PARAMETER_NAMES),
        "if2_particles": int(mif_entry.J),
        "if2_iterations": int(mif_entry.M),
        "if2_elapsed_seconds": mif_seconds,
        "gradient_particles": int(timing_train[0].J),
        "gradient_iterations": int(timing_train[0].M),
        "remaining_elapsed_seconds": train_seconds,
        "total_elapsed_seconds": mif_seconds + train_seconds,
    }
    return starts, metadata


def write_ifad_mif_starts(
    output: Path,
    method: str = "IFAD-0.97",
    effort: str = "comparable",
) -> dict[str, Any]:
    starts, metadata = extract_ifad_mif_starts(method, effort)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output, starts=starts)
    _atomic_json(output.with_suffix(".json"), metadata)
    return metadata


def load_starts_file(path: Path, count: int) -> list[np.ndarray]:
    """Load and validate a benchmark start matrix without pickle support."""

    with np.load(path, allow_pickle=False) as archive:
        if "starts" not in archive:
            raise ValueError(f"{path} has no 'starts' array")
        starts = np.asarray(archive["starts"], dtype=float)
    expected_width = len(ESTIMATED_PARAMETER_NAMES)
    if starts.ndim != 2 or starts.shape[1] != expected_width:
        raise ValueError(
            f"expected start matrix (*, {expected_width}), found {starts.shape}"
        )
    if count < 1 or count > starts.shape[0]:
        raise ValueError(f"requested {count} starts from a file with {starts.shape[0]}")
    if not np.isfinite(starts[:count]).all():
        raise ValueError("start file contains non-finite values")
    return [row.copy() for row in starts[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("IFAD-0", "IFAD-0.97", "IFAD-1"), default="IFAD-0.97"
    )
    parser.add_argument(
        "--effort", choices=("comparable", "extended"), default="comparable"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_ifad_mif_starts(args.output, args.method, args.effort), indent=2
        )
    )


if __name__ == "__main__":
    main()
