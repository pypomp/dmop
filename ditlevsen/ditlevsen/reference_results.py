"""Export the exact DMOP manuscript results to compact comparison CSVs.

The IFAD pickle files remain in ``code/global_search/dmop_results``.  The IF2
files used to make the manuscript figures were deleted in commit 7dc0a6a but
remain available in this repository's history.  This module reads both sets
without restoring or modifying the manuscript checkout.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DMOP_ROOT = Path(__file__).resolve().parents[2]
IF2_REVISION = "7dc0a6a^"
KEY_PARAMETERS = (
    "sigma",
    "tau",
    "gamma",
    "epsilon",
    "S_0",
    "I_0",
    "R1_0",
    "bs3",
    "omegas5",
)


class _LegacyResult:
    """State-only shim for pre-unification Pypomp result classes."""


def _install_pickle_shims() -> None:
    import pypomp.core.results.pomp as result_module

    for name in (
        "PompBaseResult",
        "PompPFilterResult",
        "PompMIFResult",
        "PompTrainResult",
    ):
        if not hasattr(result_module, name):
            setattr(result_module, name, _LegacyResult)


def _read_pickle(relative_path: str, *, historical: bool) -> object:
    _install_pickle_shims()
    if historical:
        completed = subprocess.run(
            ["git", "show", f"{IF2_REVISION}:{relative_path}"],
            cwd=DMOP_ROOT,
            check=True,
            capture_output=True,
        )
        payload = completed.stdout
    else:
        payload = (DMOP_ROOT / relative_path).read_bytes()
    return pickle.loads(payload)


def _entries(model: object) -> list[object]:
    return list(model.results_history._entries)


def _full_pfilter_entry(model: object) -> object:
    candidates = [entry for entry in _entries(model) if entry.method == "pfilter"]
    if not candidates:
        raise ValueError("result object has no particle-filter evaluation")
    return max(candidates, key=lambda entry: int(entry.logLiks.shape[0]))


def _logmeanexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.mean(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _paths(method: str, effort: str) -> tuple[str, str, bool]:
    suffix = "_long" if effort == "extended" else ""
    if method == "IF2":
        stem = f"code/global_search/mif_results/dacca_results_rl4_nm{{monitor}}{suffix}.pkl"
        return stem.format(monitor=1), stem.format(monitor=0), True
    alpha = {"IFAD-0": "0.0", "IFAD-0.97": "0.97", "IFAD-1": "1.0"}[method]
    stem = (
        "code/global_search/dmop_results/"
        f"dacca_results_rl4_alpha{alpha}_nm{{monitor}}{suffix}.pkl"
    )
    return stem.format(monitor=1), stem.format(monitor=0), False


def export_reference_results(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    likelihood_rows: list[dict[str, float | str | int]] = []
    parameter_rows: list[dict[str, float | str | int]] = []
    trace_rows: list[dict[str, float | str | int]] = []

    methods = ("IFAD-0", "IFAD-0.97", "IFAD-1", "IF2")
    for effort in ("comparable", "extended"):
        for method in methods:
            monitored_path, timing_path, historical = _paths(method, effort)
            monitored = _read_pickle(monitored_path, historical=historical)
            timing = _read_pickle(timing_path, historical=historical)

            pfilter = _full_pfilter_entry(monitored)
            estimates = _logmeanexp(np.asarray(pfilter.logLiks, dtype=float), axis=1)
            for replicate, (loglik, parameters) in enumerate(
                zip(estimates, pfilter.theta, strict=True)
            ):
                likelihood_rows.append(
                    {
                        "method": method,
                        "effort": effort,
                        "replicate": replicate,
                        "euler_loglik": float(loglik),
                    }
                )
                parameter_rows.append(
                    {
                        "method": method,
                        "effort": effort,
                        "replicate": replicate,
                        **{name: float(parameters[name]) for name in KEY_PARAMETERS},
                    }
                )

            runtime_by_stage: dict[str, float] = {}
            for entry in _entries(timing):
                if entry.method in ("mif", "train"):
                    runtime_by_stage[entry.method] = (
                        runtime_by_stage.get(entry.method, 0.0)
                        + float(entry.execution_time)
                    )
            stage_start = 0.0
            for entry in _entries(monitored):
                if entry.method not in ("mif", "train"):
                    continue
                values = np.asarray(entry.traces_da.sel(variable="logLik"), dtype=float)
                iterations = np.asarray(entry.traces_da.coords["iteration"], dtype=int)
                denominator = max(int(iterations.max() - iterations.min()), 1)
                elapsed = stage_start + (
                    (iterations - iterations.min())
                    * runtime_by_stage[entry.method]
                    / denominator
                )
                for column, seconds in enumerate(elapsed):
                    finite_values = values[:, column]
                    finite_values = finite_values[np.isfinite(finite_values)]
                    if finite_values.size == 0:
                        continue
                    trace_rows.append(
                        {
                            "method": method,
                            "effort": effort,
                            "stage": entry.method,
                            "stage_iteration": int(iterations[column]),
                            "elapsed_seconds": float(seconds),
                            "median": float(np.median(finite_values)),
                            "q10": float(np.percentile(finite_values, 10)),
                            "maximum": float(np.max(finite_values)),
                            "replicates": int(finite_values.size),
                            "timing_basis": "monitored trace scaled to unmonitored component runtime",
                        }
                    )
                stage_start += runtime_by_stage[entry.method]

    pd.DataFrame(likelihood_rows).to_csv(
        output / "manuscript_likelihood.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        output / "manuscript_parameters.csv", index=False
    )
    pd.DataFrame(trace_rows).to_csv(
        output / "manuscript_trace_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/reference"))
    args = parser.parse_args()
    export_reference_results(args.output)


if __name__ == "__main__":
    main()
