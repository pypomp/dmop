"""Reproducible Dacca benchmark driver.

Run from ``dmop/ditlevsen`` with the dedicated Pypomp environment, for example::

    PYTHONPATH=.:../.. /home/kevin/anaconda3/envs/pypomp/bin/python \
        -m ditlevsen.benchmark --nsteps 5 10 20 --starts 3 --iterations 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from .data import load_dacca_data
from .ekf import filtered_means, fit_ekf
from .mixing import backward_ess_experiment
from .model import (
    DEFAULT_PARAMETERS,
    ESTIMATED_PARAMETER_NAMES,
    INITIAL_PARAMETER_NAMES,
    RESERVOIR_PARAMETER_NAMES,
    SPLINE_PARAMETER_NAMES,
    default_unconstrained_parameters,
    encode_parameters,
    initial_state,
    physical_parameter_dict,
)
from .transition import transition_diagnostics
from .transition import lie_bracket_diagnostics


METHOD_ORDERS = {
    "DS-local+nugget QML": 2,
    "DS-Krylov QML": 6,
}


def _sample_global_parameters(rng: np.random.Generator) -> dict[str, float]:
    """Sample the physical-scale global box used in the DMOP manuscript."""

    result = dict(DEFAULT_PARAMETERS)
    ranges = {
        "gamma": (10.0, 40.0),
        "m": (0.03, 0.60),
        "epsilon": (0.20, 30.0),
        "beta_trend": (-0.01, 0.0),
        "sigma": (1.0, 5.0),
        "tau": (0.10, 0.50),
        "bs1": (-4.0, 4.0),
        "bs2": (0.0, 8.0),
        "bs3": (-4.0, 4.0),
        "bs4": (0.0, 8.0),
        "bs5": (0.0, 8.0),
        "bs6": (0.0, 8.0),
        **{name: (-10.0, 0.0) for name in RESERVOIR_PARAMETER_NAMES},
    }
    for name, (lower, upper) in ranges.items():
        result[name] = float(rng.uniform(lower, upper))
    initial = rng.uniform(0.0, 1.0, size=len(INITIAL_PARAMETER_NAMES))
    initial /= initial.sum()
    result.update(dict(zip(INITIAL_PARAMETER_NAMES, initial)))
    return result


def make_starts(count: int, seed: int) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("starts must be positive")
    rng = np.random.default_rng(seed)
    return [
        encode_parameters(_sample_global_parameters(rng)) for _ in range(count)
    ]


def _parameter_columns(unconstrained: np.ndarray) -> dict[str, float]:
    physical = physical_parameter_dict(unconstrained)
    result = {
        name: float(physical[name]) for name in ESTIMATED_PARAMETER_NAMES
    }
    result.update(
        {f"raw_{index}": float(value) for index, value in enumerate(unconstrained)}
    )
    return result


def run_fits(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    starts = make_starts(args.starts, args.seed)
    summaries: list[dict[str, float | str | bool]] = []
    trace_rows: list[dict[str, float | str | bool]] = []

    for nstep in args.nsteps:
        data = load_dacca_data(nstep, max_observations=args.max_observations)
        if args.krylov_only:
            methods = (("DS-Krylov QML", 6),)
        elif args.completed:
            methods = METHOD_ORDERS.items()
        else:
            methods = (("DS-local+nugget QML", 2),)
        for method, order in methods:
            for start_index, start in enumerate(starts):
                result = fit_ekf(
                    start,
                    data,
                    order=order,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    relative_floor=args.relative_floor,
                )
                for iteration, (value, gradient_norm, elapsed, parameters) in enumerate(
                    zip(
                        result.objective_trace,
                        result.gradient_norm_trace,
                        result.elapsed_trace,
                        result.parameter_trace,
                        strict=True,
                    )
                ):
                    trace_rows.append(
                        {
                            "method": method,
                            "order": order,
                            "inference_nstep": nstep,
                            "start": start_index,
                            "iteration": iteration,
                            "elapsed_seconds": elapsed,
                            "approx_loglik": value,
                            "gradient_norm": gradient_norm,
                            **_parameter_columns(parameters),
                        }
                    )
                summaries.append(
                    {
                        "method": method,
                        "order": order,
                        "inference_nstep": nstep,
                        "start": start_index,
                        "start_type": "global-box",
                        "approx_start_loglik": result.objective_trace[0],
                        "approx_final_loglik": result.objective_trace[-1],
                        "iterations_completed": len(result.objective_trace) - 1,
                        "elapsed_seconds": result.elapsed_seconds,
                        "converged": result.converged,
                        **_parameter_columns(result.unconstrained),
                    }
                )
                print(
                    f"{method} nstep={nstep} start={start_index}: "
                    f"{result.objective_trace[0]:.2f} -> "
                    f"{result.objective_trace[-1]:.2f} "
                    f"({result.elapsed_seconds:.1f}s)",
                    flush=True,
                )

    pd.DataFrame(trace_rows).to_csv(output_dir / "optimization_traces.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "fit_summary.csv", index=False)
    return summary


def evaluate_optimization_traces(
    args: argparse.Namespace, output_dir: Path
) -> pd.DataFrame:
    """Monitor DS optimization under the common Euler-20 likelihood target.

    The optimizer is timed without these particle-filter calls.  Evaluations
    are done afterward, just as the manuscript obtains likelihood traces from
    monitored runs but uses the corresponding unmonitored runtimes on the
    horizontal axis.
    """

    import pypomp as pp
    from pypomp.core.parameters import PompParameters

    traces = pd.read_csv(output_dir / "optimization_traces.csv")
    raw_columns = [f"raw_{index}" for index in range(23)]
    rows: list[dict[str, float | str | int]] = []
    model = pp.models.dacca(nstep=args.evaluation_nstep, dt=None)
    finite = traces.loc[
        traces["approx_loglik"].notna()
        & np.isfinite(traces[raw_columns]).all(axis=1)
    ].copy()
    for candidate_index, (_, row) in enumerate(finite.iterrows()):
        unconstrained = row[raw_columns].to_numpy(dtype=float)
        theta = PompParameters(physical_parameter_dict(unconstrained))
        try:
            model.pfilter(
                theta=theta,
                J=args.trace_eval_particles,
                reps=args.trace_eval_replicates,
                key=jax.random.key(args.seed + 100_000 + candidate_index),
            )
            evaluation = model.results().iloc[0]
            loglik = float(evaluation["logLik"])
            standard_error = float(evaluation["se"])
            status = "ok" if np.isfinite(loglik) else "non-finite-evaluation"
        except (FloatingPointError, RuntimeError, ValueError) as error:
            loglik = np.nan
            standard_error = np.nan
            status = f"evaluation-failed:{type(error).__name__}"
        rows.append(
            {
                "method": row["method"],
                "inference_nstep": int(row["inference_nstep"]),
                "start": int(row["start"]),
                "iteration": int(row["iteration"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "approx_loglik": float(row["approx_loglik"]),
                "euler_loglik": loglik,
                "euler_se": standard_error,
                "evaluation_status": status,
                "eval_particles": args.trace_eval_particles,
                "eval_replicates": args.trace_eval_replicates,
            }
        )
        print(
            f"trace eval nstep={int(row['inference_nstep'])} "
            f"start={int(row['start'])} iter={int(row['iteration'])}: "
            f"{loglik:.2f} ({status})",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "target_optimization_traces.csv", index=False)
    return frame


def run_transition_diagnostics(
    args: argparse.Namespace, output_dir: Path
) -> pd.DataFrame:
    unconstrained = default_unconstrained_parameters()
    rows: list[dict[str, float]] = []
    for nstep in args.diagnostic_nsteps:
        data = load_dacca_data(nstep, max_observations=1)
        state = np.concatenate(
            (
                np.asarray(
                    [DEFAULT_PARAMETERS[name] for name in INITIAL_PARAMETER_NAMES]
                )
                / sum(DEFAULT_PARAMETERS[name] for name in INITIAL_PARAMETER_NAMES),
                [0.0],
            )
        )
        for order in range(1, 7):
            row = transition_diagnostics(
                state,
                unconstrained,
                data.initial_covariates,
                1.0 / (12.0 * nstep),
                order=order,
            )
            rows.append({"inference_nstep": nstep, "order": order, **row})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "transition_rank.csv", index=False)
    return frame


def run_bracket_diagnostics(output_dir: Path) -> pd.DataFrame:
    data = load_dacca_data(20, max_observations=1)
    unconstrained = default_unconstrained_parameters()
    state = np.asarray(initial_state(unconstrained))
    frame = pd.DataFrame(
        lie_bracket_diagnostics(
            state,
            unconstrained,
            data.initial_covariates,
            maximum_depth=5,
        )
    )
    frame.to_csv(output_dir / "bracket_rank.csv", index=False)
    return frame


def run_taylor_stability_diagnostics(
    args: argparse.Namespace, output_dir: Path
) -> pd.DataFrame:
    """Track when the DS mean/filter first leaves a usable state region."""

    rows: list[dict[str, float | int | bool]] = []
    unconstrained = default_unconstrained_parameters()
    stability_nsteps = (
        args.stability_nsteps
        if getattr(args, "stability_nsteps", None) is not None
        else args.nsteps
    )
    for nstep in stability_nsteps:
        data = load_dacca_data(nstep, max_observations=args.max_observations)
        means, increments = filtered_means(unconstrained, data, order=2)
        for month, (mean, increment) in enumerate(
            zip(means, increments, strict=True), start=1
        ):
            active = mean[:5]
            rows.append(
                {
                    "inference_nstep": nstep,
                    "month": month,
                    "finite_loglik_increment": bool(np.isfinite(increment)),
                    "state_in_box": bool(
                        np.all(np.isfinite(active))
                        and np.all(active >= 0.0)
                        and np.all(active <= 1.2)
                    ),
                    "minimum_filtered_state": float(np.min(active)),
                    "maximum_filtered_state": float(np.max(active)),
                    "maximum_absolute_filtered_state": float(
                        np.max(np.abs(active))
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "taylor_stability.csv", index=False)
    return frame


def run_mixing_diagnostics(
    args: argparse.Namespace, output_dir: Path
) -> pd.DataFrame:
    rows = []
    for nstep in args.diagnostic_nsteps:
        data = load_dacca_data(nstep, max_observations=1)
        row = backward_ess_experiment(
            data,
            default_unconstrained_parameters(),
            particles=args.mixing_particles,
            replicates=args.mixing_replicates,
            seed=args.seed + nstep,
            parent_spread=args.parent_spread,
        )
        rows.append(row)
        print(
            f"mixing nstep={nstep}: unit ESS={row['unit_backward_ess_mean']:.2f}, "
            f"bridge ESS={row['bridge_backward_ess_mean']:.2f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "backward_mixing.csv", index=False)
    return frame


def run_mixing_sensitivity(
    args: argparse.Namespace, output_dir: Path
) -> pd.DataFrame:
    """Repeat the mixing diagnostic over fixed physical parent spreads."""

    rows = []
    for parent_spread in args.mixing_sensitivity_spreads:
        for nstep in args.diagnostic_nsteps:
            if nstep == 1:
                continue
            data = load_dacca_data(nstep, max_observations=1)
            row = backward_ess_experiment(
                data,
                default_unconstrained_parameters(),
                particles=args.mixing_particles,
                replicates=args.mixing_sensitivity_replicates,
                seed=args.seed + nstep + int(parent_spread * 1e6),
                parent_spread=parent_spread,
            )
            rows.append({"parent_spread": parent_spread, **row})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "backward_mixing_sensitivity.csv", index=False)
    return frame


def evaluate_with_pypomp(
    args: argparse.Namespace, summary: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Evaluate every fit under the paper's fixed Euler numerical model.

    ``inference_nstep`` controls only the grid used by the Ditlevsen Gaussian
    criterion.  It must not also change the evaluation target: the manuscript
    defines and reports the Dacca likelihood with 20 Euler substeps per month.
    """

    import pypomp as pp
    from pypomp.core.parameters import PompParameters

    rows: list[dict[str, float | str | int]] = []
    raw_columns = [f"raw_{index}" for index in range(23)]
    candidates: list[tuple[str, int, int, bool, np.ndarray]] = [
        (
            "Published parameter point",
            0,
            0,
            True,
            default_unconstrained_parameters(),
        )
    ]
    for _, row in summary.iterrows():
        candidates.append(
            (
                str(row["method"]),
                int(row["inference_nstep"]),
                int(row["start"]),
                bool(row["converged"]),
                row[raw_columns].to_numpy(dtype=float),
            )
        )

    # Evaluate separately to keep GPU memory bounded and make failures local.
    for candidate_index, (
        method,
        inference_nstep,
        start,
        fit_converged,
        unconstrained,
    ) in enumerate(candidates):
        if not fit_converged or not np.all(np.isfinite(unconstrained)):
            rows.append(
                {
                    "method": method,
                    "inference_nstep": inference_nstep,
                    "evaluation_nstep": args.evaluation_nstep,
                    "start": start,
                    "fit_converged": False,
                    "evaluation_status": "fit-failed",
                    "euler_loglik": np.nan,
                    "euler_se": np.nan,
                    "eval_particles": args.eval_particles,
                    "eval_replicates": args.eval_replicates,
                }
            )
            print(
                f"Euler-{args.evaluation_nstep} eval skipped for {method} "
                f"fit_nstep={inference_nstep} start={start}: fit failed",
                flush=True,
            )
            continue
        try:
            theta = PompParameters(physical_parameter_dict(unconstrained))
            model = pp.models.dacca(nstep=args.evaluation_nstep, dt=None)
            model.pfilter(
                theta=theta,
                J=args.eval_particles,
                reps=args.eval_replicates,
                key=jax.random.key(args.seed + candidate_index),
            )
            result = model.results().iloc[0]
            loglik = float(result["logLik"])
            standard_error = float(result["se"])
            status = "ok" if np.isfinite(loglik) else "non-finite-evaluation"
        except (FloatingPointError, RuntimeError, ValueError) as error:
            loglik = np.nan
            standard_error = np.nan
            status = f"evaluation-failed:{type(error).__name__}"
        rows.append(
            {
                "method": method,
                "inference_nstep": inference_nstep,
                "evaluation_nstep": args.evaluation_nstep,
                "start": start,
                "fit_converged": True,
                "evaluation_status": status,
                "euler_loglik": loglik,
                "euler_se": standard_error,
                "eval_particles": args.eval_particles,
                "eval_replicates": args.eval_replicates,
            }
        )
        print(
            f"Euler-{args.evaluation_nstep} eval {method} "
            f"fit_nstep={inference_nstep} start={start}: "
            f"{loglik:.2f} (SE {standard_error:.2f}; {status})",
            flush=True,
        )

    # This is copied from the authoritative table included by ms.tex.  It is not
    # a fresh run and is labeled as such in every output.
    rows.append(
        {
            "method": "IFAD-0.97 (paper)",
            "inference_nstep": np.nan,
            "evaluation_nstep": 20,
            "start": -1,
            "fit_converged": True,
            "evaluation_status": "paper-value",
            "euler_loglik": -3744.17,
            "euler_se": np.nan,
            "eval_particles": 5000,
            "eval_replicates": 36,
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "euler_loglik.csv", index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsteps", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument(
        "--diagnostic-nsteps", nargs="+", type=int, default=[1, 2, 5, 10, 20, 40]
    )
    parser.add_argument(
        "--stability-nsteps", nargs="+", type=int, default=None
    )
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--relative-floor", type=float, default=1e-10)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--completed", action="store_true")
    parser.add_argument("--krylov-only", action="store_true")
    parser.add_argument("--eval-particles", type=int, default=2000)
    parser.add_argument("--eval-replicates", type=int, default=8)
    parser.add_argument("--trace-evaluation", action="store_true")
    parser.add_argument("--trace-eval-particles", type=int, default=5000)
    parser.add_argument("--trace-eval-replicates", type=int, default=4)
    parser.add_argument("--evaluation-nstep", type=int, default=20)
    parser.add_argument("--mixing-particles", type=int, default=128)
    parser.add_argument("--mixing-replicates", type=int, default=200)
    parser.add_argument("--mixing-sensitivity-replicates", type=int, default=100)
    parser.add_argument(
        "--mixing-sensitivity-spreads",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1],
    )
    parser.add_argument("--parent-spread", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--output", type=Path, default=Path("results/data"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.completed and args.krylov_only:
        raise ValueError("--completed and --krylov-only are mutually exclusive")
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration["output"] = str(configuration["output"])
    (args.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    run_transition_diagnostics(args, args.output)
    run_bracket_diagnostics(args.output)
    run_mixing_diagnostics(args, args.output)
    run_mixing_sensitivity(args, args.output)
    run_taylor_stability_diagnostics(args, args.output)
    summary = run_fits(args, args.output)
    evaluate_with_pypomp(args, summary, args.output)
    if args.trace_evaluation:
        evaluate_optimization_traces(args, args.output)


if __name__ == "__main__":
    main()
