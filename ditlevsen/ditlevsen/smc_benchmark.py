"""Resumable benchmark for the Dacca Ditlevsen-style pseudo-score method.

The fitting target is the regularized Gaussian transition pseudo-model.  The
scientific target is always the manuscript's Euler-20 POMP likelihood, which is
evaluated independently with Pypomp.  Keeping those columns and stages
separate prevents the surrogate score from being reported as the unavailable
score of the original Dacca process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from .benchmark import _parameter_columns, make_starts
from .data import load_dacca_data
from .model import physical_parameter_dict
from .smc import fit_pseudo_score


FIT_SIGNATURE_KEYS = (
    "seed",
    "starts",
    "nsteps",
    "particles",
    "iterations",
    "learning_rate",
    "burnin",
    "gain_exponent",
    "order",
    "relative_floor",
    "gradient_clip",
    "maximum_backtracks",
    "maximum_consecutive_rejections",
    "maximum_path_retries",
    "backtrack_factor",
    "maximum_acceptable_invalid_fraction",
    "maximum_elapsed_seconds",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _json_ready(value) for key, value in vars(args).items()}


def _prepare_output(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    (args.output / "evaluations").mkdir(exist_ok=True)
    (args.output / "trace_evaluations").mkdir(exist_ok=True)
    path = args.output / "configuration.json"
    configuration = _configuration(args)
    if path.exists():
        previous = json.loads(path.read_text())
        changed = [
            key
            for key in FIT_SIGNATURE_KEYS
            if previous.get(key) != configuration.get(key)
        ]
        if changed:
            raise ValueError(
                "output contains a different fit configuration; changed keys: "
                + ", ".join(changed)
            )
    else:
        _atomic_json(path, configuration)


def _checkpoint_path(output: Path, nstep: int, start: int) -> Path:
    return output / "checkpoints" / f"fit_r{nstep:02d}_start{start:03d}.npz"


def _evaluation_path(output: Path, nstep: int, start: int) -> Path:
    return output / "evaluations" / f"euler_r{nstep:02d}_start{start:03d}.json"


def _trace_evaluation_path(
    output: Path, nstep: int, start: int, iteration: int
) -> Path:
    return (
        output
        / "trace_evaluations"
        / f"euler_r{nstep:02d}_start{start:03d}_iter{iteration:03d}.json"
    )


def _fit_seed(seed: int, nstep: int, start: int) -> int:
    return seed + 1_000_003 * nstep + 10_007 * start


def _eval_seed(seed: int, nstep: int, start: int, iteration: int) -> int:
    return seed + 30_000_001 + 100_003 * nstep + 997 * start + iteration


def _fit_one(
    args: argparse.Namespace, start_index: int, start: np.ndarray, nstep: int
) -> None:
    checkpoint = _checkpoint_path(args.output, nstep, start_index)
    if checkpoint.exists():
        print(f"fit exists: r={nstep} start={start_index}", flush=True)
        return
    data = load_dacca_data(nstep)
    result = fit_pseudo_score(
        start,
        data,
        particles=args.particles,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        burnin=args.burnin,
        gain_exponent=args.gain_exponent,
        seed=_fit_seed(args.seed, nstep, start_index),
        order=args.order,
        relative_floor=args.relative_floor,
        gradient_clip=args.gradient_clip,
        bridge_particles=0,
        maximum_backtracks=args.maximum_backtracks,
        maximum_consecutive_rejections=args.maximum_consecutive_rejections,
        maximum_path_retries=args.maximum_path_retries,
        backtrack_factor=args.backtrack_factor,
        maximum_acceptable_invalid_fraction=(
            args.maximum_acceptable_invalid_fraction
        ),
        maximum_elapsed_seconds=args.maximum_elapsed_seconds,
    )
    _atomic_npz(
        checkpoint,
        start=np.asarray(start),
        unconstrained=result.unconstrained,
        parameter_trace=result.parameter_trace,
        pseudo_loglik_trace=result.marginal_loglik_trace,
        complete_pseudologlik_trace=result.complete_loglik_trace,
        score_norm_trace=result.score_norm_trace,
        median_ess_trace=result.median_ess_trace,
        minimum_ess_trace=result.minimum_ess_trace,
        maximum_invalid_fraction_trace=result.maximum_invalid_fraction_trace,
        unique_initial_ancestors_trace=result.unique_initial_ancestors_trace,
        accepted_step_size_trace=result.accepted_step_size_trace,
        backtrack_count_trace=result.backtrack_count_trace,
        elapsed_trace=result.elapsed_trace,
        elapsed_seconds=np.asarray(result.elapsed_seconds),
        completed_updates=np.asarray(result.completed_updates),
        termination_reason=np.asarray(result.termination_reason),
        fit_seed=np.asarray(_fit_seed(args.seed, nstep, start_index)),
    )
    print(
        f"fit r={nstep} start={start_index}: "
        f"updates={result.completed_updates}/{args.iterations}, "
        f"status={result.termination_reason}, "
        f"pseudo={result.marginal_loglik_trace[0]:.2f} -> "
        f"{result.marginal_loglik_trace[-1]:.2f}, "
        f"time={result.elapsed_seconds:.2f}s",
        flush=True,
    )


def _assigned_starts(args: argparse.Namespace) -> range:
    return range(args.worker_index, args.starts, args.workers)


def run_fits(args: argparse.Namespace, starts: list[np.ndarray]) -> None:
    for nstep in args.nsteps:
        # Compile the full-data kernels before timing individual searches.  As
        # with the manuscript's unmonitored timing, one-time JIT compilation is
        # not charged to the optimization trace.
        data = load_dacca_data(nstep)
        fit_pseudo_score(
            starts[0],
            data,
            particles=args.particles,
            iterations=0,
            learning_rate=args.learning_rate,
            burnin=args.burnin,
            gain_exponent=args.gain_exponent,
            seed=_fit_seed(args.seed, nstep, 10_000),
            order=args.order,
            relative_floor=args.relative_floor,
            gradient_clip=args.gradient_clip,
            bridge_particles=0,
        )
        for start_index in _assigned_starts(args):
            _fit_one(args, start_index, starts[start_index], nstep)
        write_fit_tables(args)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_fit_tables(args: argparse.Namespace) -> None:
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for nstep in args.nsteps:
        for start_index in range(args.starts):
            path = _checkpoint_path(args.output, nstep, start_index)
            if not path.exists():
                continue
            fit = _load_checkpoint(path)
            termination = str(fit["termination_reason"])
            parameter_trace = fit["parameter_trace"]
            summary_rows.append(
                {
                    "method": "DS regularized pseudo-score",
                    "inference_nstep": nstep,
                    "start": start_index,
                    "start_type": "global-box",
                    "particles": args.particles,
                    "iterations_requested": args.iterations,
                    "updates_completed": int(fit["completed_updates"]),
                    "termination_reason": termination,
                    "elapsed_seconds": float(fit["elapsed_seconds"]),
                    "initial_pseudo_loglik": float(fit["pseudo_loglik_trace"][0]),
                    "final_pseudo_loglik": float(fit["pseudo_loglik_trace"][-1]),
                    "minimum_ess": float(np.min(fit["minimum_ess_trace"])),
                    "maximum_invalid_fraction": float(
                        np.max(fit["maximum_invalid_fraction_trace"])
                    ),
                    "total_backtracks": int(np.sum(fit["backtrack_count_trace"])),
                    **_parameter_columns(fit["unconstrained"]),
                }
            )
            for iteration in range(parameter_trace.shape[0]):
                trace_rows.append(
                    {
                        "method": "DS regularized pseudo-score",
                        "inference_nstep": nstep,
                        "start": start_index,
                        "iteration": iteration,
                        "elapsed_seconds": float(fit["elapsed_trace"][iteration]),
                        "pseudo_loglik": float(
                            fit["pseudo_loglik_trace"][iteration]
                        ),
                        "complete_pseudologlik": float(
                            fit["complete_pseudologlik_trace"][iteration]
                        ),
                        "score_norm": float(fit["score_norm_trace"][iteration]),
                        "minimum_ess": float(fit["minimum_ess_trace"][iteration]),
                        "maximum_invalid_fraction": float(
                            fit["maximum_invalid_fraction_trace"][iteration]
                        ),
                        "unique_initial_ancestors": int(
                            fit["unique_initial_ancestors_trace"][iteration]
                        ),
                        "accepted_step_size": float(
                            fit["accepted_step_size_trace"][iteration]
                        ),
                        "backtrack_count": int(
                            fit["backtrack_count_trace"][iteration]
                        ),
                        **_parameter_columns(parameter_trace[iteration]),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(args.output / "fit_summary.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(
        args.output / "optimization_training_traces.csv", index=False
    )


def _euler_evaluate(
    unconstrained: np.ndarray,
    *,
    particles: int,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    import pypomp as pp
    from pypomp.core.parameters import PompParameters

    model = pp.models.dacca(nstep=20, dt=None)
    model.pfilter(
        theta=PompParameters(physical_parameter_dict(unconstrained)),
        J=particles,
        reps=replicates,
        key=jax.random.key(seed),
    )
    row = model.results().iloc[0]
    return float(row["logLik"]), float(row["se"])


def evaluate_finals(args: argparse.Namespace) -> None:
    for nstep in args.nsteps:
        for start_index in _assigned_starts(args):
            checkpoint = _checkpoint_path(args.output, nstep, start_index)
            if not checkpoint.exists():
                continue
            destination = _evaluation_path(args.output, nstep, start_index)
            if destination.exists():
                continue
            fit = _load_checkpoint(checkpoint)
            seed = _eval_seed(args.seed, nstep, start_index, args.iterations + 1)
            try:
                loglik, standard_error = _euler_evaluate(
                    fit["unconstrained"],
                    particles=args.eval_particles,
                    replicates=args.eval_replicates,
                    seed=seed,
                )
                status = "ok" if np.isfinite(loglik) else "non-finite"
            except (FloatingPointError, RuntimeError, ValueError) as error:
                loglik = standard_error = float("nan")
                status = f"failed:{type(error).__name__}"
            row = {
                "method": "DS regularized pseudo-score",
                "inference_nstep": nstep,
                "evaluation_nstep": 20,
                "start": start_index,
                "updates_completed": int(fit["completed_updates"]),
                "termination_reason": str(fit["termination_reason"]),
                "euler20_loglik": loglik,
                "euler20_se": standard_error,
                "eval_particles": args.eval_particles,
                "eval_replicates": args.eval_replicates,
                "evaluation_seed": seed,
                "evaluation_status": status,
            }
            _atomic_json(destination, row)
            print(
                f"final eval r={nstep} start={start_index}: "
                f"Euler-20={loglik:.2f} (SE {standard_error:.2f}; {status})",
                flush=True,
            )
    rows = [json.loads(path.read_text()) for path in sorted((args.output / "evaluations").glob("euler_*.json"))]
    pd.DataFrame(rows).to_csv(args.output / "final_evaluations.csv", index=False)


def _checkpoint_iterations(elapsed: np.ndarray, every_seconds: float) -> list[int]:
    """Choose roughly equally spaced wall-clock checkpoints, including both ends."""

    elapsed = np.asarray(elapsed, dtype=float)
    if elapsed.size == 0:
        return []
    targets = np.arange(0.0, elapsed[-1] + every_seconds, every_seconds)
    values = np.searchsorted(elapsed, targets, side="left")
    values = np.clip(values, 0, elapsed.size - 1)
    return sorted({0, *values.tolist(), elapsed.size - 1})


def evaluate_traces(args: argparse.Namespace) -> None:
    for nstep in args.nsteps:
        for start_index in _assigned_starts(args):
            checkpoint = _checkpoint_path(args.output, nstep, start_index)
            if not checkpoint.exists():
                continue
            fit = _load_checkpoint(checkpoint)
            for iteration in _checkpoint_iterations(
                fit["elapsed_trace"], args.trace_every_seconds
            ):
                destination = _trace_evaluation_path(
                    args.output, nstep, start_index, iteration
                )
                if destination.exists():
                    continue
                seed = _eval_seed(args.seed, nstep, start_index, iteration)
                try:
                    loglik, standard_error = _euler_evaluate(
                        fit["parameter_trace"][iteration],
                        particles=args.trace_eval_particles,
                        replicates=args.trace_eval_replicates,
                        seed=seed,
                    )
                    status = "ok" if np.isfinite(loglik) else "non-finite"
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    loglik = standard_error = float("nan")
                    status = f"failed:{type(error).__name__}"
                row = {
                    "method": "DS regularized pseudo-score",
                    "inference_nstep": nstep,
                    "evaluation_nstep": 20,
                    "start": start_index,
                    "iteration": iteration,
                    "elapsed_seconds": float(fit["elapsed_trace"][iteration]),
                    "pseudo_loglik": float(fit["pseudo_loglik_trace"][iteration]),
                    "euler20_loglik": loglik,
                    "euler20_se": standard_error,
                    "eval_particles": args.trace_eval_particles,
                    "eval_replicates": args.trace_eval_replicates,
                    "evaluation_seed": seed,
                    "evaluation_status": status,
                }
                _atomic_json(destination, row)
                print(
                    f"trace eval r={nstep} start={start_index} "
                    f"iteration={iteration}: Euler-20={loglik:.2f}",
                    flush=True,
                )
    rows = [json.loads(path.read_text()) for path in sorted((args.output / "trace_evaluations").glob("euler_*.json"))]
    pd.DataFrame(rows).to_csv(
        args.output / "optimization_euler_traces.csv", index=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("fit", "final-eval", "trace-eval"),
        default=("fit", "final-eval", "trace-eval"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/smc_benchmark"))
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--nsteps", nargs="+", type=int, default=(5, 10, 20))
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--burnin", type=int, default=30)
    parser.add_argument("--gain-exponent", type=float, default=0.9)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--relative-floor", type=float, default=1e-7)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--maximum-backtracks", type=int, default=6)
    parser.add_argument("--maximum-consecutive-rejections", type=int, default=10)
    parser.add_argument("--maximum-path-retries", type=int, default=3)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument(
        "--maximum-acceptable-invalid-fraction", type=float, default=0.5
    )
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=1000.0)
    parser.add_argument("--eval-particles", type=int, default=5000)
    parser.add_argument("--eval-replicates", type=int, default=36)
    parser.add_argument("--trace-every-seconds", type=float, default=100.0)
    parser.add_argument("--trace-eval-particles", type=int, default=1000)
    parser.add_argument("--trace-eval-replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.starts < 1:
        raise ValueError("starts must be positive")
    if any(nstep < 1 for nstep in args.nsteps):
        raise ValueError("nsteps must be positive")
    if args.trace_every_seconds <= 0.0:
        raise ValueError("trace-every-seconds must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("worker-index must be in [0, workers)")
    _prepare_output(args)
    starts = make_starts(args.starts, args.seed)
    if "fit" in args.stages:
        run_fits(args, starts)
    else:
        write_fit_tables(args)
    if "final-eval" in args.stages:
        evaluate_finals(args)
    if "trace-eval" in args.stages:
        evaluate_traces(args)


if __name__ == "__main__":
    main()
