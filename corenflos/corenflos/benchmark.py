"""Resumable Dacca benchmark for differentiable OT particle filtering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from ditlevsen.benchmark import _parameter_columns, make_starts
from ditlevsen.data import load_dacca_data
from ditlevsen.model import physical_parameter_dict
from ditlevsen.warm_starts import load_starts_file

from .dpf import DPFConfig, DaccaArrays
from .fit import fit_dpf, make_value_and_gradient
from .transport import TransportConfig


FIT_KEYS = (
    "seed",
    "starts",
    "starts_file",
    "elapsed_time_offset_seconds",
    "nstep",
    "particles",
    "ess_threshold",
    "epsilon",
    "sinkhorn_scaling",
    "sinkhorn_threshold",
    "sinkhorn_iterations",
    "optimizer",
    "learning_rate",
    "learning_rate_decay",
    "gradient_clip",
    "maximum_acceptable_invalid_fraction",
    "maximum_pseudo_loglik_drop",
    "rollback_patience",
    "maximum_restarts",
    "restart_factor",
    "maximum_updates",
    "maximum_elapsed_seconds",
    "output_selection_seconds",
    "output_selection",
    "change_seed",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _checkpoint_path(output: Path, start: int) -> Path:
    return output / "checkpoints" / f"fit_start{start:03d}.npz"


def _evaluation_path(output: Path, start: int) -> Path:
    return output / "evaluations" / f"euler_start{start:03d}.json"


def _trace_evaluation_path(output: Path, start: int, iteration: int) -> Path:
    return (
        output
        / "trace_evaluations"
        / f"euler_start{start:03d}_iter{iteration:04d}.json"
    )


def _fit_seed(seed: int, start: int) -> int:
    return seed + 20_000_003 + 10_007 * start


def _eval_seed(seed: int, start: int, iteration: int) -> int:
    return seed + 70_000_001 + 997 * start + iteration


def _output_iteration(elapsed: np.ndarray, seconds: float | None) -> int:
    if seconds is None:
        return len(elapsed) - 1
    return min(int(np.searchsorted(elapsed, seconds, side="left")), len(elapsed) - 1)


def _prepare_output(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ("checkpoints", "evaluations", "trace_evaluations"):
        (args.output / directory).mkdir(exist_ok=True)
    path = args.output / "configuration.json"
    configuration = {key: _json_ready(value) for key, value in vars(args).items()}
    if path.exists():
        previous = json.loads(path.read_text())
        changed = [
            key
            for key in FIT_KEYS
            if previous.get(key, configuration.get(key)) != configuration.get(key)
        ]
        if changed:
            raise ValueError("different fit configuration: " + ", ".join(changed))
    else:
        _atomic_json(path, configuration)


def _assigned_starts(args: argparse.Namespace) -> range:
    return range(args.worker_index, args.starts, args.workers)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _config(args: argparse.Namespace) -> DPFConfig:
    return DPFConfig(
        particles=args.particles,
        ess_threshold=args.ess_threshold,
        transport=TransportConfig(
            epsilon=args.epsilon,
            scaling=args.sinkhorn_scaling,
            threshold=args.sinkhorn_threshold,
            max_iterations=args.sinkhorn_iterations,
        ),
    )


def run_fits(args: argparse.Namespace, starts: list[np.ndarray]) -> None:
    data = DaccaArrays.from_data(load_dacca_data(args.nstep))
    evaluator = make_value_and_gradient(data, _config(args))
    # Compile once, outside all reported run times.
    warm_result, warm_gradient = evaluator(
        jax.numpy.asarray(starts[0]), jax.random.key(_fit_seed(args.seed, 100_000))
    )
    jax.block_until_ready(warm_gradient)
    print(
        f"compiled J={args.particles}; warm pseudo={float(warm_result.loglik):.2f}",
        flush=True,
    )
    for start_index in _assigned_starts(args):
        destination = _checkpoint_path(args.output, start_index)
        if destination.exists():
            print(f"fit exists: start={start_index}", flush=True)
            continue
        result = fit_dpf(
            starts[start_index],
            evaluator,
            seed=_fit_seed(args.seed, start_index),
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            gradient_clip=args.gradient_clip,
            maximum_acceptable_invalid_fraction=(
                args.maximum_acceptable_invalid_fraction
            ),
            maximum_pseudo_loglik_drop=args.maximum_pseudo_loglik_drop,
            rollback_patience=args.rollback_patience,
            maximum_restarts=args.maximum_restarts,
            restart_factor=args.restart_factor,
            maximum_updates=args.maximum_updates,
            maximum_elapsed_seconds=args.maximum_elapsed_seconds,
            change_seed=args.change_seed,
        )
        elapsed_trace = result.elapsed_trace + args.elapsed_time_offset_seconds
        acceptable = (
            np.isfinite(result.pseudo_loglik_trace)
            & np.isfinite(result.gradient_norm_trace)
            & (result.maximum_invalid_fraction_trace < 1.0)
            & (
                result.maximum_invalid_fraction_trace
                <= args.maximum_acceptable_invalid_fraction
            )
        )
        before_cutoff = (
            np.ones_like(acceptable, dtype=bool)
            if args.output_selection_seconds is None
            else elapsed_trace <= args.output_selection_seconds
        )
        available = np.flatnonzero(acceptable & before_cutoff)
        if available.size == 0:
            available = np.flatnonzero(acceptable)
        if available.size == 0:
            selected = 0
        elif args.output_selection == "maximum-pseudo":
            selected = int(available[np.argmax(result.pseudo_loglik_trace[available])])
        else:
            requested = _output_iteration(elapsed_trace, args.output_selection_seconds)
            at_or_before = available[available <= requested]
            selected = int(at_or_before[-1] if at_or_before.size else available[0])
        _atomic_npz(
            destination,
            start=starts[start_index],
            unconstrained=result.parameter_trace[selected],
            terminal_unconstrained=result.unconstrained,
            output_selected_iteration=np.asarray(selected),
            output_selected_elapsed_seconds=np.asarray(elapsed_trace[selected]),
            parameter_trace=result.parameter_trace,
            pseudo_loglik_trace=result.pseudo_loglik_trace,
            gradient_norm_trace=result.gradient_norm_trace,
            minimum_ess_trace=result.minimum_ess_trace,
            median_ess_trace=result.median_ess_trace,
            resampling_count_trace=result.resampling_count_trace,
            maximum_invalid_fraction_trace=result.maximum_invalid_fraction_trace,
            learning_rate_trace=result.learning_rate_trace,
            incumbent_pseudo_loglik_trace=result.incumbent_pseudo_loglik_trace,
            restart_count_trace=result.restart_count_trace,
            elapsed_trace=elapsed_trace,
            optimizer_elapsed_trace=result.elapsed_trace,
            elapsed_seconds=np.asarray(
                result.elapsed_seconds + args.elapsed_time_offset_seconds
            ),
            optimizer_elapsed_seconds=np.asarray(result.elapsed_seconds),
            elapsed_time_offset_seconds=np.asarray(args.elapsed_time_offset_seconds),
            completed_updates=np.asarray(result.completed_updates),
            termination_reason=np.asarray(result.termination_reason),
            fit_seed=np.asarray(_fit_seed(args.seed, start_index)),
        )
        print(
            f"fit start={start_index}: updates={result.completed_updates}, "
            f"status={result.termination_reason}, "
            f"pseudo={result.pseudo_loglik_trace[0]:.2f} -> "
            f"{result.pseudo_loglik_trace[-1]:.2f}, "
            f"optimizer_time={result.elapsed_seconds:.1f}s, "
            f"total_time={result.elapsed_seconds + args.elapsed_time_offset_seconds:.1f}s, "
            f"selected={elapsed_trace[selected]:.1f}s",
            flush=True,
        )
        write_fit_tables(args)


def write_fit_tables(args: argparse.Namespace) -> None:
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for start_index in range(args.starts):
        path = _checkpoint_path(args.output, start_index)
        if not path.exists():
            continue
        fit = _load_checkpoint(path)
        selected = int(fit["output_selected_iteration"])
        parameter_trace = fit["parameter_trace"]
        summary_rows.append(
            {
                "method": "Corenflos DPF",
                "inference_nstep": args.nstep,
                "start": start_index,
                "start_type": (
                    "if2-warm-start" if args.starts_file is not None else "global-box"
                ),
                "starts_file": (
                    str(args.starts_file) if args.starts_file is not None else ""
                ),
                "particles": args.particles,
                "updates_completed": int(fit["completed_updates"]),
                "termination_reason": str(fit["termination_reason"]),
                "elapsed_seconds": float(fit["elapsed_seconds"]),
                "optimizer_elapsed_seconds": float(
                    fit.get("optimizer_elapsed_seconds", fit["elapsed_seconds"])
                ),
                "elapsed_time_offset_seconds": float(
                    fit.get("elapsed_time_offset_seconds", 0.0)
                ),
                "output_selected_iteration": selected,
                "output_selected_elapsed_seconds": float(
                    fit["output_selected_elapsed_seconds"]
                ),
                "initial_pseudo_loglik": float(fit["pseudo_loglik_trace"][0]),
                "selected_pseudo_loglik": float(fit["pseudo_loglik_trace"][selected]),
                "terminal_pseudo_loglik": float(fit["pseudo_loglik_trace"][-1]),
                "minimum_ess": float(np.min(fit["minimum_ess_trace"])),
                "maximum_invalid_fraction": float(
                    np.max(fit["maximum_invalid_fraction_trace"])
                ),
                "restarts": int(np.max(fit.get("restart_count_trace", [0]))),
                **_parameter_columns(fit["unconstrained"]),
            }
        )
        for iteration, parameters in enumerate(parameter_trace):
            trace_rows.append(
                {
                    "method": "Corenflos DPF",
                    "inference_nstep": args.nstep,
                    "start": start_index,
                    "iteration": iteration,
                    "elapsed_seconds": float(fit["elapsed_trace"][iteration]),
                    "pseudo_loglik": float(fit["pseudo_loglik_trace"][iteration]),
                    "gradient_norm": float(fit["gradient_norm_trace"][iteration]),
                    "minimum_ess": float(fit["minimum_ess_trace"][iteration]),
                    "median_ess": float(fit["median_ess_trace"][iteration]),
                    "resampling_count": int(fit["resampling_count_trace"][iteration]),
                    "maximum_invalid_fraction": float(
                        fit["maximum_invalid_fraction_trace"][iteration]
                    ),
                    "learning_rate": float(fit["learning_rate_trace"][iteration]),
                    "incumbent_pseudo_loglik": float(
                        fit.get(
                            "incumbent_pseudo_loglik_trace",
                            fit["pseudo_loglik_trace"],
                        )[iteration]
                    ),
                    "restart_count": int(
                        fit.get(
                            "restart_count_trace",
                            np.zeros(len(parameter_trace), dtype=int),
                        )[iteration]
                    ),
                    **_parameter_columns(parameters),
                }
            )
    _atomic_csv(args.output / "fit_summary.csv", pd.DataFrame(summary_rows))
    _atomic_csv(
        args.output / "optimization_training_traces.csv", pd.DataFrame(trace_rows)
    )


def _euler_evaluate(
    unconstrained: np.ndarray, *, particles: int, replicates: int, seed: int
) -> tuple[float, float]:
    import pypomp as pp

    model = pp.models.dacca(nstep=20, dt=None)
    model.pfilter(
        theta=pp.PompParameters(physical_parameter_dict(unconstrained)),
        J=particles,
        reps=replicates,
        key=jax.random.key(seed),
    )
    row = model.results().iloc[0]
    return float(row["logLik"]), float(row["se"])


def evaluate_finals(args: argparse.Namespace) -> None:
    for start_index in _assigned_starts(args):
        checkpoint = _checkpoint_path(args.output, start_index)
        destination = _evaluation_path(args.output, start_index)
        if not checkpoint.exists() or destination.exists():
            continue
        fit = _load_checkpoint(checkpoint)
        seed = _eval_seed(args.seed, start_index, args.maximum_updates + 1)
        try:
            loglik, se = _euler_evaluate(
                fit["unconstrained"],
                particles=args.eval_particles,
                replicates=args.eval_replicates,
                seed=seed,
            )
            status = "ok" if np.isfinite(loglik) else "non-finite"
        except (FloatingPointError, RuntimeError, ValueError) as error:
            loglik = se = float("nan")
            status = f"failed:{type(error).__name__}"
        _atomic_json(
            destination,
            {
                "method": "Corenflos DPF",
                "inference_nstep": args.nstep,
                "evaluation_nstep": 20,
                "start": start_index,
                "euler20_loglik": loglik,
                "euler20_se": se,
                "eval_particles": args.eval_particles,
                "eval_replicates": args.eval_replicates,
                "evaluation_seed": seed,
                "evaluation_status": status,
            },
        )
        print(f"final eval start={start_index}: {loglik:.2f} ({status})", flush=True)
    rows = [
        json.loads(path.read_text())
        for path in sorted((args.output / "evaluations").glob("euler_*.json"))
    ]
    _atomic_csv(args.output / "final_evaluations.csv", pd.DataFrame(rows))


def evaluate_traces(args: argparse.Namespace) -> None:
    """Evaluate every stored optimizer iterate with Euler-20, without thinning."""

    for start_index in _assigned_starts(args):
        checkpoint = _checkpoint_path(args.output, start_index)
        if not checkpoint.exists():
            continue
        fit = _load_checkpoint(checkpoint)
        for iteration, parameters in enumerate(fit["parameter_trace"]):
            destination = _trace_evaluation_path(args.output, start_index, iteration)
            if destination.exists():
                previous = json.loads(destination.read_text())
                if (
                    previous.get("eval_particles") == args.trace_eval_particles
                    and previous.get("eval_replicates") == args.trace_eval_replicates
                ):
                    continue
            seed = _eval_seed(args.seed, start_index, iteration)
            try:
                loglik, se = _euler_evaluate(
                    parameters,
                    particles=args.trace_eval_particles,
                    replicates=args.trace_eval_replicates,
                    seed=seed,
                )
                status = "ok" if np.isfinite(loglik) else "non-finite"
            except (FloatingPointError, RuntimeError, ValueError) as error:
                loglik = se = float("nan")
                status = f"failed:{type(error).__name__}"
            _atomic_json(
                destination,
                {
                    "method": "Corenflos DPF",
                    "inference_nstep": args.nstep,
                    "evaluation_nstep": 20,
                    "start": start_index,
                    "iteration": iteration,
                    "elapsed_seconds": float(fit["elapsed_trace"][iteration]),
                    "pseudo_loglik": float(fit["pseudo_loglik_trace"][iteration]),
                    "euler20_loglik": loglik,
                    "euler20_se": se,
                    "eval_particles": args.trace_eval_particles,
                    "eval_replicates": args.trace_eval_replicates,
                    "evaluation_seed": seed,
                    "evaluation_status": status,
                },
            )
            print(
                f"trace eval start={start_index} iteration={iteration}: {loglik:.2f}",
                flush=True,
            )
    rows = [
        json.loads(path.read_text())
        for path in sorted((args.output / "trace_evaluations").glob("euler_*.json"))
    ]
    _atomic_csv(args.output / "optimization_euler_traces.csv", pd.DataFrame(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("fit", "final-eval", "trace-eval"),
        default=("fit",),
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--starts-file", type=Path)
    parser.add_argument("--elapsed-time-offset-seconds", type=float, default=0.0)
    parser.add_argument("--nstep", type=int, default=20)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--sinkhorn-scaling", type=float, default=0.75)
    parser.add_argument("--sinkhorn-threshold", type=float, default=1.0e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--learning-rate-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument(
        "--maximum-acceptable-invalid-fraction", type=float, default=1.0
    )
    parser.add_argument("--maximum-pseudo-loglik-drop", type=float, default=30.0)
    parser.add_argument("--rollback-patience", type=int, default=10)
    parser.add_argument("--maximum-restarts", type=int, default=6)
    parser.add_argument("--restart-factor", type=float, default=0.5)
    parser.add_argument("--maximum-updates", type=int, default=5000)
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=800.0)
    parser.add_argument("--output-selection-seconds", type=float, default=700.0)
    parser.add_argument(
        "--output-selection",
        choices=("maximum-pseudo", "fixed-time"),
        default="maximum-pseudo",
    )
    parser.add_argument(
        "--change-seed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--eval-particles", type=int, default=5000)
    parser.add_argument("--eval-replicates", type=int, default=36)
    parser.add_argument("--trace-eval-particles", type=int, default=5000)
    parser.add_argument("--trace-eval-replicates", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("worker-index must be in [0, workers)")
    if not 0.0 <= args.ess_threshold <= 1.0:
        raise ValueError("ess-threshold must be in [0, 1]")
    if args.elapsed_time_offset_seconds < 0.0:
        raise ValueError("elapsed-time-offset-seconds must be nonnegative")
    _prepare_output(args)
    starts = (
        load_starts_file(args.starts_file, args.starts)
        if args.starts_file is not None
        else make_starts(args.starts, args.seed)
    )
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
