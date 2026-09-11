"""Resumable benchmark for the Dacca Ditlevsen-style pseudo-score method.

Each internal step uses the Ditlevsen--Samson order-1.5 moments.  Their
linearized Gaussian recursions are composed into a monthly endpoint
transition, and SMC is run only at observation times.  The scientific target
remains the manuscript's Euler-20 POMP likelihood, evaluated independently
with Pypomp, so the block-Gaussian surrogate is never reported as the original
Dacca likelihood.
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
from .block_smc import block_bootstrap_filter, fit_block_pseudo_score
from .data import load_dacca_data
from .model import physical_parameter_dict
from .smc import fit_pseudo_score


FIT_SIGNATURE_KEYS = (
    "seed",
    "starts",
    "nsteps",
    "transition",
    "proposal",
    "particles",
    "iterations",
    "learning_rate",
    "learning_rate_decay_start",
    "learning_rate_decay_exponent",
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
    "likelihood_guard_particles",
    "likelihood_guard_interval",
    "maximum_guard_loglik_drop",
    "maximum_elapsed_seconds",
    "output_selection_seconds",
)
EVALUATION_STAGES = {
    "block-eval",
    "final-eval",
    "trace-eval",
    "selected-eval",
}


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


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _json_ready(value) for key, value in vars(args).items()}


def _prepare_output(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    (args.output / "evaluations").mkdir(exist_ok=True)
    (args.output / "trace_evaluations").mkdir(exist_ok=True)
    (args.output / "block_trace_evaluations").mkdir(exist_ok=True)
    (args.output / "selected_evaluations").mkdir(exist_ok=True)
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
    if args.worker_index == 0 and EVALUATION_STAGES.intersection(args.stages):
        _atomic_json(
            args.output / "evaluation_configuration.json",
            {
                "stages": list(args.stages),
                "evaluation_model": "Dacca Euler-20 POMP",
                "final_eval_particles": args.eval_particles,
                "final_eval_replicates": args.eval_replicates,
                "trace_eval_particles": args.trace_eval_particles,
                "trace_eval_replicates": args.trace_eval_replicates,
                "trace_every_seconds": args.trace_every_seconds,
                "trace_every_update": args.trace_every_update,
                "block_eval_particles": args.block_eval_particles,
                "block_eval_replicates": args.block_eval_replicates,
            },
        )


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


def _selected_evaluation_path(output: Path, nstep: int, start: int) -> Path:
    return (
        output
        / "selected_evaluations"
        / f"best_block_r{nstep:02d}_start{start:03d}.json"
    )


def _block_trace_evaluation_path(
    output: Path, nstep: int, start: int, target_seconds: int
) -> Path:
    return (
        output
        / "block_trace_evaluations"
        / f"block_r{nstep:02d}_start{start:03d}_sec{target_seconds:04d}.json"
    )


def _fit_seed(seed: int, nstep: int, start: int) -> int:
    return seed + 1_000_003 * nstep + 10_007 * start


def _eval_seed(seed: int, nstep: int, start: int, iteration: int) -> int:
    return seed + 30_000_001 + 100_003 * nstep + 997 * start + iteration


def _block_eval_seed(
    seed: int, nstep: int, start: int, target_seconds: int, replicate: int
) -> int:
    return (
        seed
        + 60_000_001
        + 100_003 * nstep
        + 997 * start
        + 37 * target_seconds
        + replicate
    )


def _fit_one(
    args: argparse.Namespace, start_index: int, start: np.ndarray, nstep: int
) -> None:
    checkpoint = _checkpoint_path(args.output, nstep, start_index)
    if checkpoint.exists():
        print(f"fit exists: r={nstep} start={start_index}", flush=True)
        return
    data = load_dacca_data(nstep)
    common = dict(
        particles=args.particles,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        burnin=args.burnin,
        gain_exponent=args.gain_exponent,
        seed=_fit_seed(args.seed, nstep, start_index),
        gradient_clip=args.gradient_clip,
        maximum_backtracks=args.maximum_backtracks,
        maximum_consecutive_rejections=args.maximum_consecutive_rejections,
        maximum_path_retries=args.maximum_path_retries,
        backtrack_factor=args.backtrack_factor,
        maximum_acceptable_invalid_fraction=(args.maximum_acceptable_invalid_fraction),
        likelihood_guard_particles=args.likelihood_guard_particles,
        likelihood_guard_interval=args.likelihood_guard_interval,
        maximum_guard_loglik_drop=args.maximum_guard_loglik_drop,
        maximum_elapsed_seconds=args.maximum_elapsed_seconds,
    )
    if args.transition == "block":
        result = fit_block_pseudo_score(
            start,
            data,
            endpoint_relative_floor=args.relative_floor,
            proposal=args.proposal,
            learning_rate_decay_start=args.learning_rate_decay_start,
            learning_rate_decay_exponent=args.learning_rate_decay_exponent,
            **common,
        )
    else:
        result = fit_pseudo_score(
            start,
            data,
            order=args.order,
            relative_floor=args.relative_floor,
            bridge_particles=0,
            **common,
        )
    selected_iteration = _output_iteration(
        result.elapsed_trace, args.output_selection_seconds
    )
    selected_parameters = result.parameter_trace[selected_iteration]
    _atomic_npz(
        checkpoint,
        start=np.asarray(start),
        unconstrained=selected_parameters,
        terminal_unconstrained=result.unconstrained,
        output_selected_iteration=np.asarray(selected_iteration),
        output_selected_elapsed_seconds=np.asarray(
            result.elapsed_trace[selected_iteration]
        ),
        parameter_trace=result.parameter_trace,
        pseudo_loglik_trace=result.marginal_loglik_trace,
        complete_pseudologlik_trace=result.complete_loglik_trace,
        score_norm_trace=result.score_norm_trace,
        median_ess_trace=result.median_ess_trace,
        minimum_ess_trace=result.minimum_ess_trace,
        maximum_invalid_fraction_trace=result.maximum_invalid_fraction_trace,
        unique_initial_ancestors_trace=result.unique_initial_ancestors_trace,
        backward_fallback_trace=result.backward_fallback_trace,
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
        f"time={result.elapsed_seconds:.2f}s, "
        f"selected={result.elapsed_trace[selected_iteration]:.2f}s",
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
        if args.transition == "block":
            fit_block_pseudo_score(
                starts[0],
                data,
                particles=args.particles,
                iterations=1,
                learning_rate=args.learning_rate,
                learning_rate_decay_start=args.learning_rate_decay_start,
                learning_rate_decay_exponent=(args.learning_rate_decay_exponent),
                burnin=args.burnin,
                gain_exponent=args.gain_exponent,
                seed=_fit_seed(args.seed, nstep, 10_000),
                endpoint_relative_floor=args.relative_floor,
                proposal=args.proposal,
                gradient_clip=args.gradient_clip,
                maximum_backtracks=0,
                maximum_consecutive_rejections=1,
                maximum_path_retries=args.maximum_path_retries,
                maximum_acceptable_invalid_fraction=(
                    args.maximum_acceptable_invalid_fraction
                ),
                likelihood_guard_particles=args.likelihood_guard_particles,
                likelihood_guard_interval=1,
                maximum_guard_loglik_drop=args.maximum_guard_loglik_drop,
            )
        else:
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
            backward_fallback_trace = fit.get(
                "backward_fallback_trace",
                np.zeros(parameter_trace.shape[0], dtype=int),
            )
            summary_rows.append(
                {
                    "method": (
                        "DS order-1.5 block Gaussian"
                        if args.transition == "block"
                        else "DS regularized pseudo-score"
                    ),
                    "inference_nstep": nstep,
                    "start": start_index,
                    "start_type": "global-box",
                    "particles": args.particles,
                    "iterations_requested": args.iterations,
                    "updates_completed": int(fit["completed_updates"]),
                    "termination_reason": termination,
                    "elapsed_seconds": float(fit["elapsed_seconds"]),
                    "output_selected_iteration": int(
                        fit.get(
                            "output_selected_iteration",
                            np.asarray(parameter_trace.shape[0] - 1),
                        )
                    ),
                    "output_selected_elapsed_seconds": float(
                        fit.get(
                            "output_selected_elapsed_seconds",
                            fit["elapsed_seconds"],
                        )
                    ),
                    "initial_pseudo_loglik": float(fit["pseudo_loglik_trace"][0]),
                    "final_pseudo_loglik": float(fit["pseudo_loglik_trace"][-1]),
                    "minimum_ess": float(np.min(fit["minimum_ess_trace"])),
                    "maximum_invalid_fraction": float(
                        np.max(fit["maximum_invalid_fraction_trace"])
                    ),
                    "maximum_backward_fallbacks": int(np.max(backward_fallback_trace)),
                    "total_backtracks": int(np.sum(fit["backtrack_count_trace"])),
                    **_parameter_columns(fit["unconstrained"]),
                }
            )
            for iteration in range(parameter_trace.shape[0]):
                trace_rows.append(
                    {
                        "method": (
                            "DS order-1.5 block Gaussian"
                            if args.transition == "block"
                            else "DS regularized pseudo-score"
                        ),
                        "inference_nstep": nstep,
                        "start": start_index,
                        "iteration": iteration,
                        "elapsed_seconds": float(fit["elapsed_trace"][iteration]),
                        "pseudo_loglik": float(fit["pseudo_loglik_trace"][iteration]),
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
                        "backward_fallbacks": int(backward_fallback_trace[iteration]),
                        "accepted_step_size": float(
                            fit["accepted_step_size_trace"][iteration]
                        ),
                        "backtrack_count": int(fit["backtrack_count_trace"][iteration]),
                        **_parameter_columns(parameter_trace[iteration]),
                    }
                )
    _atomic_csv(args.output / "fit_summary.csv", pd.DataFrame(summary_rows))
    _atomic_csv(
        args.output / "optimization_training_traces.csv",
        pd.DataFrame(trace_rows),
    )


def _euler_evaluate(
    unconstrained: np.ndarray,
    *,
    particles: int,
    replicates: int,
    seed: int,
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
                "method": (
                    "DS order-1.5 block Gaussian"
                    if args.transition == "block"
                    else "DS regularized pseudo-score"
                ),
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
    rows = [
        json.loads(path.read_text())
        for path in sorted((args.output / "evaluations").glob("euler_*.json"))
    ]
    _atomic_csv(args.output / "final_evaluations.csv", pd.DataFrame(rows))


def _checkpoint_iterations(
    elapsed: np.ndarray, every_seconds: float, every_update: bool = False
) -> list[int]:
    """Choose every update or wall-clock checkpoints, including both ends."""

    elapsed = np.asarray(elapsed, dtype=float)
    if elapsed.size == 0:
        return []
    if every_update:
        return list(range(elapsed.size))
    targets = np.arange(0.0, elapsed[-1] + every_seconds, every_seconds)
    values = np.searchsorted(elapsed, targets, side="left")
    values = np.clip(values, 0, elapsed.size - 1)
    return sorted({0, *values.tolist(), elapsed.size - 1})


def _output_iteration(elapsed: np.ndarray, selection_seconds: float | None) -> int:
    """Select the first completed evaluation crossing a fixed time threshold."""

    elapsed = np.asarray(elapsed, dtype=float)
    if elapsed.size == 0:
        raise ValueError("elapsed trace must be nonempty")
    if selection_seconds is None:
        return elapsed.size - 1
    return min(
        int(np.searchsorted(elapsed, selection_seconds, side="left")),
        elapsed.size - 1,
    )


def _checkpoint_targets(
    elapsed: np.ndarray, every_seconds: float
) -> list[tuple[int, int]]:
    """Map fixed wall-clock targets to their nearest stored iterates."""

    elapsed = np.asarray(elapsed, dtype=float)
    if elapsed.size == 0:
        return []
    targets = np.arange(0.0, elapsed[-1] + every_seconds, every_seconds)
    targets = targets[targets <= elapsed[-1] + every_seconds / 2.0]
    pairs: list[tuple[int, int]] = []
    for target in targets:
        iteration = int(np.argmin(np.abs(elapsed - target)))
        pairs.append((int(round(target)), iteration))
    final_pair = (int(round(elapsed[-1])), elapsed.size - 1)
    if not pairs or pairs[-1][1] != final_pair[1]:
        pairs.append(final_pair)
    return pairs


def _combine_loglik_replicates(values: np.ndarray) -> tuple[float, float]:
    """Log of the average likelihood and its delta-method standard error."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty vector")
    if not np.all(np.isfinite(values)):
        return float("nan"), float("nan")
    maximum = float(np.max(values))
    relative = np.exp(values - maximum)
    mean_relative = float(np.mean(relative))
    combined = maximum + float(np.log(mean_relative))
    if values.size == 1:
        return combined, float("nan")
    standard_error = float(
        np.std(relative, ddof=1) / np.sqrt(values.size) / mean_relative
    )
    return combined, standard_error


def evaluate_block_traces(args: argparse.Namespace) -> None:
    """Evaluate stored iterates under an independent block particle filter."""

    for nstep in args.nsteps:
        data = load_dacca_data(nstep)
        for start_index in _assigned_starts(args):
            checkpoint = _checkpoint_path(args.output, nstep, start_index)
            if not checkpoint.exists():
                continue
            fit = _load_checkpoint(checkpoint)
            for target_seconds, iteration in _checkpoint_targets(
                fit["elapsed_trace"], args.trace_every_seconds
            ):
                destination = _block_trace_evaluation_path(
                    args.output, nstep, start_index, target_seconds
                )
                if destination.exists():
                    continue
                replicate_values = []
                for replicate in range(args.block_eval_replicates):
                    seed = _block_eval_seed(
                        args.seed,
                        nstep,
                        start_index,
                        target_seconds,
                        replicate,
                    )
                    result = block_bootstrap_filter(
                        fit["parameter_trace"][iteration],
                        data,
                        particles=args.block_eval_particles,
                        seed=seed,
                        endpoint_relative_floor=args.relative_floor,
                        proposal=args.proposal,
                    )
                    replicate_values.append(result.loglik)
                loglik, standard_error = _combine_loglik_replicates(
                    np.asarray(replicate_values)
                )
                row = {
                    "method": "DS order-1.5 block Gaussian",
                    "inference_nstep": nstep,
                    "start": start_index,
                    "target_elapsed_seconds": target_seconds,
                    "iteration": iteration,
                    "elapsed_seconds": float(fit["elapsed_trace"][iteration]),
                    "training_pseudo_loglik": float(
                        fit["pseudo_loglik_trace"][iteration]
                    ),
                    "block_loglik": loglik,
                    "block_loglik_se": standard_error,
                    "eval_particles": args.block_eval_particles,
                    "eval_replicates": args.block_eval_replicates,
                    "replicate_logliks": replicate_values,
                }
                _atomic_json(destination, row)
                print(
                    f"block eval r={nstep} start={start_index} "
                    f"target={target_seconds}s: {loglik:.2f} "
                    f"(SE {standard_error:.2f})",
                    flush=True,
                )
    rows = [
        json.loads(path.read_text())
        for path in sorted(
            (args.output / "block_trace_evaluations").glob("block_*.json")
        )
    ]
    _atomic_csv(args.output / "optimization_block_traces.csv", pd.DataFrame(rows))


def evaluate_traces(args: argparse.Namespace) -> None:
    for nstep in args.nsteps:
        for start_index in _assigned_starts(args):
            checkpoint = _checkpoint_path(args.output, nstep, start_index)
            if not checkpoint.exists():
                continue
            fit = _load_checkpoint(checkpoint)
            for iteration in _checkpoint_iterations(
                fit["elapsed_trace"],
                args.trace_every_seconds,
                every_update=args.trace_every_update,
            ):
                destination = _trace_evaluation_path(
                    args.output, nstep, start_index, iteration
                )
                if destination.exists():
                    previous = json.loads(destination.read_text())
                    matching_effort = (
                        previous.get("eval_particles") == args.trace_eval_particles
                        and previous.get("eval_replicates")
                        == args.trace_eval_replicates
                    )
                    if matching_effort:
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
                    "method": (
                        "DS order-1.5 block Gaussian"
                        if args.transition == "block"
                        else "DS regularized pseudo-score"
                    ),
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
    rows = [
        json.loads(path.read_text())
        for path in sorted((args.output / "trace_evaluations").glob("euler_*.json"))
    ]
    _atomic_csv(args.output / "optimization_euler_traces.csv", pd.DataFrame(rows))


def evaluate_selected(args: argparse.Namespace) -> None:
    """Evaluate the best independently scored block-likelihood iterate."""

    for nstep in args.nsteps:
        for start_index in _assigned_starts(args):
            checkpoint = _checkpoint_path(args.output, nstep, start_index)
            if not checkpoint.exists():
                continue
            destination = _selected_evaluation_path(args.output, nstep, start_index)
            if destination.exists():
                continue
            fit = _load_checkpoint(checkpoint)
            pseudo = np.asarray(fit["pseudo_loglik_trace"], dtype=float)
            block_paths = sorted(
                (args.output / "block_trace_evaluations").glob(
                    f"block_r{nstep:02d}_start{start_index:03d}_sec*.json"
                )
            )
            block_rows = [json.loads(path.read_text()) for path in block_paths]
            finite_block = [
                row for row in block_rows if np.isfinite(row["block_loglik"])
            ]
            if finite_block:
                selected_block = max(finite_block, key=lambda row: row["block_loglik"])
                selected_iteration = int(selected_block["iteration"])
                selection = "maximum-held-out-block-loglik"
            else:
                finite = np.flatnonzero(np.isfinite(pseudo))
                if finite.size == 0:
                    raise ValueError(f"no finite pseudo likelihoods in {checkpoint}")
                selected_iteration = int(finite[np.argmax(pseudo[finite])])
                selected_block = None
                selection = "maximum-training-block-pseudo-loglik"
            selected_parameters = fit["parameter_trace"][selected_iteration]
            seed = _eval_seed(
                args.seed,
                nstep,
                start_index,
                1_000_000 + selected_iteration,
            )
            try:
                loglik, standard_error = _euler_evaluate(
                    selected_parameters,
                    particles=args.eval_particles,
                    replicates=args.eval_replicates,
                    seed=seed,
                )
                status = "ok" if np.isfinite(loglik) else "non-finite"
            except (FloatingPointError, RuntimeError, ValueError) as error:
                loglik = standard_error = float("nan")
                status = f"failed:{type(error).__name__}"
            row = {
                "method": "DS order-1.5 block Gaussian",
                "selection": selection,
                "inference_nstep": nstep,
                "evaluation_nstep": 20,
                "start": start_index,
                "selected_iteration": selected_iteration,
                "selected_elapsed_seconds": float(
                    fit["elapsed_trace"][selected_iteration]
                ),
                "selected_pseudo_loglik": float(pseudo[selected_iteration]),
                "selected_block_loglik": (
                    float(selected_block["block_loglik"])
                    if selected_block is not None
                    else float("nan")
                ),
                "selected_block_loglik_se": (
                    float(selected_block["block_loglik_se"])
                    if selected_block is not None
                    else float("nan")
                ),
                "selected_block_target_seconds": (
                    int(selected_block["target_elapsed_seconds"])
                    if selected_block is not None
                    else -1
                ),
                "euler20_loglik": loglik,
                "euler20_se": standard_error,
                "eval_particles": args.eval_particles,
                "eval_replicates": args.eval_replicates,
                "evaluation_seed": seed,
                "evaluation_status": status,
                **_parameter_columns(selected_parameters),
            }
            _atomic_json(destination, row)
            print(
                f"selected eval r={nstep} start={start_index} "
                f"iteration={selected_iteration}: Euler-20={loglik:.2f} "
                f"(SE {standard_error:.2f}; {status})",
                flush=True,
            )
    rows = [
        json.loads(path.read_text())
        for path in sorted(
            (args.output / "selected_evaluations").glob("best_block_*.json")
        )
    ]
    _atomic_csv(args.output / "selected_evaluations.csv", pd.DataFrame(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "fit",
            "block-eval",
            "final-eval",
            "trace-eval",
            "selected-eval",
        ),
        default=(
            "fit",
            "final-eval",
            "trace-eval",
            "selected-eval",
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/smc_benchmark"))
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--nsteps", nargs="+", type=int, default=(5, 10, 20))
    parser.add_argument("--transition", choices=("block", "local"), default="block")
    parser.add_argument("--proposal", choices=("bootstrap", "guided"), default="guided")
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--learning-rate-decay-start", type=int, default=30)
    parser.add_argument("--learning-rate-decay-exponent", type=float, default=0.0)
    parser.add_argument("--burnin", type=int, default=30)
    parser.add_argument("--gain-exponent", type=float, default=0.9)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--relative-floor", type=float, default=1e-12)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--maximum-backtracks", type=int, default=6)
    parser.add_argument("--maximum-consecutive-rejections", type=int, default=10)
    parser.add_argument("--maximum-path-retries", type=int, default=3)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument(
        "--maximum-acceptable-invalid-fraction", type=float, default=0.5
    )
    parser.add_argument("--likelihood-guard-particles", type=int, default=50)
    parser.add_argument("--likelihood-guard-interval", type=int, default=10)
    parser.add_argument("--maximum-guard-loglik-drop", type=float, default=5.0)
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=800.0)
    parser.add_argument("--output-selection-seconds", type=float)
    parser.add_argument("--eval-particles", type=int, default=5000)
    parser.add_argument("--eval-replicates", type=int, default=36)
    parser.add_argument("--trace-every-seconds", type=float, default=100.0)
    parser.add_argument("--trace-every-update", action="store_true")
    parser.add_argument("--block-eval-particles", type=int, default=1000)
    parser.add_argument("--block-eval-replicates", type=int, default=4)
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
    if (
        args.output_selection_seconds is not None
        and args.output_selection_seconds <= 0.0
    ):
        raise ValueError("output-selection-seconds must be positive")
    if args.block_eval_particles < 2:
        raise ValueError("block-eval-particles must be at least two")
    if args.block_eval_replicates < 1:
        raise ValueError("block-eval-replicates must be positive")
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
    if "block-eval" in args.stages:
        evaluate_block_traces(args)
    if "final-eval" in args.stages:
        evaluate_finals(args)
    if "trace-eval" in args.stages:
        evaluate_traces(args)
    if "selected-eval" in args.stages:
        evaluate_selected(args)


if __name__ == "__main__":
    main()
