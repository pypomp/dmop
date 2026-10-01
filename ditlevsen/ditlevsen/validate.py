"""Run pre-benchmark validation gates for the DS-SMC Dacca extension.

This command intentionally runs only a small number of bounding-box starts.
It is not the final 100-replicate experiment and its outputs are stored under
``results/validation`` rather than consumed by the manuscript comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .benchmark import _parameter_columns, make_starts
from .data import load_dacca_data
from .model import physical_parameter_dict
from .smc import bootstrap_filter, fit_pseudo_score
from .validation_ho import (
    HOParameters,
    analytic_observed_loglik,
    conditional_particle_filter,
    fit_ho_smc_score,
    simulate_ho,
)


def _parse_fit_config(value: str) -> tuple[int, int]:
    try:
        nstep_text, start_text = value.split(":", maxsplit=1)
        result = (int(nstep_text), int(start_text))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("fit configs must have form NSTEP:START") from error
    if result[0] < 1 or result[1] < 0:
        raise argparse.ArgumentTypeError("NSTEP must be positive and START nonnegative")
    return result


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


def run_validation(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration["output"] = str(args.output)
    configuration["fit_config"] = [list(value) for value in args.fit_config]
    (args.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )

    trajectory = simulate_ho(points=1000, seed=args.seed + 1)
    ho = conditional_particle_filter(
        trajectory[:, 0], particles=100, seed=args.seed + 2
    )
    ho_summary = {
        "points": 1000,
        "particles": 100,
        "smc_vs_analytic_rmse": float(
            np.sqrt(np.mean((ho.filtered_mean - ho.analytic_mean) ** 2))
        ),
        "smc_vs_hidden_rmse": float(
            np.sqrt(np.mean((ho.filtered_mean - trajectory[:, 1]) ** 2))
        ),
        "analytic_vs_hidden_rmse": float(
            np.sqrt(np.mean((ho.analytic_mean - trajectory[:, 1]) ** 2))
        ),
        "median_ess": float(np.median(ho.ess)),
        "minimum_ess": float(np.min(ho.ess)),
        "unique_initial_ancestors": ho.unique_initial_ancestors,
        "loglik": ho.loglik,
    }
    (args.output / "harmonic_oscillator.json").write_text(
        json.dumps(ho_summary, indent=2) + "\n"
    )
    print(f"HO validation: {ho_summary}", flush=True)

    ho_fit = fit_ho_smc_score(
        trajectory[:, 0],
        particles=100,
        iterations=args.ho_score_iterations,
        learning_rate=args.ho_learning_rate,
        burnin=30,
        seed=args.seed + 3,
    )

    def negative_ho_loglik(raw: np.ndarray) -> float:
        physical = np.exp(raw)
        return -analytic_observed_loglik(
            trajectory[:, 0],
            parameters=HOParameters(
                stiffness=float(physical[0]),
                damping=float(physical[1]),
                diffusion=float(physical[2]),
            ),
        )

    ho_mle = minimize(
        negative_ho_loglik,
        np.log(np.asarray([4.0, 0.5, 0.5])),
        method="L-BFGS-B",
        bounds=[
            (np.log(0.1), np.log(10.0)),
            (np.log(0.05), np.log(5.0)),
            (np.log(0.05), np.log(2.0)),
        ],
    )
    mle_parameters = np.exp(ho_mle.x)
    fitted_parameters = np.asarray(
        [
            ho_fit.parameters.stiffness,
            ho_fit.parameters.damping,
            ho_fit.parameters.diffusion,
        ]
    )
    ho_score_summary = {
        "iterations": args.ho_score_iterations,
        "particles": 100,
        "truth": [4.0, 0.5, 0.5],
        "start": ho_fit.parameter_trace[0].tolist(),
        "final": fitted_parameters.tolist(),
        "trajectory_mle": mle_parameters.tolist(),
        "start_loglik": float(ho_fit.marginal_loglik_trace[0]),
        "final_loglik": float(ho_fit.marginal_loglik_trace[-1]),
        "trajectory_mle_loglik": float(-ho_mle.fun),
        "loglik_gap_to_mle": float(-ho_mle.fun - ho_fit.marginal_loglik_trace[-1]),
        "mle_converged": bool(ho_mle.success),
    }
    (args.output / "harmonic_oscillator_score.json").write_text(
        json.dumps(ho_score_summary, indent=2) + "\n"
    )
    pd.DataFrame(
        {
            "iteration": np.arange(len(ho_fit.marginal_loglik_trace)),
            "D": ho_fit.parameter_trace[:, 0],
            "gamma": ho_fit.parameter_trace[:, 1],
            "sigma": ho_fit.parameter_trace[:, 2],
            "observed_loglik": ho_fit.marginal_loglik_trace,
            "complete_loglik": ho_fit.complete_loglik_trace,
            "score_norm": ho_fit.score_norm_trace,
        }
    ).to_csv(args.output / "harmonic_oscillator_score_trace.csv", index=False)
    print(f"HO score validation: {ho_score_summary}", flush=True)

    highest_start = max(
        [args.starts - 1] + [start_index for _, start_index in args.fit_config]
    )
    starts = make_starts(highest_start + 1, args.seed)
    filter_rows: list[dict[str, float | int | bool]] = []
    for nstep in args.filter_nsteps:
        data = load_dacca_data(nstep)
        for start_index in range(args.starts):
            result = bootstrap_filter(
                starts[start_index],
                data,
                particles=args.filter_particles,
                seed=args.seed + 1000 * nstep + start_index,
                order=args.order,
                relative_floor=args.relative_floor,
            )
            row = {
                "inference_nstep": nstep,
                "start": start_index,
                "particles": args.filter_particles,
                "loglik": result.loglik,
                "median_ess": float(np.median(result.ess)),
                "minimum_ess": float(np.min(result.ess)),
                "maximum_invalid_fraction": float(np.max(result.invalid_fraction)),
                "final_invalid_fraction": float(result.invalid_fraction[-1]),
                "collapsed": bool(np.max(result.invalid_fraction) >= 1.0),
            }
            filter_rows.append(row)
            print(f"filter validation: {row}", flush=True)
    pd.DataFrame(filter_rows).to_csv(args.output / "filter_viability.csv", index=False)

    summary_rows: list[dict[str, float | int | bool]] = []
    trace_rows: list[dict[str, float | int]] = []
    for fit_index, (nstep, start_index) in enumerate(args.fit_config):
        data = load_dacca_data(nstep)
        start = starts[start_index]
        fit = fit_pseudo_score(
            start,
            data,
            particles=args.fit_particles,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            burnin=args.burnin,
            gain_exponent=args.gain_exponent,
            seed=args.seed + 100_000 + fit_index,
            order=args.order,
            relative_floor=args.relative_floor,
            bridge_particles=args.bridge_particles,
        )

        before_local = []
        after_local = []
        for replicate in range(args.independent_replicates):
            evaluation_seed = args.seed + 200_000 + 100 * fit_index + replicate
            before_local.append(
                bootstrap_filter(
                    start,
                    data,
                    particles=args.independent_particles,
                    seed=evaluation_seed,
                    order=args.order,
                    relative_floor=args.relative_floor,
                ).loglik
            )
            after_local.append(
                bootstrap_filter(
                    fit.unconstrained,
                    data,
                    particles=args.independent_particles,
                    seed=evaluation_seed,
                    order=args.order,
                    relative_floor=args.relative_floor,
                ).loglik
            )
        paired_local = np.asarray(after_local) - np.asarray(before_local)

        if args.skip_euler:
            euler_before = euler_after = euler_before_se = euler_after_se = np.nan
        else:
            euler_before, euler_before_se = _euler_evaluate(
                start,
                particles=args.euler_particles,
                replicates=args.euler_replicates,
                seed=args.seed + 300_000 + 2 * fit_index,
            )
            euler_after, euler_after_se = _euler_evaluate(
                fit.unconstrained,
                particles=args.euler_particles,
                replicates=args.euler_replicates,
                seed=args.seed + 300_001 + 2 * fit_index,
            )

        summary = {
            "inference_nstep": nstep,
            "start": start_index,
            "updates": fit.completed_updates,
            "termination_reason": fit.termination_reason,
            "total_backtracks": int(np.sum(fit.backtrack_count_trace)),
            "fit_particles": args.fit_particles,
            "bridge_particles": args.bridge_particles,
            "elapsed_seconds": fit.elapsed_seconds,
            "independent_smc_before": float(np.mean(before_local)),
            "independent_smc_after": float(np.mean(after_local)),
            "independent_smc_paired_change": float(np.mean(paired_local)),
            "independent_smc_paired_se": float(
                np.std(paired_local, ddof=1) / np.sqrt(len(paired_local))
            )
            if len(paired_local) > 1
            else np.nan,
            "euler20_before": euler_before,
            "euler20_before_se": euler_before_se,
            "euler20_after": euler_after,
            "euler20_after_se": euler_after_se,
            "euler20_change": euler_after - euler_before,
            "minimum_ess": float(np.min(fit.minimum_ess_trace)),
            "maximum_invalid_fraction": float(
                np.max(fit.maximum_invalid_fraction_trace)
            ),
            "maximum_unique_initial_ancestors": int(
                np.max(fit.unique_initial_ancestors_trace)
            ),
            **{f"start_{key}": value for key, value in _parameter_columns(start).items()},
            **{
                f"final_{key}": value
                for key, value in _parameter_columns(fit.unconstrained).items()
            },
        }
        summary_rows.append(summary)
        print(
            f"score validation nstep={nstep} start={start_index}: "
            f"DS-SMC {summary['independent_smc_before']:.2f} -> "
            f"{summary['independent_smc_after']:.2f}; Euler-20 "
            f"{summary['euler20_before']:.2f} -> {summary['euler20_after']:.2f}",
            flush=True,
        )
        for iteration in range(len(fit.marginal_loglik_trace)):
            trace_rows.append(
                {
                    "inference_nstep": nstep,
                    "start": start_index,
                    "iteration": iteration,
                    "elapsed_seconds": float(fit.elapsed_trace[iteration]),
                    "smc_loglik": float(fit.marginal_loglik_trace[iteration]),
                    "complete_loglik": float(fit.complete_loglik_trace[iteration]),
                    "score_norm": float(fit.score_norm_trace[iteration]),
                    "median_ess": float(fit.median_ess_trace[iteration]),
                    "minimum_ess": float(fit.minimum_ess_trace[iteration]),
                    "maximum_invalid_fraction": float(
                        fit.maximum_invalid_fraction_trace[iteration]
                    ),
                    "unique_initial_ancestors": int(
                        fit.unique_initial_ancestors_trace[iteration]
                    ),
                    "bridge_update_fraction": float(
                        fit.bridge_update_fraction_trace[iteration]
                    ),
                    "bridge_median_ess": float(
                        fit.bridge_median_ess_trace[iteration]
                    ),
                    "accepted_step_size": float(
                        fit.accepted_step_size_trace[iteration]
                    ),
                    "backtrack_count": int(
                        fit.backtrack_count_trace[iteration]
                    ),
                    **_parameter_columns(fit.parameter_trace[iteration]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(args.output / "score_pilot_summary.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(args.output / "score_pilot_traces.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/validation"))
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--ho-score-iterations", type=int, default=160)
    parser.add_argument("--ho-learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--filter-nsteps", nargs="+", type=int, default=[1, 2, 5, 10, 20]
    )
    parser.add_argument("--filter-particles", type=int, default=128)
    parser.add_argument(
        "--fit-config",
        nargs="+",
        type=_parse_fit_config,
        default=[(5, 0), (5, 1), (5, 2), (20, 0)],
        metavar="NSTEP:START",
    )
    parser.add_argument("--fit-particles", type=int, default=128)
    parser.add_argument("--bridge-particles", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--burnin", type=int, default=30)
    parser.add_argument("--gain-exponent", type=float, default=0.9)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--relative-floor", type=float, default=1e-7)
    parser.add_argument("--independent-particles", type=int, default=512)
    parser.add_argument("--independent-replicates", type=int, default=4)
    parser.add_argument("--euler-particles", type=int, default=2000)
    parser.add_argument("--euler-replicates", type=int, default=4)
    parser.add_argument("--skip-euler", action="store_true")
    return parser


def main() -> None:
    run_validation(build_parser().parse_args())


if __name__ == "__main__":
    main()
