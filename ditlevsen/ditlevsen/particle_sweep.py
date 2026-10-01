"""Resumable particle-count and optimizer sweep for the Dacca block method."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pandas as pd


@dataclass(frozen=True)
class SweepConfiguration:
    particles: int
    learning_rate: float
    burnin: int
    decay_start: int
    name: str | None = None

    @property
    def label(self) -> str:
        if self.name is not None:
            return self.name
        rate = f"{self.learning_rate:.3f}".replace(".", "p")
        return f"j{self.particles:04d}_lr{rate}"


def sweep_configurations() -> tuple[SweepConfiguration, ...]:
    """Two learning rates at each of four particle counts.

    The SA averaging and learning-rate schedules are shortened as particle
    count increases because fewer score updates fit into a fixed wall-clock
    budget.
    """

    settings = (
        (100, 0.050, 30, 90),
        (500, 0.090, 20, 60),
        (1000, 0.110, 15, 45),
        (5000, 0.200, 5, 15),
    )
    return tuple(
        SweepConfiguration(particles, multiplier * rate, burnin, decay_start)
        for particles, rate, burnin, decay_start in settings
        for multiplier in (1.0, 2.0)
    )


def focused_j5000_configurations() -> tuple[SweepConfiguration, ...]:
    """Refine learning rate and SA burn-in after the particle-count screen."""

    settings = (
        (0.100, 5, 45),
        (0.100, 30, 45),
        (0.150, 15, 45),
        (0.200, 30, 45),
    )
    return tuple(
        SweepConfiguration(
            5000,
            learning_rate,
            burnin,
            decay_start,
            name=(
                f"j5000_lr{learning_rate:.3f}_b{burnin:02d}_d{decay_start:02d}".replace(
                    ".", "p"
                )
            ),
        )
        for learning_rate, burnin, decay_start in settings
    )


def _run_configuration(
    configuration: SweepConfiguration,
    *,
    output: Path,
    seed: int,
    starts: int,
    maximum_elapsed_seconds: float,
    eval_replicates: int,
) -> None:
    destination = output / configuration.label
    command = [
        sys.executable,
        "-u",
        "-m",
        "ditlevsen.smc_benchmark",
        "--stages",
        "fit",
        "final-eval",
        "--output",
        str(destination),
        "--seed",
        str(seed),
        "--starts",
        str(starts),
        "--nsteps",
        "20",
        "--transition",
        "block",
        "--proposal",
        "guided",
        "--particles",
        str(configuration.particles),
        "--iterations",
        "5000",
        "--learning-rate",
        str(configuration.learning_rate),
        "--learning-rate-decay-start",
        str(configuration.decay_start),
        "--learning-rate-decay-exponent",
        "0.3",
        "--burnin",
        str(configuration.burnin),
        "--gain-exponent",
        "0.9",
        "--order",
        "2",
        "--relative-floor",
        "1e-12",
        "--gradient-clip",
        "100",
        "--maximum-backtracks",
        "6",
        "--maximum-consecutive-rejections",
        "10",
        "--maximum-path-retries",
        "3",
        "--backtrack-factor",
        "0.5",
        "--maximum-acceptable-invalid-fraction",
        "0.8",
        "--likelihood-guard-particles",
        str(configuration.particles),
        "--likelihood-guard-interval",
        "10",
        "--maximum-guard-loglik-drop",
        "5",
        "--maximum-elapsed-seconds",
        str(maximum_elapsed_seconds),
        "--eval-particles",
        "5000",
        "--eval-replicates",
        str(eval_replicates),
        "--workers",
        "1",
        "--worker-index",
        "0",
    ]
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    subprocess.run(command, check=True, env=environment)


def _write_summary(
    output: Path, configurations: tuple[SweepConfiguration, ...]
) -> None:
    rows: list[pd.DataFrame] = []
    for configuration in configurations:
        destination = output / configuration.label
        fit_path = destination / "fit_summary.csv"
        evaluation_path = destination / "final_evaluations.csv"
        if not fit_path.exists() or not evaluation_path.exists():
            continue
        fit = pd.read_csv(fit_path)
        evaluation = pd.read_csv(evaluation_path)
        merged = evaluation.merge(
            fit[
                [
                    "start",
                    "elapsed_seconds",
                    "updates_completed",
                    "initial_pseudo_loglik",
                    "final_pseudo_loglik",
                ]
            ],
            on=["start", "updates_completed"],
            how="left",
        )
        merged.insert(0, "configuration", configuration.label)
        merged.insert(1, "fit_particles", configuration.particles)
        merged.insert(2, "learning_rate", configuration.learning_rate)
        merged.insert(3, "burnin", configuration.burnin)
        merged.insert(4, "decay_start", configuration.decay_start)
        rows.append(merged)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            output / "sweep_summary.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/particle_sweep"))
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=400.0)
    parser.add_argument("--eval-replicates", type=int, default=36)
    parser.add_argument(
        "--configuration-set",
        choices=("particle", "focused-j5000"),
        default="particle",
    )
    args = parser.parse_args()
    configurations = (
        sweep_configurations()
        if args.configuration_set == "particle"
        else focused_j5000_configurations()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for configuration in configurations:
        print(f"starting {configuration.label}", flush=True)
        _run_configuration(
            configuration,
            output=args.output,
            seed=args.seed,
            starts=args.starts,
            maximum_elapsed_seconds=args.maximum_elapsed_seconds,
            eval_replicates=args.eval_replicates,
        )
        _write_summary(args.output, configurations)


if __name__ == "__main__":
    main()
