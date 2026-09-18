"""Resumable Euler-20 evaluation of an IF2 checkpoint start file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from .model import physical_parameter_dict
from .warm_starts import load_starts_file


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _evaluate(
    unconstrained: np.ndarray, particles: int, replicates: int, seed: int
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


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    starts = load_starts_file(args.starts_file, args.starts)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluations = args.output / "evaluations"
    evaluations.mkdir(exist_ok=True)
    for start, parameters in enumerate(starts):
        destination = evaluations / f"euler_start{start:03d}.json"
        if destination.exists():
            continue
        seed = args.seed + 80_000_003 + 997 * start
        try:
            loglik, standard_error = _evaluate(
                parameters, args.particles, args.replicates, seed
            )
            status = "ok" if np.isfinite(loglik) else "non-finite"
        except (FloatingPointError, RuntimeError, ValueError) as error:
            loglik = standard_error = float("nan")
            status = f"failed:{type(error).__name__}"
        _atomic_json(
            destination,
            {
                "method": "IF2 checkpoint (175 updates)",
                "start": start,
                "if2_elapsed_seconds": args.if2_elapsed_seconds,
                "evaluation_nstep": 20,
                "euler20_loglik": loglik,
                "euler20_se": standard_error,
                "eval_particles": args.particles,
                "eval_replicates": args.replicates,
                "evaluation_seed": seed,
                "evaluation_status": status,
            },
        )
        print(f"warm-start eval start={start}: {loglik:.2f} ({status})", flush=True)
    rows = [
        json.loads(path.read_text())
        for path in sorted(evaluations.glob("euler_*.json"))
    ]
    frame = pd.DataFrame(rows)
    _atomic_csv(args.output / "warm_start_evaluations.csv", frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--replicates", type=int, default=36)
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--if2-elapsed-seconds", type=float, default=244.44243359565735)
    args = parser.parse_args()
    if args.particles < 2 or args.replicates < 1:
        raise ValueError("particles must be at least two and replicates positive")
    evaluate(args)


if __name__ == "__main__":
    main()
