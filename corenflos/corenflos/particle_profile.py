"""Runtime and pseudo-likelihood profile across fitting particle counts."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ditlevsen.benchmark import make_starts
from ditlevsen.data import load_dacca_data
from ditlevsen.model import default_unconstrained_parameters

from .dpf import DPFConfig, DaccaArrays
from .fit import make_value_and_gradient
from .transport import TransportConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/particle_profile.csv")
    )
    parser.add_argument(
        "--particles", nargs="+", type=int, default=(25, 50, 100, 250, 500)
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=631409)
    parser.add_argument("--epsilon", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = DaccaArrays.from_data(load_dacca_data(20))
    parameter_sets = {
        "reference": jnp.asarray(default_unconstrained_parameters()),
        "box-start-0": jnp.asarray(make_starts(1, args.seed)[0]),
    }
    rows: list[dict[str, float | int | str]] = []
    for particles in args.particles:
        config = DPFConfig(
            particles=particles,
            transport=TransportConfig(epsilon=args.epsilon),
        )
        evaluator = make_value_and_gradient(data, config)
        compiled = False
        for parameter_name, parameters in parameter_sets.items():
            for replicate in range(args.replicates):
                key = jax.random.key(args.seed + 1_000_003 * particles + 97 * replicate)
                started = time.perf_counter()
                result, gradient = evaluator(parameters, key)
                jax.block_until_ready(gradient)
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "particles": particles,
                        "parameter_set": parameter_name,
                        "replicate": replicate,
                        "includes_compilation": not compiled,
                        "elapsed_seconds": elapsed,
                        "pseudo_loglik": float(result.loglik),
                        "gradient_norm": float(jnp.linalg.norm(gradient)),
                        "resampling_count": int(jnp.sum(result.resampled)),
                        "minimum_ess": float(jnp.min(result.ess)),
                        "maximum_invalid_fraction": float(
                            jnp.max(result.invalid_fraction)
                        ),
                    }
                )
                compiled = True
        del evaluator
        jax.clear_caches()
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = (
        frame.loc[~frame["includes_compilation"]]
        .groupby(["particles", "parameter_set"])
        .agg(
            median_seconds=("elapsed_seconds", "median"),
            mean_pseudo_loglik=("pseudo_loglik", "mean"),
            sd_pseudo_loglik=("pseudo_loglik", "std"),
        )
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
