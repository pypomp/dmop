"""Replay one fixed-path Dhaka M-step without changing the live A/B experiment.

Positive scaling leaves the mathematical M-step unchanged but can affect
L-BFGS-B's numerical line search and stopping rules. This is a diagnostic,
not an additional fitted benchmark or a rule for selecting reported fits.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ditlevsen.block_saem import maximize_objective
from ditlevsen.block_smc import _complete_block_path_loglik, sample_block_smoothing_path
from ditlevsen.data import load_dacca_data
from ditlevsen.model import ESTIMATED_PARAMETER_NAMES, normalize_parameters, parameter_bounds

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = ROOT / "ditlevsen/results/saem_ab"
    config = json.loads((root / "protocol.json").read_text())
    initial = np.asarray(normalize_parameters(np.load(root / "starts.npz")["cold"][args.start]))
    data = load_dacca_data(20)
    seed = config["seed"] + 1000003 * 20 + 10007 * args.start
    path = sample_block_smoothing_path(initial, data, particles=config["particles"],
        seed=seed, endpoint_relative_floor=1e-12, proposal="guided")
    if not path.valid_path:
        raise ValueError("Cannot replay an invalid first path")
    target = jnp.asarray(path.path[1:])
    cov = jnp.asarray(data.step_covariates)
    pop = jnp.asarray(data.observation_covariates[:, 2])
    obs = jnp.asarray(data.observations)
    value_grad = jax.jit(jax.value_and_grad(lambda theta:
        _complete_block_path_loglik(theta, target, cov, pop, obs, 1e-12)))
    value, grad = value_grad(initial)
    value, grad = float(value), np.asarray(grad)
    reference = pd.read_csv(root / "cold" / f"start{args.start:03d}" / "saem_trace.csv").iloc[0]
    np.testing.assert_allclose(value, reference.q_before, rtol=1e-10, atol=1e-6)
    lower, upper = map(np.asarray, parameter_bounds())
    scale = max(1., float(np.abs(grad).max()))
    direction = grad / scale
    probes = []
    for step in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        candidate = np.clip(initial + step * direction, lower, upper)
        trial, _ = value_grad(candidate)
        probes.append({"step": step, "q_change": float(trial) - value if np.isfinite(trial) else None,
                       "linear_prediction": float(grad @ (candidate - initial))})
    runs, estimates = [], {}
    for name, divisor in (("original", 1.), ("gradient_scaled", scale)):
        def scaled(theta):
            q, g = value_grad(theta)
            return q / divisor, g / divisor
        candidate, details = maximize_objective(scaled, initial,
            maxiter=config["mstep_iterations"], bounds=list(zip(lower, upper)),
            scale_objective=False)
        q, _ = value_grad(candidate)
        estimates[name] = candidate
        runs.append({"variant": name, "divisor": divisor, "q_change": float(q) - value,
                     "parameter_displacement": float(np.linalg.norm(candidate - initial)), **details})
        print(runs[-1], flush=True)
    result = {"cold_start": args.start, "seed": seed, "particles": config["particles"],
              "q_initial": value, "gradient_norm": float(np.linalg.norm(grad)),
              "largest_gradient_parameter": ESTIMATED_PARAMETER_NAMES[int(np.argmax(np.abs(grad)))],
              "largest_absolute_gradient": scale, "directional_probes": probes, "msteps": runs,
              "scope": "one fixed path; no observed likelihood evaluation or benchmark selection"}
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output / "fixed_path.npz", initial=initial, path=np.asarray(target),
                        gradient=grad, **estimates)


if __name__ == "__main__":
    main()
