"""Numerical generalized-SAEM pilot for the existing monthly Dhaka model.

This preserves the transition approximation and smoother in block_smc. It
averages complete-path objective functions, reevaluating every retained path
at the candidate parameter. It does not average gradients computed at old
parameters. Without finite sufficient statistics, storage and M-step cost grow
with the number of iterations after burn-in. This is an exploratory extension,
not a claim that the convergence assumptions of DS19 hold for Dhaka.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .block_smc import _complete_block_path_loglik, sample_block_smoothing_path
from .data import load_dacca_data
from .model import normalize_parameters
from .warm_starts import load_starts_file


def update_weights(weights, index, gain):
    """Weights representing Q_m = (1-gain) Q_{m-1} + gain log p(path_m)."""
    if not 0 < gain <= 1 or not 0 <= index < len(weights):
        raise ValueError("Invalid averaging gain or path index")
    updated = (1.-gain)*np.asarray(weights, dtype=float)
    updated[index] += gain
    return updated


def averaged_objective(path_loglik):
    """Return the value and derivative of a weighted complete-path objective."""
    path_value_grad = jax.value_and_grad(path_loglik)
    def objective(theta, paths, weights):
        def add(total, item):
            path, weight = item
            value, grad = jax.lax.cond(weight > 0,
                lambda: path_value_grad(theta, path),
                lambda: (jnp.asarray(0., dtype=theta.dtype), jnp.zeros_like(theta)))
            return (total[0]+weight*value, total[1]+weight*grad), None
        # Differentiate one path at a time. Differentiating the outer scan
        # would store the large transition-derivative tape for every path.
        return jax.lax.scan(add, (jnp.asarray(0., dtype=theta.dtype), jnp.zeros_like(theta)),
                            (paths, weights))[0]
    return jax.jit(objective)


class TimeBudgetExceeded(RuntimeError):
    """An incomplete M-step must not replace the last completed estimate."""


def maximize_objective(value_and_grad, start, *, maxiter, bounds=None, deadline=None):
    """Numerically increase the fixed Q function, retaining the input on failure.

    A truncated M-step is generalized EM. Improvement here is in Q, not a
    guarantee of improvement in the observed likelihood with finite particles.
    """
    initial_value, initial_grad = value_and_grad(start)
    if not np.isfinite(initial_value) or not np.isfinite(initial_grad).all():
        raise ValueError("Nonfinite starting complete-data objective or gradient")

    def loss(theta):
        if deadline is not None and perf_counter() >= deadline:
            raise TimeBudgetExceeded
        value, grad = value_and_grad(theta)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            # Reject undefined line-search trials while preserving a finite
            # input estimate; these evaluations are not candidate estimates.
            return 1e100, np.zeros_like(theta)
        return -float(value), -np.asarray(grad, dtype=float)

    result = minimize(loss, np.asarray(start), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "maxls": 30,
                               "ftol": 1e-10, "gtol": 1e-6})
    final_value, final_grad = value_and_grad(result.x)
    accepted = (np.isfinite(result.x).all() and np.isfinite(final_value)
                and np.isfinite(final_grad).all() and final_value >= initial_value)
    return (result.x.copy() if accepted else np.asarray(start).copy()), {
        "q_before": float(initial_value),
        "q_after": float(final_value) if accepted else float(initial_value),
        "accepted": bool(accepted), "mstep_converged": bool(result.success),
        "mstep_iterations": int(result.nit), "mstep_evaluations": int(result.nfev),
        "mstep_message": str(result.message).strip(),
    }


def fit(start, data, *, particles=100, iterations=8, burnin=3, seed=631480,
        maxiter=25, floor=1e-12, checkpoint=None, maximum_elapsed_seconds=None,
        bounds=None, warmup_start=None):
    if iterations < 1 or not 1 <= burnin <= iterations:
        raise ValueError("Require iterations >= burnin >= 1")
    theta = np.asarray(normalize_parameters(start))
    cov = jnp.asarray(data.step_covariates)
    pop = jnp.asarray(data.observation_covariates[:, 2])
    obs = jnp.asarray(data.observations)
    value_grad = averaged_objective(lambda candidate, path:
        _complete_block_path_loglik(candidate, path, cov, pop, obs, floor))
    paths = np.zeros((iterations, len(obs), 6))
    weights = np.zeros(iterations)

    # Full-shape compilation uses a separate random seed and does not update
    # the starting parameter or contribute a trajectory to the SA average.
    warm_started = perf_counter()
    warm_theta = theta if warmup_start is None else np.asarray(normalize_parameters(warmup_start))
    warm = sample_block_smoothing_path(warm_theta, data, particles=particles,
        seed=seed+10000019, endpoint_relative_floor=floor, proposal="guided")
    if not warm.valid_path or not np.isfinite(warm.loglik):
        raise ValueError("Invalid preliminary smoothing trajectory")
    paths[0] = warm.path[1:]
    value_grad(warm_theta, paths, update_weights(weights, 0, 1.))[0].block_until_ready()
    compilation_seconds = perf_counter()-warm_started
    if checkpoint:
        checkpoint(np.asarray([theta.copy()]), pd.DataFrame(), compilation_seconds)

    rows, estimates = [], [theta.copy()]
    started = perf_counter()
    deadline = None if maximum_elapsed_seconds is None else started+maximum_elapsed_seconds
    termination = "maximum-iterations"
    for i in range(iterations):
        if deadline is not None and perf_counter() >= deadline:
            termination = "time-budget"
            break
        for retry in range(4):
            path = sample_block_smoothing_path(theta, data, particles=particles,
                seed=seed+104729*i+15485863*retry, endpoint_relative_floor=floor, proposal="guided")
            if path.valid_path and np.isfinite(path.loglik):
                break
        if not path.valid_path or not np.isfinite(path.loglik):
            raise ValueError(f"Invalid smoothing trajectory at iteration {i+1}")
        paths[i] = path.path[1:]
        gain = 1. if i < burnin else (i-burnin+1.)**(-.9)
        weights = update_weights(weights, i, gain)
        try:
            candidate, details = maximize_objective(
                lambda z: value_grad(z, paths, weights), theta, maxiter=maxiter,
                bounds=bounds, deadline=deadline)
        except TimeBudgetExceeded:
            termination = "time-budget"
            break
        if deadline is not None and perf_counter() >= deadline:
            termination = "time-budget"
            break
        # This removes only a null softmax shift, leaving the model unchanged.
        theta = np.asarray(normalize_parameters(candidate))
        row = dict(iteration=i+1, seconds=perf_counter()-started, gain=gain,
                   active_paths=int(np.count_nonzero(weights)),
                   fitting_loglik_before=path.loglik,
                   minimum_ess=float(np.min(path.ess)),
                   median_ess=float(np.median(path.ess)),
                   backward_fallbacks=path.backward_fallbacks, **details)
        rows.append(row)
        estimates.append(theta.copy())
        if checkpoint:
            checkpoint(np.asarray(estimates), pd.DataFrame(rows), compilation_seconds)
        print(f"iteration {i+1}: Q {details['q_before']:.3f} -> "
              f"{details['q_after']:.3f}, {row['seconds']:.2f}s", flush=True)
    trace = pd.DataFrame(rows)
    trace.attrs.update(termination_reason=termination, seconds=perf_counter()-started)
    return np.asarray(estimates), trace, compilation_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts-file", type=Path, default=Path(__file__).resolve().parents[1]
                        /"results/reference/ifad097_comparable_post_if2.npz")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--burnin", type=int, default=3)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--mstep-iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=631480)
    parser.add_argument("--eval-particles", type=int, default=5000)
    parser.add_argument("--eval-replicates", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(purpose="exploratory numerical generalized-SAEM pilot",
                  transition="unchanged monthly Gaussian, 20 internal steps, floor 1e-12",
                  source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)))
    source = args.output/"source"
    source.mkdir()
    config["source_hashes"] = {}
    for name in ("block_saem.py", "block_smc.py", "transition.py", "model.py", "data.py"):
        p = Path(__file__).parent/name
        shutil.copy2(p, source/name)
        config["source_hashes"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    starts = load_starts_file(args.starts_file, args.start+1)
    start = starts[args.start]
    def checkpoint(estimates, trace, compilation_seconds):
        np.savez_compressed(args.output/"fit.npz", parameters=estimates)
        trace.to_csv(args.output/"trace.csv", index=False)
        (args.output/"status.json").write_text(json.dumps({"complete": False,
            "updates": len(trace), "compilation_seconds": compilation_seconds}, indent=2)+"\n")
    estimates, trace, compilation = fit(start, load_dacca_data(20),
        particles=args.particles, iterations=args.iterations, burnin=args.burnin,
        seed=args.seed, maxiter=args.mstep_iterations, checkpoint=checkpoint)
    from .smc_benchmark import _euler_evaluate
    evaluations = []
    for name, theta in (("initial", estimates[0]), ("final", estimates[-1])):
        ll, se = _euler_evaluate(theta, particles=args.eval_particles,
            replicates=args.eval_replicates, seed=args.seed+20000033)
        evaluations.append(dict(estimate=name, loglik=ll, mcse=se))
        pd.DataFrame(evaluations).to_csv(args.output/"evaluation.csv", index=False)
        print(f"Euler {name}: {ll:.3f} (MCSE {se:.3f})", flush=True)
    (args.output/"status.json").write_text(json.dumps({"complete": True,
        "updates": len(trace), "compilation_seconds": compilation}, indent=2)+"\n")


if __name__ == "__main__":
    main()
