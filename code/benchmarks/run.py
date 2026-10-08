"""Reproducible repeated fits for the small-model comparison.

Run pilots before choosing final settings. Outputs retain all checkpoints,
independent evaluations, seeds, timings and failures. Existing output folders
are never overwritten. Example: python code/benchmarks/run.py --model linear
--output /tmp/linear-pilot --starts 2 --iterations 10 --warm 5.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypomp as pp
from scipy.optimize import minimize
from scipy.special import logsumexp

import models
from ctdd import make_filter
from smoothing import make_smoother
from pypomp.core.algorithms.contexts import MifContext, MopContext, PfilterContext
from pypomp.core.algorithms.mif import _perfilter_internal
from pypomp.core.algorithms.mop import _mop_internal
from pypomp.core.algorithms.pfilter import _pfilter_internal


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def setup(name):
    if name == "linear":
        model, y = models.linear_gaussian()
        bounds = np.array([[-.98, .98]] * 2)
        center = np.array([.5, .5])
        exact = lambda z: models.linear_loglik(dict(zip(model.canonical_param_names, z)), y)
    elif name == "oscillator":
        model, y = models.oscillator()
        bounds = np.log([[.1, 10.], [.05, 5.], [.05, 2.]])
        center = np.log([3., 1., .8])
        exact = lambda z: models.oscillator_loglik(
            dict(zip(model.canonical_param_names, jnp.exp(z))), y)
    else:
        model, y = models.spx(), None
        raw = model.theta.params(as_list=True)[0]
        from pypomp.models.spx import _to_est
        center = np.array([_to_est(raw)[n] for n in model.canonical_param_names])
        natural_bounds = {"mu": (1e-7, .01), "kappa": (1e-5, 1.),
                          "theta": (1e-7, .01), "xi": (1e-6, .1),
                          "V_0": (1e-8, .01), "rho": (-.99, .99)}
        bounds = np.array([np.log(natural_bounds[n]) if n != "rho" else
                           np.log((1 + np.array(natural_bounds[n])) /
                                  (1 - np.array(natural_bounds[n])))
                           for n in model.canonical_param_names])
        exact = None
    return model, y, bounds, center, exact


def make_if2(model, count, rw_sd):
    rw = pp.RWSigma({n: (.1 if n == "V_0" else rw_sd)
                     for n in model.canonical_param_names},
                    init_names=["V_0"] if "V_0" in model.canonical_param_names else [])
    rw = rw.geometric_cooling(.5)._canonicalize(model.canonical_param_names)
    context = MifContext.from_struct(model.to_struct(), rw, J=count, M=1)
    return jax.jit(lambda swarm, key, iteration: _perfilter_internal(
        m_current=iteration, thetas_Jd=swarm, key=key, context=context)[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "oscillator", "spx"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warm", type=int, default=100)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--ds-particles", type=int, default=100)
    parser.add_argument("--ct-particles", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--rw-sd", type=float, default=.02)
    parser.add_argument("--seed", type=int, default=631450)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--eval-particles", type=int, default=5000)
    parser.add_argument("--eval-reps", type=int, default=24)
    parser.add_argument("--methods", nargs="+", default=["IF2", "IFAD", "DS19", "CTDD21"],
                        choices=["IF2", "IFAD", "DS19", "CTDD21"])
    parser.add_argument("--purpose", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()
    if min(args.starts, args.iterations, args.particles, args.ds_particles, args.ct_particles,
           args.checkpoint_every, args.eval_particles) < 1 or args.eval_reps < 2 or args.warm < 0:
        parser.error("Invalid counts")
    args.output.mkdir(parents=True, exist_ok=False)
    model, y, bounds, center, exact = setup(args.model)
    rng = np.random.default_rng(args.seed)
    starts = np.clip(center + rng.normal(0, .35, (args.starts, len(center))),
                     bounds[:, 0], bounds[:, 1])
    pd.DataFrame(starts, columns=model.canonical_param_names).to_csv(args.output / "starts.csv", index=False)
    source_dir = Path(__file__).parent
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "utc": datetime.now(timezone.utc).isoformat(), "jax": jax.__version__,
              "device": [d.device_kind for d in jax.devices()],
              "pypomp_source": str(Path(pp.__file__).resolve()),
              "parameter_names": model.canonical_param_names,
              "bounds_estimation_scale": bounds.tolist(),
              "start_jitter_sd": .35, "alpha": .97,
              "selection": "last finite parameter vector; failures retained",
              "runtime": "synchronized serial fitting; excludes compilation and evaluation",
              "ds19": "backward path simulation and numerical score ascent; not SAEM M-step",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_dir.glob("*.py")}}
    for label, directory in (("dmop", source_dir.parents[1]), ("pypomp", Path(pp.__file__).parents[1])):
        config[label + "_commit"] = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if args.model == "spx":
        config["spx"] = models.source_manifest(model)
    else:
        pd.DataFrame(y).to_csv(args.output / "observations.csv", index=False)
    write_json(args.output / "configuration.json", config)

    if exact is not None:
        exact = jax.jit(exact)
        vg = jax.jit(jax.value_and_grad(lambda z: -exact(z)))
        optimum = minimize(lambda z: tuple(np.asarray(a) for a in vg(z)), center,
                           jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        write_json(args.output / "analytic_reference.json", {
            "parameters_estimation_scale": optimum.x.tolist(),
            "loglik": float(-optimum.fun), "converged": bool(optimum.success),
            "message": str(optimum.message)})

    if2 = make_if2(model, args.particles, args.rw_sd)
    mop_context = MopContext.from_struct(model.to_struct(), J=args.particles, alpha=.97)
    mop_score = jax.jit(jax.value_and_grad(lambda z, key: -_mop_internal(z, key, mop_context)))
    ct = make_filter(model, args.ct_particles,
                     active_names=["U"] if args.model == "oscillator" else
                                  ["V"] if args.model == "spx" else None)
    ct_score = jax.jit(jax.value_and_grad(ct))
    _, _, ds_score = make_smoother(args.model, model, y, args.ds_particles)
    scores = {"IFAD": mop_score, "DS19": ds_score, "CTDD21": ct_score}
    pf_context = PfilterContext.from_struct(model.to_struct(), J=args.eval_particles, should_trans=True)
    evaluation = jax.jit(jax.vmap(lambda z, key: -_pfilter_internal(z, key, pf_context)["neg_loglik"],
                                 in_axes=(None, 0)))

    # Warm each required executable independently; these draws never enter fits.
    compilation = {}
    for method in args.methods:
        begin = time.perf_counter()
        if method in ("IF2", "IFAD"):
            jax.block_until_ready(if2(jnp.tile(center, (args.particles, 1)),
                                      jax.random.key(9), jnp.array(0)))
        if method != "IF2":
            jax.block_until_ready(scores[method](jnp.array(center), jax.random.key(9)))
        compilation[method] = time.perf_counter() - begin
    write_json(args.output / "compilation.json", compilation)

    rows, timings, raw_evaluations = [], [], []
    for start_id, start in enumerate(starts):
        for method_id, method in enumerate(args.methods):
            z = jnp.array(start)
            swarm = jnp.tile(z, (args.particles, 1))
            m, v, average = jnp.zeros_like(z), jnp.zeros_like(z), jnp.zeros_like(z)
            elapsed, status = 0., "complete"
            snapshots = [(0, 0., np.asarray(z))]
            warm = args.warm if method == "IFAD" else 0
            for iteration in range(args.iterations + warm):
                key = jax.random.key(args.seed + 100000 * start_id + 10000 * method_id + iteration)
                begin = time.perf_counter()
                if method == "IF2" or iteration < warm:
                    swarm, _ = if2(swarm, key, jnp.array(iteration))
                    swarm = jnp.clip(swarm, bounds[:, 0], bounds[:, 1])
                    candidate = swarm.mean(0)
                else:
                    _, g = scores[method](z, key)
                    k = iteration - warm
                    if method == "DS19":
                        gain = 1. if k < 30 else (k - 29.)**(-.9)
                        average = (1 - gain) * average + gain * g
                        g = average
                    g = g * jnp.minimum(1., 100. / jnp.maximum(jnp.linalg.norm(g), 1e-12))
                    m, v = .9*m + .1*g, .999*v + .001*g*g
                    rate = args.learning_rate * (.1 + .9 * .5 * (1 + np.cos(np.pi*k/args.iterations)))
                    candidate = z + rate * (m / (1 - .9**(k+1))) / (jnp.sqrt(v / (1 - .999**(k+1))) + 1e-8)
                    candidate = jnp.clip(candidate, bounds[:, 0], bounds[:, 1])
                jax.block_until_ready(candidate)
                elapsed += time.perf_counter() - begin
                if not np.isfinite(candidate).all():
                    status = "nonfinite_update"
                    break
                z = candidate
                if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 in (warm, warm + args.iterations):
                    snapshots.append((iteration + 1, elapsed, np.asarray(z)))
            if snapshots[-1][0] != iteration + 1:
                snapshots.append((iteration + 1, elapsed, np.asarray(z)))
            timings.append({"start": start_id, "method": method, "seconds": elapsed, "status": status})
            # Independent seeds and evaluation only after fitting is finished.
            for step, seconds, params in snapshots:
                if exact is not None:
                    ll, se = float(exact(params)), 0.
                else:
                    keys = jax.random.split(jax.random.key(args.seed + 90000000 + 100000*start_id
                                            + 10000*method_id + step), args.eval_reps)
                    raw = np.asarray(evaluation(params, keys))
                    ll = float(logsumexp(raw) - np.log(len(raw)))
                    w = np.exp(raw - raw.max())
                    se = float(w.std(ddof=1) / np.sqrt(len(w)) / w.mean())
                    raw_evaluations.extend({"start": start_id, "method": method, "iteration": step,
                                            "replicate": r, "loglik": x} for r, x in enumerate(raw))
                rows.append({"model": args.model, "start": start_id, "method": method,
                             "iteration": step, "seconds": seconds, "loglik": ll, "mcse": se,
                             "final": step == snapshots[-1][0], "status": status,
                             **dict(zip(model.canonical_param_names, params))})
            pd.DataFrame(rows).to_csv(args.output / "checkpoints.csv", index=False)
            pd.DataFrame(timings).to_csv(args.output / "timings.csv", index=False)
            if raw_evaluations:
                pd.DataFrame(raw_evaluations).to_csv(args.output / "evaluation_replicates.csv", index=False)
            print(f"{args.model} {start_id+1}/{args.starts} {method}: {ll:.3f}, {elapsed:.2f}s, {status}", flush=True)
    write_json(args.output / "status.json", {"complete": True, "fits": len(timings),
        "failed": sum(row["status"] != "complete" for row in timings), "purpose": args.purpose})


if __name__ == "__main__":
    main()
