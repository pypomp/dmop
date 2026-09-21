"""Shared settings and fitting functions for the Daphnia comparison."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import pandas as pd
import pypomp as pp
from scipy.special import logsumexp

import model

PYPOMP_COMMIT = "4f02494185dd5b959e26a868d0814f1f190081ad"
REFERENCE_COMMIT = "f3183301273a409e5b079bb0ae3a22246f69ba7a"
FIXED = ("sigSn", "sigSi")


def check_model(config):
    source = Path(model.__file__).read_text()
    original = source.replace('Path(__file__).with_name("data") / "Mesocosmdata.xls"',
                              'Path(__file__).resolve().parents[1] / "daphnia-reference/data/Mesocosmdata.xls"')
    if config["source_sha256"]["model.py"] not in {
        hashlib.sha256(text.encode()).hexdigest() for text in (source, original)
    }:
        raise ValueError("Model differs from the saved search")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def payloads(theta):
    return theta.params(as_list=True)


def save_parameters(path, theta):
    rows = []
    for i, value in enumerate(payloads(theta)):
        for block, frame in value.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            for parameter in frame.index:
                for unit in frame.columns:
                    rows.append((i, block, str(unit), parameter, float(frame.loc[parameter, unit])))
    pd.DataFrame(rows, columns=["start", "block", "unit", "parameter", "value"]).to_csv(path, index=False)


def check_parameters(panel):
    for value in payloads(panel.theta):
        for block in ("shared", "unit_specific"):
            frame = value[block]
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError("Non-finite parameter estimate")
            free = frame.drop(index=list(FIXED), errors="ignore")
            if not (free.to_numpy(dtype=float) > 0).all():
                raise ValueError("A positive parameter reached zero or became negative")
        for name in FIXED:
            if float(value["shared"].loc[name].iloc[0]) != 0:
                raise ValueError(f"Fixed parameter changed: {name}")


def learning_rate(panel, peak, iterations):
    warmup = min(10, iterations)
    rates = np.concatenate((np.linspace(0.1 * peak, peak, warmup), np.full(iterations - warmup, peak)))
    return pp.LearningRate({
        name: 0.0 if name in FIXED else rates
        for name in panel.canonical_param_names
    }).cosine_decay(final_factor=0.1, M=iterations)


def evaluate(panel, args, seed):
    panel.pfilter(J=args.eval_particles, reps=args.eval_reps, key=jax.random.key(seed))
    raw = np.asarray(panel.results_history[-1].logLiks.values)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid PF evaluation: shape {raw.shape}")
    # Repeated likelihoods are averaged within units on the natural scale.
    unit_ll = logsumexp(raw, axis=-1) - np.log(raw.shape[-1])
    weights = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
    unit_se = weights.std(axis=-1, ddof=1) / np.sqrt(raw.shape[-1]) / weights.mean(axis=-1)
    table = pd.DataFrame({
        "start": np.arange(raw.shape[0]),
        "logLik": unit_ll.sum(axis=-1),
        "MCSE": np.sqrt(np.square(unit_se).sum(axis=-1)),
    })
    return table, raw


def checkpoint(panel, directory, label):
    check_parameters(panel)
    save_parameters(directory / f"{label}_parameters.csv", panel.theta)
    with (directory / f"{label}.pkl").open("wb") as stream:
        pickle.dump(panel.theta, stream)
    panel.traces().to_csv(directory / f"{label}_traces.csv.gz", index=False)


def warm(starts, args, directory, seed):
    panel = model.build_panel(starts=starts)
    save_parameters(directory / "initial_parameters.csv", panel.theta)
    started = time.perf_counter()
    rw = pp.RWSigma({name: 0.0 if name in FIXED else 0.05 for name in panel.canonical_param_names}, init_names=[])
    panel.mif(J=args.particles, M=args.warm_iterations, rw_sd=rw.geometric_cooling(0.7),
              block=True, key=jax.random.key(seed))
    seconds = time.perf_counter() - started
    checkpoint(panel, directory, "mpif")
    table, raw = evaluate(panel, args, seed + 100)
    table.to_csv(directory / "mpif_loglik.csv", index=False)
    np.save(directory / "mpif_pf.npy", raw)
    write_json(directory / "mpif_timing.json", {"iterations": args.warm_iterations, "seconds": seconds})
    return panel.theta, table, seconds


def train(theta, args, directory, lr, alpha, seed):
    panel = model.build_panel(starts=theta)
    started = time.perf_counter()
    # A single call preserves Adam moments for all 200 updates.
    panel.train(J=args.particles, M=args.adam_iterations,
                eta=learning_rate(panel, lr, args.adam_iterations),
                optimizer=pp.Adam(beta1=0.9, beta2=0.999, epsilon=1e-8),
                alpha=alpha, alpha_cooling=1.0, chunk_size=8, key=jax.random.key(seed))
    seconds = time.perf_counter() - started
    trace = np.asarray(panel.results_history[-1].shared_traces.values)
    unit_trace = np.asarray(panel.results_history[-1].unit_traces.values)
    if not np.isfinite(trace[:, 1:, :]).all() or not np.isfinite(unit_trace[:, 1:, :, 1:]).all():
        raise ValueError("Non-finite training objective or parameters")
    checkpoint(panel, directory, "ifad")
    table, raw = evaluate(panel, args, seed + 100)
    table.to_csv(directory / "ifad_loglik.csv", index=False)
    np.save(directory / "ifad_pf.npy", raw)
    write_json(directory / "train_timing.json", {"iterations": args.adam_iterations, "seconds": seconds})
    return table, seconds


def arguments(method):
    parser = argparse.ArgumentParser(description=f"Daphnia {method} search")
    if method == "IFAD":
        parser.add_argument("--alpha", type=float, choices=(0.0, 0.97, 1.0),
                            default=float(os.environ.get("ALPHA", "0.97")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--starts", type=int, default=50)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--warm-iterations", type=int, default=200 if method == "IFAD" else 680)
    if method == "IFAD":
        parser.add_argument("--adam-iterations", type=int, default=200)
    parser.add_argument("--eval-particles", type=int, default=2000)
    parser.add_argument("--eval-reps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if method == "MPIF":
        args.adam_iterations = 0
    if not args.allow_cpu and jax.default_backend() != "gpu":
        parser.error("A CUDA GPU is required; use --allow-cpu for small checks")
    if pp.__version__ != "1.0.5" or not jax.config.jax_enable_x64:
        parser.error("Expected PyPOMP 1.0.5 and float64")
    if (min(args.starts, args.particles, args.warm_iterations, args.eval_particles,
            args.batch_size) < 1 or args.eval_reps < 2
            or (method == "IFAD" and args.adam_iterations < 1)):
        parser.error("Counts must be positive; evaluation needs at least two replicates")
    arm = {0.0: "alpha0", 0.97: "formal", 1.0: "alpha1"}[args.alpha] if method == "IFAD" else "mpif680"
    args.output = args.output or Path("results") / arm
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    names = ("model.py", "prep.py", "dmop_search.py" if method == "IFAD" else "mif_search.py")
    write_json(args.output / "config.json", {
        **config, "method": method, "learning_rate": 0.01 if method == "IFAD" else None,
        "start_seed": 2026091802, "rw_sd": 0.05, "if2_cooling": 0.7,
        "chunk_size": 8, "warmup": 10, "cosine_final_factor": 0.1,
        "pypomp_version": pp.__version__, "pypomp_commit": PYPOMP_COMMIT,
        "reference_commit": REFERENCE_COMMIT, "jax_version": jax.__version__,
        "device": [d.device_kind for d in jax.devices()], "float64": True,
        "source_sha256": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                          for n in names},
    })
    return args
