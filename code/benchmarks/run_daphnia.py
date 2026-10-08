"""Common-start Daphnia comparison with independent Euler evaluation.

MPIF and IFAD retain S10's settings. DS19 and CTDD21 start from the same
200-iteration MPIF estimate as IFAD and receive its measured continuation
time. The last valid update is retained. Each method pays for the warm start.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypomp as pp
from scipy.special import logsumexp

import daphnia as d
from run import write_json


def panel_trace(panel, seconds, stride, offset=0, initial_seconds=0.):
    result = panel.results_history[-1]
    shared = np.asarray(result.shared_traces.values)[0]
    unit = np.asarray(result.unit_traces.values)[0]
    total = len(shared) - 1
    steps = sorted(set(range(0, total + 1, stride)) | {total})
    snapshots = []
    for i in steps:
        p = dict(zip(panel.canonical_shared_param_names, shared[i, 1:]))
        p.update({f"{n}_{u}": unit[i, j, k+1]
                  for j, u in enumerate(panel.get_unit_names())
                  for k, n in enumerate(panel.canonical_unit_param_names)})
        z = np.log(np.array([p[n] for n in d.NAMES]))
        snapshots.append((offset+i, initial_seconds+seconds*i/max(total, 1), z))
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026091802)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--warm-iterations", type=int, default=200)
    parser.add_argument("--mif-iterations", type=int, default=680)
    parser.add_argument("--adam-iterations", type=int, default=200)
    parser.add_argument("--ds-particles", type=int, default=100)
    parser.add_argument("--ct-particles", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--competitor-learning-rate", type=float, default=.01)
    parser.add_argument("--ds-learning-rate", type=float)
    parser.add_argument("--ct-learning-rate", type=float)
    parser.add_argument("--eval-particles", type=int, default=2000)
    parser.add_argument("--eval-reps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--purpose", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()
    if min(args.starts, args.particles, args.warm_iterations, args.mif_iterations,
           args.adam_iterations, args.ds_particles, args.ct_particles,
           args.eval_particles, args.stride, args.max_updates) < 1 or args.eval_reps < 2 or args.start_index < 0:
        parser.error("Invalid counts")
    args.output.mkdir(parents=True, exist_ok=False)
    starts = d.reference.sample_starts(args.starts+args.start_index, args.seed)[args.start_index:]
    initial = np.array([d.pack(x) for x in starts])
    pd.DataFrame(initial, columns=d.NAMES).to_csv(args.output/"starts.csv", index=False)
    root = Path(__file__).resolve().parents[2]
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "utc": datetime.now(timezone.utc).isoformat(), "device": [x.device_kind for x in jax.devices()],
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "jax": jax.__version__,
              "dmop_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "pypomp_commit": subprocess.check_output(["git", "-C", str(Path(pp.__file__).parents[1]),
                                                        "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), Path(d.__file__), d._path, d.reference.DATA_PATH]},
              "ds_model": "composed strong-1.5 Gaussian moments with endpoint floor 1e-8; guided NB surrogate",
              "ds_update": "backward-simulated complete-data score and Adam; not the SAEM M-step",
              "selection": "last finite parameter vector",
              "timing": "serial fits; compilation reported separately; evaluation excluded",
              "progress_time": "uniform within the two Pypomp stages; measured per-update for DS19/CTDD21"}
    write_json(args.output/"configuration.json", config)
    (args.output/"source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, args.output/"source"/path.name)

    ct, pf = d.make_euler_objectives(args.ct_particles, args.eval_particles)
    ct_score = jax.jit(jax.value_and_grad(ct))
    ds_score, _, _ = d.make_ds_score(args.ds_particles)
    evaluate = jax.jit(jax.vmap(pf, in_axes=(None, 0)))
    scores = {"DS19": ds_score, "CTDD21": ct_score}
    fixed = d.reference.FIXED_PARAMETERS

    def run_mif(payload, count, seed):
        panel = d.reference.build_panel([payload])
        rw = pp.RWSigma({n: 0. if n in fixed else .05 for n in panel.canonical_param_names},
                        init_names=[]).geometric_cooling(.7)
        begin = time.perf_counter()
        panel.mif(J=args.particles, M=count, rw_sd=rw, block=True, key=jax.random.key(seed))
        return panel, time.perf_counter()-begin

    def run_adam(theta, seed):
        panel = d.reference.build_panel(theta)
        rates = np.concatenate((np.linspace(.1, 1., min(10, args.adam_iterations)),
                                np.ones(max(0, args.adam_iterations-10))))*args.learning_rate
        eta = pp.LearningRate({n: 0. if n in fixed else rates for n in panel.canonical_param_names})
        eta = eta.cosine_decay(final_factor=.1, M=args.adam_iterations)
        begin = time.perf_counter()
        panel.train(J=args.particles, M=args.adam_iterations, eta=eta,
                    optimizer=pp.Adam(beta1=.9, beta2=.999, epsilon=1e-8),
                    alpha=.97, alpha_cooling=1., chunk_size=8, key=jax.random.key(seed))
        return panel, time.perf_counter()-begin

    # Compile the exact shapes used below, on a separate initialization.
    # Do not reuse the resulting fit in the reported comparison.
    compilation = {}
    warm_payload = d.reference.sample_starts(1, args.seed+999)[0]
    warm, compilation["MPIF_warm"] = run_mif(warm_payload, args.warm_iterations, args.seed+9000)
    _, compilation["MPIF_full"] = run_mif(warm_payload, args.mif_iterations, args.seed+9001)
    _, compilation["IFAD"] = run_adam(warm.theta, args.seed+9002)
    warm_z = jnp.array(d.pack(warm.theta.params(as_list=True)[0]))
    for name, fn in scores.items():
        begin = time.perf_counter()
        jax.block_until_ready(fn(warm_z, jax.random.key(args.seed+9003)))
        compilation[name] = time.perf_counter()-begin
    write_json(args.output/"compilation.json", compilation)
    print("compilation complete", compilation, flush=True)

    rows, timing, replicates, objectives = [], [], [], []
    for start_id, payload in enumerate(starts, args.start_index):
        fits = {}

        def save_fit(name, trace, seconds, status="complete"):
            if any(not np.isfinite(z).all() for _, _, z in trace):
                trace = [row for row in trace if np.isfinite(row[2]).all()]
                status = "nonfinite_update"
            fits[name] = (trace, seconds, status)
            pd.DataFrame([{"iteration": i, "seconds": t, **dict(zip(d.NAMES, z))}
                          for i,t,z in trace]).to_csv(args.output/f"parameters_{name}_{start_id:03d}.csv", index=False)
            write_json(args.output/f"fit_{name}_{start_id:03d}.json",
                       {"start": start_id, "method": name, "seconds": seconds, "status": status})

        warm, warm_time = run_mif(payload, args.warm_iterations, args.seed+10000+start_id)
        print(f"start {start_id}: MPIF warm {warm_time:.2f}s", flush=True)
        warm_theta = deepcopy(warm.theta)
        warm_z = jnp.array(d.pack(warm_theta.params(as_list=True)[0]))
        warm_trace = panel_trace(warm, warm_time, args.stride)
        p, seconds = run_mif(payload, args.mif_iterations, args.seed+20000+start_id)
        save_fit("MPIF", panel_trace(p, seconds, args.stride), seconds)
        print(f"start {start_id}: MPIF full {seconds:.2f}s", flush=True)
        p, continuation_time = run_adam(warm_theta, args.seed+30000+start_id)
        trace = warm_trace + panel_trace(p, continuation_time, args.stride,
                                        args.warm_iterations, warm_time)[1:]
        save_fit("IFAD", trace, warm_time+continuation_time)
        print(f"start {start_id}: IFAD continuation {continuation_time:.2f}s", flush=True)

        for method_id, (name, fn) in enumerate(scores.items()):
            z = warm_z
            m, v, average = jnp.zeros_like(z), jnp.zeros_like(z), jnp.zeros_like(z)
            elapsed, status, updates = 0., "complete", 0
            trace = list(warm_trace)
            for i in range(args.max_updates):
                if elapsed >= continuation_time:
                    break
                begin = time.perf_counter()
                value, g = fn(z, jax.random.key(args.seed+1000000+10000*start_id+3000*method_id+i))
                jax.block_until_ready(g)
                record = {"start": start_id, "method": name, "iteration": i,
                          "fitting_loglik": float(value), "score_norm": float(jnp.linalg.norm(g)),
                          "accepted": False}
                objectives.append(record)
                if not np.isfinite(value) or not np.isfinite(g).all():
                    elapsed += time.perf_counter()-begin
                    status = "nonfinite_update"
                    break
                if name == "DS19":
                    gain = 1. if i < 30 else (i-29.)**(-.9)
                    average = (1-gain)*average+gain*g
                    g = average
                g *= jnp.minimum(1., 100./jnp.maximum(jnp.linalg.norm(g), 1e-12))
                m, v = .9*m+.1*g, .999*v+.001*g*g
                specific_rate = args.ds_learning_rate if name == "DS19" else args.ct_learning_rate
                base_rate = args.competitor_learning_rate if specific_rate is None else specific_rate
                rate = base_rate*(.1+.9*.5*(1+np.cos(np.pi*min(elapsed/continuation_time, 1.))))
                record["learning_rate"] = float(rate)
                candidate = z+rate*(m/(1-.9**(i+1)))/(jnp.sqrt(v/(1-.999**(i+1)))+1e-8)
                jax.block_until_ready(candidate)
                elapsed += time.perf_counter()-begin
                record["elapsed_seconds"] = elapsed
                if not np.isfinite(candidate).all():
                    status = "nonfinite_update"
                    break
                # Only estimates completed within the assigned time are eligible.
                if elapsed > continuation_time:
                    break
                z = candidate
                record["accepted"] = True
                updates += 1
                if updates % 10 == 0:
                    trace.append((args.warm_iterations+updates, warm_time+elapsed, np.asarray(z)))
            if trace[-1][0] != args.warm_iterations+updates:
                trace.append((args.warm_iterations+updates, warm_time+elapsed, np.asarray(z)))
            save_fit(name, trace, warm_time+elapsed, status)
            pd.DataFrame(objectives).to_csv(args.output/"fitting_objectives.csv", index=False)
            print(f"{start_id} {name}: {updates} updates, {elapsed:.2f}s, {status}", flush=True)

        for method_id, (name, (trace, seconds, status)) in enumerate(fits.items()):
            timing.append({"start": start_id, "method": name, "seconds": seconds, "status": status,
                           "warm_seconds": warm_time if name != "MPIF" else 0.,
                           "continuation_budget": continuation_time if name != "MPIF" else 0.})
            for i,t,z in trace:
                raw = np.asarray(evaluate(z, jax.random.split(jax.random.key(
                    args.seed+5000000+100000*start_id+10000*method_id+i), args.eval_reps)))
                ll = float((logsumexp(raw, axis=0)-np.log(args.eval_reps)).sum())
                weights = np.exp(raw-raw.max(0))
                se = float(np.linalg.norm(weights.std(0, ddof=1)/np.sqrt(args.eval_reps)/weights.mean(0)))
                rows.append({"model": "daphnia", "start": start_id, "method": name,
                             "iteration": i, "seconds": t, "loglik": ll, "mcse": se,
                             "final": i == trace[-1][0], "status": status, **dict(zip(d.NAMES, z))})
                replicates.extend({"start": start_id, "method": name, "iteration": i,
                    "replicate": r, "unit": unit, "loglik": raw[r,u]}
                    for r in range(args.eval_reps) for u,unit in enumerate(d.reference.UNITS))
            print(f"start {start_id+1} {name}: {ll:.3f} ({se:.3f}), {seconds:.2f}s", flush=True)
            pd.DataFrame(rows).to_csv(args.output/"checkpoints.csv", index=False)
            pd.DataFrame(timing).to_csv(args.output/"timings.csv", index=False)
            pd.DataFrame(replicates).to_csv(args.output/"evaluation_replicates.csv", index=False)
        write_json(args.output/"status.json", {"complete": start_id+1 == args.start_index+args.starts,
                                               "fits": len(timing), "purpose": args.purpose})


if __name__ == "__main__":
    main()
