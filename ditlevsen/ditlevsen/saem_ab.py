"""Paired Dhaka score/SAEM experiment from global and IF2 starting points.

Prepare once, then run disjoint workers. Both optimizers receive the same
particle count and continuation budget, and retain the last finite estimate
recorded before the deadline. Evaluation draws never select an iterate.
"""

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import jax
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .benchmark import make_starts
from .block_saem import fit as fit_saem
from .block_smc import fit_block_pseudo_score
from .data import load_dacca_data
from .model import normalize_parameters, parameter_bounds, physical_parameter_dict, project_parameters
from .warm_starts import load_starts_file

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(path)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output/"source"
    source.mkdir()
    hashes = {}
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, source/p.name)
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    starts_file = ROOT/"ditlevsen/results/reference/ifad097_comparable_post_if2.npz"
    warm = np.asarray([normalize_parameters(x) for x in load_starts_file(starts_file, args.starts)])
    cold = np.asarray([project_parameters(x) for x in make_starts(args.starts, 631409)])
    np.savez_compressed(args.output/"starts.npz", cold=cold, warm=warm)
    config = dict(starts=args.starts, particles=args.particles, budget_seconds=args.budget,
        saem_iterations=args.saem_iterations, saem_burnin=args.saem_burnin,
        mstep_iterations=25, eval_particles=args.eval_particles, eval_replicates=args.eval_reps,
        seed=631409, evaluation_seed=9641409, source_hashes=hashes,
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        pypomp_revision=subprocess.check_output(["git", "-C", str(ROOT.parent/"pypomp"),
            "rev-parse", "HEAD"], text=True).strip(),
        starts_file=str(starts_file.relative_to(ROOT)),
        starts_file_sha256=hashlib.sha256(starts_file.read_bytes()).hexdigest(),
        initialization_cost="precomputed inputs; IF2 fitting cost excluded from both warm arms",
        output_selection="last finite recorded estimate within the fitting budget; initial if none",
        common_transition="monthly Gaussian with 20 internal steps and relative floor 1e-12",
        comparison="optimizer packages: archived score settings versus pilot numerical-SAEM settings",
        score_settings={regime: score_settings(regime, args.particles, args.budget)
                        for regime in ("cold", "warm")})
    write_json(args.output/"protocol.json", config)


def score_settings(regime, particles, budget):
    directory = "ditlevsen_warm" if regime == "warm" else "ditlevsen_vanilla"
    archived = json.loads((ROOT/"corenflos/results/particle_increase_j1000_final_100"
                           /directory/"configuration.json").read_text())
    names = inspect.signature(fit_block_pseudo_score).parameters
    settings = {name: value for name, value in archived.items() if name in names}
    settings.update(particles=particles, maximum_elapsed_seconds=budget,
                    endpoint_relative_floor=archived["relative_floor"],
                    project_to_bounds=regime == "cold", compile_before_timing=True)
    return settings


def selected_score(result, initial, budget):
    eligible = (np.asarray(result.elapsed_trace) <= budget
                ) & np.isfinite(result.parameter_trace).all(axis=1) & np.isfinite(result.marginal_loglik_trace)
    indices = np.flatnonzero(eligible)
    index = int(indices[-1]) if len(indices) else -1
    return (result.parameter_trace[index] if index >= 0 else initial), index


def evaluate(theta, *, particles, replicates, seed):
    import pypomp as pp
    model = pp.models.dacca(nstep=20, dt=None)
    model.pfilter(theta=pp.PompParameters(physical_parameter_dict(theta)), J=particles,
                  reps=replicates, key=jax.random.key(seed))
    raw = np.asarray(model.results_history[-1].logLiks).reshape(-1)
    if len(raw) != replicates or not np.isfinite(raw).all():
        raise ValueError("Unresolved independent Euler evaluation")
    likelihood = float(logsumexp(raw)-np.log(replicates))
    weights = np.exp(raw-raw.max())
    se = float(weights.std(ddof=1)/np.sqrt(replicates)/weights.mean())
    return likelihood, se, raw


def run_pair(output, config, regime, index, initial, warmup, order):
    directory = output/regime/f"start{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"status.json").exists():
        if json.loads((directory/"status.json").read_text()).get("complete"):
            return
    data = load_dacca_data(20)
    estimates = {"initial": initial}
    seed = config["seed"]+1000003*20+10007*index
    for method in order:
        fit_file = directory/f"{method}.json"
        parameter_file = directory/f"{method}_parameters.npz"
        if fit_file.exists():
            estimates[method] = np.load(parameter_file)["selected"]
            continue
        started = time.perf_counter()
        if method == "score":
            settings = dict(config["score_settings"][regime], seed=seed)
            result = fit_block_pseudo_score(initial, data, **settings)
            selected, selected_index = selected_score(result, initial, config["budget_seconds"])
            arrays = {key: value for key, value in asdict(result).items() if isinstance(value, np.ndarray)}
            arrays["selected"] = selected
            np.savez_compressed(parameter_file, **arrays)
            info = dict(status=result.termination_reason, selected_index=selected_index,
                updates=max(selected_index, 0), updates_attempted=result.completed_updates,
                seconds=result.elapsed_seconds,
                preliminary_seconds=time.perf_counter()-started-result.elapsed_seconds,
                selected_seconds=float(result.elapsed_trace[selected_index]) if selected_index >= 0 else 0.)
        else:
            retained = [initial.copy()]
            preliminary = [None]
            def checkpoint(parameters, trace, compilation_seconds):
                retained[:] = list(parameters)
                preliminary[0] = compilation_seconds
                np.savez_compressed(parameter_file, parameters=parameters, selected=parameters[-1])
                trace.to_csv(directory/"saem_trace.csv", index=False)
            bounds = list(zip(*[np.asarray(x) for x in parameter_bounds()])) if regime == "cold" else None
            try:
                parameters, trace, compilation = fit_saem(initial, data,
                    particles=config["particles"], iterations=config["saem_iterations"],
                    burnin=config["saem_burnin"], seed=seed, maxiter=config["mstep_iterations"],
                    checkpoint=checkpoint, maximum_elapsed_seconds=config["budget_seconds"],
                    bounds=bounds, warmup_start=warmup)
                selected = parameters[-1]
                np.savez_compressed(parameter_file, parameters=parameters, selected=selected)
                info = dict(status=trace.attrs["termination_reason"], updates=len(trace),
                    seconds=trace.attrs["seconds"], compilation_seconds=compilation,
                    selected_seconds=float(trace.seconds.iloc[-1]) if len(trace) else 0.)
            except (ValueError, FloatingPointError) as error:
                selected = retained[-1]
                np.savez_compressed(parameter_file, parameters=np.asarray(retained), selected=selected)
                info = dict(status="failed", error=str(error), updates=len(retained)-1,
                    seconds=None if preliminary[0] is None else time.perf_counter()-started-preliminary[0],
                    total_seconds=time.perf_counter()-started, compilation_seconds=preliminary[0])
        estimates[method] = selected
        info.update(regime=regime, start=index, method=method,
                    cpu_affinity=sorted(os.sched_getaffinity(0)))
        write_json(fit_file, info)
        print(f"{regime} {index} {method}: {info}", flush=True)

    evaluations, draws = [], []
    for method in ("initial", "score", "saem"):
        ll, se, raw = evaluate(estimates[method], particles=config["eval_particles"],
            replicates=config["eval_replicates"], seed=config["evaluation_seed"]+10007*index
            +(0 if regime == "cold" else 1000003))
        evaluations.append(dict(regime=regime, start=index, method=method, loglik=ll, mcse=se))
        draws.extend(dict(method=method, replicate=r, loglik=float(v)) for r,v in enumerate(raw))
        pd.DataFrame(evaluations).to_csv(directory/"evaluation.csv", index=False)
        pd.DataFrame(draws).to_csv(directory/"evaluation_replicates.csv", index=False)
        print(f"{regime} {index} Euler {method}: {ll:.3f} ({se:.3f})", flush=True)
    write_json(directory/"status.json", dict(complete=True))


def run_worker(args):
    if args.wait_for:
        while True:
            try:
                if json.loads(args.wait_for.read_text())["complete"]:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(10)
    config = json.loads((args.output/"protocol.json").read_text())
    for name, digest in config["source_hashes"].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed after protocol preparation: {name}")
    starts = np.load(args.output/"starts.npz")
    # Compile full-shape score kernels before timed fits; warm and cold share
    # these kernels. The SAEM fitter separately records its preliminary call.
    warm_settings = dict(config["score_settings"]["warm"], iterations=1,
        maximum_elapsed_seconds=None, likelihood_guard_interval=1, seed=831409)
    warm_started = time.perf_counter()
    fit_block_pseudo_score(starts["warm"][0], load_dacca_data(20), **warm_settings)
    write_json(args.output/f"worker{args.worker}.json", dict(
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        score_compilation_seconds=time.perf_counter()-warm_started))
    for pair in range(args.worker, 2*config["starts"], args.workers):
        regime = "cold" if pair < config["starts"] else "warm"
        index = pair % config["starts"]
        order = ("score", "saem") if (pair//args.workers)%2 == 0 else ("saem", "score")
        run_pair(args.output, config, regime, index, starts[regime][index], starts["warm"][0], order)
    write_json(args.output/f"worker{args.worker}_status.json", dict(complete=True))
    from .saem_ab_report import generate
    generate(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--budget", type=float, default=850.)
    parser.add_argument("--saem-iterations", type=int, default=80)
    parser.add_argument("--saem-burnin", type=int, default=3)
    parser.add_argument("--eval-particles", type=int, default=5000)
    parser.add_argument("--eval-reps", type=int, default=36)
    args = parser.parse_args()
    if min(args.starts, args.particles, args.workers, args.saem_iterations, args.eval_particles) < 1 \
            or not 0 <= args.worker < args.workers or args.budget <= 0 \
            or not 1 <= args.saem_burnin <= args.saem_iterations or args.eval_reps < 2:
        parser.error("Invalid experiment counts")
    prepare(args) if args.prepare else run_worker(args)


if __name__ == "__main__":
    main()
