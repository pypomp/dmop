"""Sequential, resumable higher-particle runs of all four Dacca variants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARMS = {
    "corenflos_warm": (
        "corenflos.benchmark", "corenflos",
        "results/if2warm_ifad097_budget_j100_final_100",
    ),
    "ditlevsen_warm": (
        "ditlevsen.smc_benchmark", "ditlevsen",
        "results/block_smc_guided_j100_if2warm_ifad097_budget_final_100",
    ),
    "corenflos_vanilla": (
        "corenflos.benchmark", "corenflos",
        "results/corenflos_j100_eps025_final_100",
    ),
    "ditlevsen_vanilla": (
        "ditlevsen.smc_benchmark", "ditlevsen",
        "results/block_smc_guided_j100_final_100",
    ),
}


def configuration(arm: str, output: Path, particles: int, starts: int) -> dict:
    _, project, baseline = ARMS[arm]
    config = json.loads((ROOT / project / baseline / "configuration.json").read_text())
    config.update(
        output=str(output.resolve()), particles=particles, starts=starts,
        workers=1, worker_index=0, stages=["fit"],
        eval_particles=5000, eval_replicates=36,
        trace_eval_particles=5000, trace_eval_replicates=1,
    )
    if config.get("starts_file"):
        config["starts_file"] = str((ROOT / project / config["starts_file"]).resolve())
    return config


def command(arm: str, config: dict, stage: str, partitions: int, partition: int) -> list[str]:
    module_name, _, _ = ARMS[arm]
    parser = importlib.import_module(module_name).build_parser()
    config = dict(config, stages=[stage], workers=partitions, worker_index=partition)
    actions = {action.dest: action for action in parser._actions}
    arguments = []
    for key, value in config.items():
        if value is None:
            continue
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            arguments.append(action.option_strings[0 if value else 1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                arguments.append(action.option_strings[0])
        else:
            arguments.append(action.option_strings[0])
            arguments.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    # Round-trip validation catches stale reference settings before starting a job.
    parser.parse_args(arguments)
    return [sys.executable, "-u", "-m", module_name, *arguments]


def checkpoint(output: Path, arm: str, start: int) -> Path:
    filename = f"fit_start{start:03d}.npz" if arm.startswith("corenflos") else f"fit_r20_start{start:03d}.npz"
    return output / "checkpoints" / filename


def stage_complete(output: Path, arm: str, stage: str, assigned: range) -> bool:
    for start in assigned:
        path = checkpoint(output, arm, start)
        if not path.exists():
            return False
        if stage == "fit":
            continue
        prefix = "euler" if arm.startswith("corenflos") else "euler_r20"
        if stage == "final-eval":
            paths = [output / "evaluations" / f"{prefix}_start{start:03d}.json"]
        else:
            with np.load(path, allow_pickle=False) as fit:
                paths = [
                    output / "trace_evaluations" / f"{prefix}_start{start:03d}_iter{iteration:04d}.json"
                    if arm.startswith("corenflos") else
                    output / "trace_evaluations" / f"{prefix}_start{start:03d}_iter{iteration:03d}.json"
                    for iteration in range(len(fit["parameter_trace"]))
                ]
        for path in paths:
            if not path.exists():
                return False
            result = json.loads(path.read_text())
            if result.get("evaluation_status") != "ok":
                raise RuntimeError(f"failed evaluation requires inspection: {path}")
            replicates = 36 if stage == "final-eval" else 1
            if result.get("eval_particles") != 5000 or result.get("eval_replicates") != replicates:
                raise RuntimeError(f"wrong evaluation effort requires inspection: {path}")
    return True


def write_status(output: Path, **status) -> None:
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output / ".status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(output / "status.json")


def table_complete(output: Path, stage: str, assigned: range) -> bool:
    filename = {
        "fit": "fit_summary.csv", "final-eval": "final_evaluations.csv",
        "trace-eval": "optimization_euler_traces.csv",
    }[stage]
    path = output / filename
    if not path.exists():
        return False
    starts = pd.read_csv(path, usecols=["start"])["start"]
    return set(assigned).issubset(set(starts))


def audit(output: Path, configs: dict[str, dict]) -> dict:
    summaries = []
    paired = []
    baseline_warm = pd.read_csv(
        ROOT / "corenflos/results/ifad097_post_if2_reference/warm_start_evaluations.csv"
    ).set_index("start")["euler20_loglik"]
    for arm, config in configs.items():
        directory = output / arm
        _, project, baseline_name = ARMS[arm]
        baseline_directory = ROOT / project / baseline_name
        observed_config = json.loads((directory / "configuration.json").read_text())
        for key, expected in config.items():
            if key not in ("stages", "workers", "worker_index") and observed_config.get(key) != expected:
                raise AssertionError(f"{arm}: changed configuration: {key}")
        if not stage_complete(directory, arm, "trace-eval", range(config["starts"])):
            raise AssertionError(f"{arm}: incomplete trace evaluations")
        training = pd.read_csv(directory / "optimization_training_traces.csv")
        traces = pd.read_csv(directory / "optimization_euler_traces.csv")
        finals = pd.read_csv(directory / "final_evaluations.csv").set_index("start")
        fits = pd.read_csv(directory / "fit_summary.csv").set_index("start")
        baseline = pd.read_csv(baseline_directory / "final_evaluations.csv").set_index("start")
        if len(training) != len(traces) or len(finals) != config["starts"]:
            raise AssertionError(f"{arm}: incomplete consolidated tables")
        if not finals["evaluation_status"].eq("ok").all() or not traces["evaluation_status"].eq("ok").all():
            raise AssertionError(f"{arm}: unsuccessful Euler evaluations")
        for table, replicates in ((finals, 36), (traces, 1)):
            if not table["eval_particles"].eq(5000).all() or not table["eval_replicates"].eq(replicates).all():
                raise AssertionError(f"{arm}: changed evaluation effort")
        for start in range(config["starts"]):
            with np.load(checkpoint(directory, arm, start), allow_pickle=False) as new:
                with np.load(checkpoint(baseline_directory, arm, start), allow_pickle=False) as old:
                    np.testing.assert_array_equal(new["start"], old["start"])
                    np.testing.assert_allclose(new["parameter_trace"][0], old["parameter_trace"][0], rtol=0, atol=1e-10)
                iterations = np.arange(len(new["parameter_trace"]))
            for table in (training, traces):
                np.testing.assert_array_equal(
                    np.sort(table.loc[table["start"].eq(start), "iteration"].to_numpy()), iterations,
                )
        difference = finals["euler20_loglik"] - baseline.loc[finals.index, "euler20_loglik"]
        summaries.append({
            "variant": arm, "particles": config["particles"], "starts": len(finals),
            "median_euler_loglik": float(finals["euler20_loglik"].median()),
            "baseline_j100_median": float(baseline.loc[finals.index, "euler20_loglik"].median()),
            "median_paired_change_from_j100": float(difference.median()),
            "improving_over_j100": int(difference.gt(0).sum()),
            "median_updates": float(fits["updates_completed"].median()),
            "median_paired_change_from_if2": (
                float((finals["euler20_loglik"] - baseline_warm.loc[finals.index]).median())
                if arm.endswith("warm") else None
            ),
        })
        paired.append(pd.DataFrame({
            "variant": arm, "start": finals.index, "particles": config["particles"],
            "euler20_loglik": finals["euler20_loglik"].to_numpy(),
            "j100_euler20_loglik": baseline.loc[finals.index, "euler20_loglik"].to_numpy(),
            "paired_change": difference.to_numpy(),
            "updates_completed": fits.loc[finals.index, "updates_completed"].to_numpy(),
        }))
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(output / "paired_particle_comparison.csv", index=False)
    return {"variants": summaries}


def figures(output: Path) -> None:
    from . import warm_plots

    args = warm_plots.build_parser().parse_args([])
    args.corenflos_warm = output / "corenflos_warm"
    args.ditlevsen_warm = output / "ditlevsen_warm"
    args.corenflos_cold = output / "corenflos_vanilla"
    args.ditlevsen_cold = output / "ditlevsen_vanilla"
    args.warm_start = ROOT / "corenflos/results/ifad097_post_if2_reference"
    args.reference = ROOT / "ditlevsen/results/reference"
    args.starts_file = args.reference / "ifad097_comparable_post_if2.npz"
    args.output = output / "figures"
    args.output.mkdir(exist_ok=True)
    warm_plots.plot_likelihood(warm_plots._likelihood_frame(args), args.output)
    warm_plots.plot_parameters(warm_plots._parameter_frame(args), args.output)
    warm_plots.plot_optimization(warm_plots._optimization_frame(args), args.output)
    mismatch = warm_plots._objective_mismatch_frame(args)
    mismatch.to_csv(output / "objective_mismatch_if2warm_r20.csv", index=False)
    warm_plots.plot_objective_mismatch(mismatch, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        args.starts = args.partitions = 1
    if args.particles <= 100 or not 1 <= args.starts <= 100 or not 1 <= args.partitions <= args.starts:
        parser.error("require particles > 100, 1 <= partitions <= starts <= 100")
    default_name = "preflight" if args.preflight else f"final_{args.starts}"
    output = (
        args.output or Path(f"results/particle_increase_j{args.particles}_{default_name}")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    configs = {
        arm: configuration(arm, output / arm, args.particles, args.starts) for arm in ARMS
    }
    if args.preflight:
        for arm, config in configs.items():
            config.update(starts=1, maximum_elapsed_seconds=90.0)
            config["maximum_updates" if arm.startswith("corenflos") else "iterations"] = 2
    manifest = {
        "particles": args.particles, "starts": args.starts,
        "partitions": args.partitions, "preflight": args.preflight, "configs": configs,
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("refusing to change an existing experiment's manifest")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(str(ROOT / name) for name in ("corenflos", "ditlevsen", "../pypomp"))
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    environment["JAX_PLATFORMS"] = "cuda"
    try:
        # Ten interleaved batches expose results for every variant early. Only
        # one fitting/evaluation subprocess ever owns the GPU at a time.
        phases = [("fit", "final-eval")]
        if not args.preflight:
            phases.append(("trace-eval",))
        for stages in phases:
            for partition in range(args.partitions):
                for arm, config in configs.items():
                    directory = output / arm
                    assigned = range(partition, config["starts"], args.partitions)
                    for stage in stages:
                        if stage_complete(directory, arm, stage, assigned) and table_complete(directory, stage, assigned):
                            print(f"already complete: {arm} partition={partition} {stage}", flush=True)
                            continue
                        write_status(output, state="running", variant=arm, partition=partition, stage=stage)
                        print(f"starting: {arm} partition={partition} {stage}", flush=True)
                        subprocess.run(
                            command(arm, config, stage, args.partitions, partition),
                            cwd=ROOT / ARMS[arm][1], env=environment, check=True,
                        )
                        if not stage_complete(directory, arm, stage, assigned):
                            raise RuntimeError(f"incomplete stage: {arm} {stage} partition={partition}")
        if args.preflight:
            write_status(output, state="preflight-complete")
        else:
            report = audit(output, configs)
            figures(output)
            write_status(output, state="complete", report=report)
    except Exception as error:
        write_status(output, state="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
