"""Finish interrupted batches without selecting fits by their likelihood.

Retain every fully evaluated start, including failed optimizations. Rerun
the unfinished suffix with the original seeds and settings. Archive all
original files before merging; record the two source/configuration records.
Run this command in a persistent service, with no other writer to the batch.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = ("checkpoints.csv", "timings.csv", "evaluation_replicates.csv", "fitting_objectives.csv")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def complete_prefix(folder, config, methods):
    """Count whole starts, retaining unsuccessful fits with final evaluations."""
    first = config.get("start_index", 0)
    frame = pd.read_csv(folder / "checkpoints.csv")
    timings = pd.read_csv(folder / "timings.csv")
    completed = []
    gap = False
    for start in range(first, first + config["starts"]):
        rows = frame.loc[frame.start.eq(start) & frame.final]
        times = timings.loc[timings.start.eq(start)]
        good = sorted(rows.method) == sorted(methods) and sorted(times.method) == sorted(methods)
        if good:
            if gap:
                raise ValueError("Recovery requires an unfinished suffix, not missing interior starts")
            completed.append(start)
        else:
            gap = True
    return completed


def command_for(config, model, output, first, count):
    runner = HERE / ("run_daphnia.py" if model == "daphnia" else "run.py")
    command = [sys.executable, str(runner), "--output", str(output),
               "--start-index", str(first), "--starts", str(count)]
    if model != "daphnia":
        command += ["--model", model]
    fields = ("seed", "particles", "ds_particles", "ct_particles", "learning_rate",
              "eval_particles", "eval_reps", "trace_eval_particles", "trace_eval_reps", "purpose")
    fields += (("warm_iterations", "mif_iterations", "adam_iterations", "competitor_learning_rate",
                "ds_learning_rate", "ct_learning_rate", "stride", "max_updates") if model == "daphnia"
               else ("iterations", "if2_iterations", "warm", "ds_update", "ifad_learning_rate",
                     "alpha", "rw_sd", "checkpoint_every"))
    for field in fields:
        if config.get(field) is not None:
            command += ["--" + field.replace("_", "-"), str(config[field])]
    if model != "daphnia":
        command += ["--methods", *config["methods"]]
    return command


def merge(folder, suffix, archive, completed, methods):
    config = json.loads((archive / "configuration.json").read_text())
    first = config.get("start_index", 0)
    expected = list(range(first, first + config["starts"]))
    frames = {}
    for name in TABLES:
        pieces = []
        for source, keep in ((archive, completed), (suffix, sorted(set(expected) - set(completed)))):
            path = source / name
            if path.exists():
                data = pd.read_csv(path)
                pieces.append(data.loc[data.start.isin(keep)])
        if pieces:
            frames[name] = pd.concat(pieces, ignore_index=True)
    final = frames["checkpoints.csv"].loc[lambda f: f.final]
    timing = frames["timings.csv"]
    expected_keys = {(start, method) for start in expected for method in methods}
    for frame in (final, timing):
        keys = list(zip(frame.start, frame.method))
        if set(keys) != expected_keys or len(keys) != len(expected_keys):
            raise ValueError("Recovery has missing or duplicate final fits")
    for name, frame in frames.items():
        temporary = folder / (name + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(folder / name)
    for pattern in ("parameters_*.csv", "fit_*.json"):
        for path in suffix.glob(pattern):
            shutil.copy2(path, folder / path.name)
    manifest = {
        "rule": "retain completely evaluated starts; rerun the interrupted suffix, regardless of likelihood",
        "completed_starts_retained": completed,
        "rerun_starts": sorted(set(expected) - set(completed)),
        "original_archive": str(archive), "rerun": str(suffix),
        "original_configuration_sha256": hashlib.sha256((archive / "configuration.json").read_bytes()).hexdigest(),
        "rerun_configuration_sha256": hashlib.sha256((suffix / "configuration.json").read_bytes()).hexdigest(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(folder / "recovery.json", manifest)
    config["recovery"] = manifest
    write_json(folder / "configuration.json", config)
    write_json(folder / "status.json", {"complete": True, "fits": len(final),
               "failed": int(final.status.ne("complete").sum()), "purpose": config["purpose"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--model", choices=["spx", "daphnia"], required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    config = json.loads((folder / "configuration.json").read_text())
    if json.loads((folder / "status.json").read_text())["complete"]:
        print("Already complete:", folder, flush=True)
        return
    methods = ["MPIF", "IFAD", "DS19", "CTDD21"] if args.model == "daphnia" else config["methods"]
    completed = complete_prefix(folder, config, methods)
    if not completed or len(completed) == config["starts"]:
        raise ValueError("Expected a nonempty completed prefix and unfinished suffix")
    archive = folder.parent / "interrupted" / folder.name
    if not archive.exists():
        shutil.copytree(folder, archive)
    suffix = folder.parent / ("recovery-" + folder.name)
    command = command_for(config, args.model, suffix, completed[-1] + 1, config["starts"] - len(completed))
    plan = {"command": command, "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "completed_starts_retained": completed, "original_archive": str(archive),
            "utc": datetime.now(timezone.utc).isoformat()}
    write_json(folder / "recovery_plan.json", plan)
    if suffix.exists():
        if not (suffix / "status.json").exists() or not json.loads((suffix / "status.json").read_text())["complete"]:
            raise ValueError(f"Partial recovery needs inspection before restarting: {suffix}")
    else:
        print("Recovering", folder.name, "with", command, flush=True)
        subprocess.run(command, check=True, cwd=HERE.parents[1])
    merge(folder, suffix, archive, completed, methods)
    print("Recovery complete:", folder.name, flush=True)


if __name__ == "__main__":
    main()
