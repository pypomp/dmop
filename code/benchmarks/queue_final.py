"""Run a final batch after its tuning runs and assigned CPU cores are free.

This scheduler only reads declared predecessor directories. It does not stop
other processes or overwrite experiments. Tuning chooses the rate with the
fewest failed fits, then the highest median independently evaluated likelihood.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def wait_for(name, timeout=8*3600):
    begin = time.monotonic()
    path = RESULTS/name/"status.json"
    while time.monotonic()-begin < timeout:
        try:
            if json.loads(path.read_text())["complete"]:
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(10)
    raise TimeoutError(f"Predecessor did not finish: {name}")


def choose_rate(folders, method, field):
    candidates = []
    for folder in folders:
        wait_for(folder)
        config = json.loads((RESULTS/folder/"configuration.json").read_text())
        if config["purpose"] != "pilot":
            raise ValueError("Only separate pilot starts may tune settings")
        frame = pd.read_csv(RESULTS/folder/"checkpoints.csv")
        frame = frame.loc[frame.final & frame.method.eq(method)]
        if sorted(frame.start) != list(range(4)):
            raise ValueError(f"Expected four tuning starts: {folder}/{method}")
        if not all(math.isfinite(x) for x in frame.loglik):
            raise ValueError(f"Unresolved tuning likelihood evaluation: {folder}/{method}")
        rate = config[field] if field in config else config["learning_rate"]
        candidates.append({"source": folder, "rate": rate,
                           "failed": int(frame.status.ne("complete").sum()),
                           "median_loglik": float(frame.loglik.median())})
    chosen = min(candidates, key=lambda x: (x["failed"], -x["median_loglik"], x["rate"]))
    return chosen["rate"], {"method": method, "candidates": candidates, "chosen": chosen}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "oscillator", "spx", "daphnia"], required=True)
    parser.add_argument("--saem", action="store_true", help="Twenty-start Gaussian SAEM comparison")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--cores", required=True, help="Comma-separated CPU identifiers")
    parser.add_argument("--wait-for", nargs="*", default=[])
    args = parser.parse_args()
    os.sched_setaffinity(0, {int(x) for x in args.cores.split(",")})
    for folder in args.wait_for:
        wait_for(folder)
    selections = []
    if args.saem and (args.model not in ("linear", "oscillator") or args.start_index != 0):
        parser.error("SAEM batch must use a Gaussian model and start at zero")
    name = f"final-{args.model}-saem" if args.saem else f"final-{args.model}-{args.start_index:02d}"
    output = RESULTS/name
    if output.exists():
        raise FileExistsError(output)
    command = [sys.executable, str(HERE/("run_daphnia.py" if args.model == "daphnia" else "run.py")),
               "--output", str(output), "--purpose", "final", "--starts", "20" if args.saem else "5",
               "--start-index", str(args.start_index)]
    if args.model == "daphnia":
        folders = ["tuning-daphnia-cpu", "tuning-daphnia-cpu-lr0001"]
        for method, flag in (("DS19", "--ds-learning-rate"), ("CTDD21", "--ct-learning-rate")):
            rate, selection = choose_rate(folders, method, "competitor_learning_rate")
            command.extend([flag, str(rate)])
            selections.append(selection)
    else:
        command.extend(["--model", args.model, "--iterations", "80" if args.saem else "300", "--if2-iterations", "600",
                        "--warm", "100", "--learning-rate", ".01", "--checkpoint-every", "25"])
        if args.saem:
            command.extend(["--methods", "DS19", "--ds-update", "saem", "--checkpoint-every", "5"])
        if args.model == "spx":
            rate, selection = choose_rate(["tuning-spx", "tuning-spx-ifad-lr0001"], "IFAD", "learning_rate")
            command.extend(["--ifad-learning-rate", str(rate)])
            selections.append(selection)
        elif args.model == "oscillator":
            wait_for("tuning-oscillator")
    plan = {"command": command, "cores": sorted(os.sched_getaffinity(0)),
            "selection_rule": "fewest failed tuning fits, then highest median independent likelihood",
            "selections": selections}
    (RESULTS/f"{name}-launch.json").write_text(json.dumps(plan, indent=2)+"\n")
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    print("Launching", name, "on", plan["cores"], flush=True)
    subprocess.run(command, cwd=HERE.parents[1], env=env, check=True)


if __name__ == "__main__":
    main()
