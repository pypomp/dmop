"""Tune IFAD on independent starts, then refit all 20 publication starts.

Run from the repo root in the documented Pypomp environment. Four CPU workers
use disjoint pairs of performance cores. Existing complete outputs are reused;
incomplete outputs require inspection instead of being silently overwritten.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "spx-retuning"
SEED = 2026101501


def save(name, data):
    path = OUT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def fit(worker, recipe, *, final=False):
    count, first = (5, 5 * worker) if final else (1, worker)
    folder = OUT / (f"final-{first:02d}" if final else f"{recipe['name']}-{worker}")
    if folder.exists():
        status = folder / "status.json"
        if not status.exists() or not json.loads(status.read_text()).get("complete"):
            raise RuntimeError(f"Inspect incomplete run: {folder}")
        config = json.loads((folder / "configuration.json").read_text())
        for key in ("alpha", "warm", "ifad_learning_rate"):
            if config[key] != recipe[key]:
                raise ValueError(f"Changed recipe for {folder}: {key}")
        return folder
    command = ["taskset", "-c", f"{4*worker}-{4*worker+3}", sys.executable,
               str(HERE / "run.py"), "--model", "spx", "--methods", "IFAD",
               "--output", str(folder), "--purpose", "final" if final else "pilot",
               "--starts", str(count), "--start-index", str(first),
               "--seed", str(631450 if final else SEED), "--particles", "500",
               "--iterations", "300", "--warm", str(recipe["warm"]),
               "--alpha", str(recipe["alpha"]),
               "--ifad-learning-rate", str(recipe["ifad_learning_rate"]),
               "--checkpoint-every", "25" if final else "100",
               "--eval-particles", "5000" if final else "2000",
               "--eval-reps", "24" if final else "8"]
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="4",
               PYTHONPATH="code/benchmarks:corenflos:ditlevsen:../pypomp",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with (OUT / (folder.name + ".log")).open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return folder


def summarize(recipe):
    folders = [OUT / f"{recipe['name']}-{i}" for i in range(4)]
    frames = [pd.read_csv(p / "checkpoints.csv").loc[lambda x: x.final] for p in folders]
    data = pd.concat(frames)
    assert sorted(data.start) == list(range(4)) and set(data.method) == {"IFAD"}
    return {**recipe, "median_loglik": float(data.loglik.median()),
            "failed": int(data.status.ne("complete").sum()),
            "median_seconds": float(data.seconds.median()),
            "folders": [str(p.relative_to(ROOT)) for p in folders]}


def main():
    OUT.mkdir(exist_ok=True)
    candidates = [dict(name=f"a{a:g}-r{r:g}-w100", alpha=a,
                       ifad_learning_rate=r, warm=100)
                  for a in (.97, .995, 1.) for r in (.001, .01)]
    protocol = {"seed": SEED, "starts": 4, "candidates": candidates,
                "second_stage": "compare winning alpha/rate at warm=100 and warm=300",
                "selection": "fewest failures, then highest median independent final likelihood, then shortest median time",
                "final": "20 original starts, all retained; 24 independent PF evaluations with J=5000",
                "learning_rate_schedule": "existing cosine decay to 0.1 times initial rate",
                "run_sha256": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()}
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed")
    save("protocol.json", protocol)
    def tune_worker(worker):
        for recipe in candidates:
            fit(worker, recipe)
            print(f"complete: {recipe['name']} start {worker}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(tune_worker, range(4)))
        results = [summarize(recipe) for recipe in candidates]
        key = lambda r: (r["failed"], -r["median_loglik"], r["median_seconds"])
        best = min(results, key=key)
        longer = {k: best[k] for k in ("name", "alpha", "ifad_learning_rate", "warm")}
        longer.update(name=longer["name"].replace("w100", "w300"), warm=300)
        list(pool.map(lambda i: fit(i, longer), range(4)))
        results.append(summarize(longer))
        chosen = min(results, key=key)
        save("selection.json", {"candidates": results, "chosen": chosen, "protocol": protocol})
        print("selected:", chosen, flush=True)
        list(pool.map(lambda i: fit(i, chosen, final=True), range(4)))
    save("status.json", {"complete": True, "final_fits": 20, "tuning_fits": 28})


if __name__ == "__main__":
    main()
