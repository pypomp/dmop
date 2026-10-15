"""Choose a common Gaussian particle count using separate IFAD tuning starts.

The same count is then used for IFAD, IF2 and DS19. Compare 100 and 500 by
the summed median IFAD likelihood deficit across the two Gaussian examples,
after preferring fewer failed fits. Final benchmark starts do not select it.
"""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from queue_final import wait_for

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    # This serial study inherits the same CPU assignment as the matched runs.
    wait_for("matched-particles", timeout=16 * 3600)
    output = RESULTS / "matched-particle-tuning"
    output.mkdir(exist_ok=True)
    candidates = []
    for particles in (100, 500):
        failed, gap, seconds = 0, 0., 0.
        sources = []
        for model in ("linear", "oscillator"):
            folder = RESULTS / f"tuning-{model}-particles{particles}"
            command = [sys.executable, str(HERE / "run.py"), "--model", model,
                       "--output", str(folder), "--purpose", "pilot", "--starts", "4",
                       "--seed", "2026100901", "--particles", str(particles),
                       "--iterations", "300", "--if2-iterations", "600", "--warm", "100",
                       "--learning-rate", ".01", "--checkpoint-every", "25", "--methods", "IFAD"]
            if folder.exists():
                status = folder / "status.json"
                if not status.exists() or not json.loads(status.read_text()).get("complete"):
                    raise RuntimeError(f"Inspect incomplete tuning batch: {folder}")
            else:
                print("Launching", folder.name, flush=True)
                subprocess.run(command, cwd=HERE.parents[1], check=True)
            final = pd.read_csv(folder / "checkpoints.csv").loc[lambda x: x.final]
            if sorted(final.start) != list(range(4)) or set(final.method) != {"IFAD"}:
                raise ValueError(f"Invalid tuning records: {folder}")
            reference = json.loads((folder / "analytic_reference.json").read_text())["loglik"]
            failed += int(final.status.ne("complete").sum())
            gap += reference - float(final.loglik.median())
            seconds += float(pd.read_csv(folder / "timings.csv").seconds.median())
            sources.append(folder.name)
        candidates.append(dict(particles=particles, failed=failed, summed_median_deficit=gap,
                               summed_median_seconds=seconds, sources=sources))
    chosen = min(candidates, key=lambda x: (x["failed"], x["summed_median_deficit"], x["summed_median_seconds"]))
    selection = {"seed": 2026100901, "starts_per_model": 4, "candidates": candidates,
                 "chosen_particles": chosen["particles"],
                 "rule": "fewest failed IFAD tuning fits, then smallest sum of median analytic-likelihood deficits, then time",
                 "scope": "same chosen count for IFAD, IF2 and DS19 in both Gaussian examples"}
    temporary = output / "selection.tmp"
    temporary.write_text(json.dumps(selection, indent=2) + "\n")
    temporary.replace(output / "selection.json")
    (output / "status.json").write_text(json.dumps({"complete": True}) + "\n")
    print("Selected common particle count:", chosen["particles"], flush=True)


if __name__ == "__main__":
    main()
