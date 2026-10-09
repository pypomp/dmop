"""Gaussian comparisons at equal particle counts for DS19, IFAD and IF2.

Run both 100 and 500 particles on the same 20 starts and CPU assignment.
Keep both configurations in the repository. Separate tuning starts select
the common count for the manuscript; see tune_particles.py. CTDD21 is unchanged.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    protocol = RESULTS / "matched-particles"
    protocol.mkdir(exist_ok=True)
    jobs = []
    for particles in (100, 500):
        for model in ("linear", "oscillator"):
            for group, methods in (("if", ["IF2", "IFAD"]), ("ds", ["DS19"])):
                name = f"matched-{model}-j{particles}-{group}"
                command = [sys.executable, str(HERE / "run.py"), "--model", model,
                           "--output", str(RESULTS / name), "--purpose", "final",
                           "--starts", "20", "--seed", "631450", "--particles", str(particles),
                           "--ds-particles", str(particles), "--ds-update", "saem",
                           "--iterations", "80" if group == "ds" else "300",
                           "--if2-iterations", "600", "--warm", "100",
                           "--learning-rate", ".01", "--checkpoint-every", "5" if group == "ds" else "25",
                           "--methods", *methods]
                jobs.append({"name": name, "command": command})
    plan = {"utc": datetime.now(timezone.utc).isoformat(), "cores": sorted(os.sched_getaffinity(0)),
            "jobs": jobs, "reporting": "separate tuning starts select the common manuscript count; retain all runs in the repository",
            "fixed_settings": "same 20 starts and method-specific iteration schedules at both counts"}
    plan_path = protocol / "configuration.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    for index, job in enumerate(jobs):
        output = RESULTS / job["name"]
        if output.exists():
            status = output / "status.json"
            if status.exists() and json.loads(status.read_text()).get("complete"):
                print("Already complete:", job["name"], flush=True)
                continue
            raise RuntimeError(f"Inspect interrupted matched-particle batch before recovery: {output}")
        print("Launching", job["name"], flush=True)
        subprocess.run(job["command"], cwd=HERE.parents[1], check=True)
        (protocol / "status.json").write_text(json.dumps({"complete": index + 1 == len(jobs),
            "batches": index + 1, "expected_batches": len(jobs)}) + "\n")
    print("Both particle counts complete; use the separate tuning selection for the manuscript", flush=True)


if __name__ == "__main__":
    main()
