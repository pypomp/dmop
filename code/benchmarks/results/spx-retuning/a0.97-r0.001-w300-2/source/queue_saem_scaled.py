"""Replace one superseded SAEM worker after its in-flight score fit is saved."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    old_unit = f"dmop-saem-ab-w{args.worker}.service"
    checkpoint = ROOT / "ditlevsen/results/saem_ab/cold" / f"start{8 + args.worker:03d}" / "score.json"
    def active():
        return subprocess.check_output(["systemctl", "--user", "show", old_unit,
                                        "--property=ActiveState", "--value"], text=True).strip() == "active"
    print(f"Waiting for the current score fit from {old_unit}: {checkpoint}", flush=True)
    while active() and not checkpoint.exists():
        time.sleep(5)
    if active():
        subprocess.run(["systemctl", "--user", "stop", old_unit], check=True)
    if active():
        raise RuntimeError(f"Old worker still active: {old_unit}")
    cores = set(range(4 * args.worker, 4 * args.worker + 4))
    os.sched_setaffinity(0, cores)
    command = [sys.executable, "-u", "-m", "ditlevsen.saem_ab", "--output",
               "ditlevsen/results/saem_ab_scaled", "--worker", str(args.worker), "--workers", "4"]
    record = dict(worker=args.worker, replaced_service=old_unit,
                  old_score_checkpoint_exists=checkpoint.exists(), cores=sorted(cores), command=command)
    (ROOT / "ditlevsen/results/saem_ab_scaled" / f"launch{args.worker}.json").write_text(
        json.dumps(record, indent=2) + "\n")
    print("Launching corrected worker", record, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
