"""Build the complete cross-model report after all declared batches finish.

Run as a persistent service. Missing/incomplete status files mean waiting;
invalid completed data cause the reporting command to fail for inspection.
This script never modifies the manuscript.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for", nargs="+", required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 24 * 3600
    previous = None
    while True:
        pending = []
        for name in args.wait_for:
            status = HERE / "results" / name / "status.json"
            try:
                complete = json.loads(status.read_text())["complete"]
            except (FileNotFoundError, json.JSONDecodeError):
                complete = False
            if not complete:
                pending.append(name)
        if not pending:
            break
        if pending != previous:
            print("Report waiting for:", ", ".join(pending), flush=True)
            previous = pending
        if time.monotonic() > deadline:
            raise TimeoutError(f"Incomplete batches: {pending}")
        time.sleep(20)
    subprocess.run([sys.executable, str(HERE / "report.py")], cwd=HERE.parents[1], check=True)
    print("Complete report saved to imgs/benchmarks; manuscript review still required", flush=True)


if __name__ == "__main__":
    main()
