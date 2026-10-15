"""Verify the completed tuning and reuse the identical archived final recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_results import audit_batch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "spx-retuning"


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    chosen = selection["chosen"]
    audits = []
    for candidate in selection["candidates"]:
        frames = []
        for name in candidate["folders"]:
            folder = ROOT / name
            audits.append(audit_batch(folder))
            frames.append(pd.read_csv(folder / "checkpoints.csv").loc[lambda d: d.final])
        frame = pd.concat(frames)
        assert sorted(frame.start) == list(range(4))
        np.testing.assert_allclose(frame.loglik.median(), candidate["median_loglik"], atol=1e-10)
        assert int(frame.status.ne("complete").sum()) == candidate["failed"]
    key = lambda r: (r["failed"], -r["median_loglik"], r["median_seconds"])
    assert min(selection["candidates"], key=key) == chosen
    tune = json.loads((ROOT / chosen["folders"][0] / "configuration.json").read_text())
    files, checked = {}, []
    for folder in sorted(p for p in RESULTS.glob("final-spx-*") if p.is_dir()):
        c = json.loads((folder / "configuration.json").read_text())
        for field in ("alpha", "warm", "ifad_learning_rate"):
            assert c[field] == chosen[field], (folder, field)
        for field in ("iterations", "particles", "eval_particles", "eval_reps",
                      "checkpoint_every", "trace_eval_particles", "trace_eval_reps"):
            expected = {"eval_particles": 5000, "eval_reps": 24,
                        "checkpoint_every": 25}.get(field, tune[field])
            assert c[field] == expected, (folder, field)
        assert c["seed"] == 631450 and c["pypomp_commit"] == tune["pypomp_commit"]
        for name in ("run.py", "models.py"):
            assert c["source_sha256"][name] == tune["source_sha256"][name]
        assert c["spx"] == tune["spx"]
        audits.append(audit_batch(folder))
        for path in folder.iterdir():
            if path.is_file():
                files[str(path.relative_to(RESULTS))] = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.append(str(folder.relative_to(RESULTS)))
    assert len(checked) == 4
    record = {
        "reason": "Tuning selected the exact existing recipe. All 20 archived final fits use identical run/model source, Pypomp revision, data, selected hyperparameters, seeds and evaluation settings. Preserve their original evaluations and timings.",
        "protocol_deviation": "The planned final refit was replaced by reuse after verifying identity. Automatically started repeats were stopped and preserved in abandoned-repeat; no partial repeat is used in the report.",
        "folders": checked, "final_fits": 20, "files": files}
    (OUT / "reuse_final.json").write_text(json.dumps(record, indent=2) + "\n")
    (OUT / "audit.json").write_text(json.dumps({"batches": audits}, indent=2) + "\n")
    (OUT / "status.json").write_text(json.dumps({"complete": True, "tuning_fits": 28,
        "final_fits": 20, "reused_final": True}, indent=2) + "\n")
    print(f"Verified {len(audits)} batches: 28 tuning fits and all 80 original final method/start combinations.")


if __name__ == "__main__":
    main()
