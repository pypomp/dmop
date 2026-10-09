import json

import pandas as pd
import pytest

from report import METHODS, new_results


def make_result(root):
    folder = root / "final-spx-00"
    folder.mkdir()
    (folder / "configuration.json").write_text(json.dumps({"purpose": "final"}))
    (folder / "status.json").write_text(json.dumps({"complete": True}))
    frame = pd.DataFrame([{"start": start, "method": method, "seconds": 1.,
                           "final": True, "loglik": 100. + start, "mcse": .1,
                           "status": "nonfinite_update" if method == "DS19" else "complete"}
                          for start in range(2) for method in METHODS])
    frame.to_csv(folder / "checkpoints.csv", index=False)
    frame[["start", "method", "seconds"]].to_csv(folder / "timings.csv", index=False)
    return folder, frame


def test_report_keeps_failed_fits(tmp_path):
    make_result(tmp_path)
    final, _, _, _ = new_results(tmp_path, "spx", expected=2)
    assert len(final) == 8
    assert final.status.ne("complete").sum() == 2


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nan", "mcse", "time", "unfinished"])
def test_report_rejects_invalid_completed_batch(tmp_path, corruption):
    folder, frame = make_result(tmp_path)
    if corruption == "missing":
        frame = frame.iloc[1:]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif corruption == "nan":
        frame.loc[0, "loglik"] = float("nan")
    elif corruption == "mcse":
        frame.loc[0, "mcse"] = -1.
    elif corruption == "time":
        times = pd.read_csv(folder / "timings.csv")
        times.loc[0, "seconds"] = -1.
        times.to_csv(folder / "timings.csv", index=False)
    else:
        (folder / "status.json").write_text(json.dumps({"complete": False}))
    frame.to_csv(folder / "checkpoints.csv", index=False)
    with pytest.raises(ValueError):
        new_results(tmp_path, "spx", expected=2)
