import json

import pandas as pd
import pytest

from report import METHODS, new_results


def make_result(root):
    folder = root / "final-spx-00"
    folder.mkdir()
    (folder / "configuration.json").write_text(json.dumps({"purpose": "final"}))
    (folder / "status.json").write_text(json.dumps({"complete": True}))
    pd.DataFrame({"start": [0, 1], "theta": [.2, .4]}).to_csv(folder / "starts.csv", index=False)
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


def test_gaussian_report_uses_one_common_particle_count(tmp_path):
    for name, keep, count in (("final-linear", METHODS, 500),
                               ("matched-linear-j500-if", ["IF2", "IFAD"], 500),
                               ("matched-linear-j500-ds", ["DS19"], 500)):
        original, frame = make_result(tmp_path)
        folder = tmp_path / name
        original.rename(folder)
        frame = frame.loc[frame.method.isin(keep)].copy()
        if name.startswith("matched-"):
            frame["loglik"] += .25
        frame.to_csv(folder / "checkpoints.csv", index=False)
        frame[["start", "method", "seconds"]].to_csv(folder / "timings.csv", index=False)
        (folder / "configuration.json").write_text(json.dumps({"purpose": "final", "particles": count,
            "ds_particles": count, "ds_update": "saem" if name.endswith("-ds") else "score"}))
        (folder / "analytic_reference.json").write_text(json.dumps({"converged": True, "loglik": 102.}))
    final, progress, _, _ = new_results(tmp_path, "linear", expected=2, gaussian_particles=500)
    assert len(final) == 8
    assert final.loc[final.method.eq("CTDD21"), "loglik"].min() == 100.
    assert final.loc[final.method.eq("IFAD"), "loglik"].min() == 100.25
    assert (progress.maximum >= progress["median"]).all()
    assert (progress.q10 <= progress["median"]).all()
    starts_path = tmp_path / "matched-linear-j500-ds" / "starts.csv"
    starts = pd.read_csv(starts_path)
    starts.loc[0, "theta"] += .01
    starts.to_csv(starts_path, index=False)
    with pytest.raises(ValueError, match="Starting vectors disagree"):
        new_results(tmp_path, "linear", expected=2, gaussian_particles=500)
    starts.loc[0, "theta"] = .2
    starts.to_csv(starts_path, index=False)
    path = tmp_path / "matched-linear-j500-ds" / "configuration.json"
    config = json.loads(path.read_text())
    config["ds_particles"] = 100
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unequal particle count"):
        new_results(tmp_path, "linear", expected=2, gaussian_particles=500)


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
