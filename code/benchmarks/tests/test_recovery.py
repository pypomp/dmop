import json

import pandas as pd
import pytest

from recover import complete_prefix, merge, command_for


def batch(folder, starts, methods=("IFAD", "DS19"), failed=False):
    folder.mkdir()
    rows = [{"start": s, "method": m, "final": True, "loglik": -100. + s,
             "status": "nonfinite_update" if failed and m == "DS19" else "complete"}
            for s in starts for m in methods]
    pd.DataFrame(rows).to_csv(folder / "checkpoints.csv", index=False)
    pd.DataFrame(rows).drop(columns=["final", "loglik"]).assign(seconds=10.).to_csv(
        folder / "timings.csv", index=False)
    (folder / "configuration.json").write_text(json.dumps({"starts": 3, "purpose": "final"}))


def test_recovery_retains_failed_completed_starts_and_replaces_partial_start(tmp_path):
    archive, suffix, output = (tmp_path / n for n in ("original", "rerun", "output"))
    batch(archive, [0], failed=True)
    partial = pd.read_csv(archive / "checkpoints.csv")
    partial = pd.concat([partial, pd.DataFrame([{"start": 1, "method": "IFAD", "final": True,
                                                "loglik": 999., "status": "complete"}])])
    partial.to_csv(archive / "checkpoints.csv", index=False)
    config = json.loads((archive / "configuration.json").read_text())
    assert complete_prefix(archive, config, ["IFAD", "DS19"]) == [0]
    batch(suffix, [1, 2])
    output.mkdir()
    merge(output, suffix, archive, [0], ["IFAD", "DS19"])
    result = pd.read_csv(output / "checkpoints.csv")
    assert len(result) == 6
    assert result.loc[result.start.eq(0), "status"].tolist() == ["complete", "nonfinite_update"]
    assert result.loglik.max() == -98.
    assert json.loads((output / "status.json").read_text())["failed"] == 1
    assert pd.read_csv(archive / "checkpoints.csv").loglik.max() == 999.


def test_recovery_rejects_missing_method(tmp_path):
    archive, suffix, output = (tmp_path / n for n in ("original", "rerun", "output"))
    batch(archive, [0])
    batch(suffix, [1, 2], methods=["IFAD"])
    output.mkdir()
    with pytest.raises(ValueError, match="missing or duplicate"):
        merge(output, suffix, archive, [0], ["IFAD", "DS19"])
    assert not (output / "status.json").exists()


def test_recovery_preserves_tuning_and_evaluation_settings(tmp_path):
    config = {"seed": 42, "competitor_learning_rate": .001, "learning_rate": .01,
              "eval_particles": 2000, "eval_reps": 10, "purpose": "pilot"}
    command = command_for(config, "daphnia", tmp_path, 1, 3)
    for name, value in (("seed", "42"), ("competitor-learning-rate", "0.001"),
                        ("eval-particles", "2000"), ("eval-reps", "10"), ("start-index", "1")):
        assert command[command.index("--" + name) + 1] == value
