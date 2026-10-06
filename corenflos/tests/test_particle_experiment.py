import importlib
import json

import numpy as np
import pandas as pd
import pytest

from corenflos import particle_experiment as experiment


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_particle_configuration_preserves_baseline_tuning(arm, tmp_path):
    _, project, reference = experiment.ARMS[arm]
    baseline = json.loads(
        (experiment.ROOT / project / reference / "configuration.json").read_text()
    )
    config = experiment.configuration(arm, tmp_path / arm, 1000, 100)
    assert config["particles"] == 1000
    assert config["starts"] == 100
    assert config["workers"] == 1
    assert config["output"] == str(tmp_path / arm)
    changed = {"output", "particles", "workers", "worker_index", "starts_file"}
    for key, value in baseline.items():
        if key not in changed:
            assert config[key] == value
    if arm.endswith("warm"):
        assert config["starts_file"] == str(
            experiment.ROOT / "ditlevsen/results/reference/ifad097_comparable_post_if2.npz"
        )
        assert config["elapsed_time_offset_seconds"] == 92.16042757034302
    else:
        assert not config.get("starts_file")


@pytest.mark.parametrize("arm", experiment.ARMS)
@pytest.mark.parametrize("stage", ["fit", "final-eval", "trace-eval"])
def test_particle_commands_round_trip_every_variant_and_stage(arm, stage, tmp_path):
    config = experiment.configuration(arm, tmp_path / arm, 1000, 100)
    original = dict(config)
    cmd = experiment.command(arm, config, stage, partitions=10, partition=3)
    parser = importlib.import_module(experiment.ARMS[arm][0]).build_parser()
    args = parser.parse_args(cmd[4:])
    assert args.stages == [stage]
    assert args.particles == 1000
    assert args.workers == 10 and args.worker_index == 3
    assert args.eval_particles == 5000 and args.eval_replicates == 36
    assert args.trace_eval_particles == 5000 and args.trace_eval_replicates == 1
    if arm.startswith("corenflos"):
        assert args.change_seed is True
    else:
        assert args.trace_every_update is True
        assert args.likelihood_guard_particles == 100  # Hold guard tuning fixed.
    assert config == original


def test_serial_partitions_cover_every_start_exactly_once():
    assigned = [start for partition in range(10) for start in range(partition, 100, 10)]
    assert len(assigned) == 100
    assert sorted(assigned) == list(range(100))


@pytest.mark.parametrize("arm", ["corenflos_warm", "ditlevsen_warm"])
def test_particle_completion_requires_every_trace_and_correct_effort(arm, tmp_path):
    path = experiment.checkpoint(tmp_path, arm, 0)
    path.parent.mkdir()
    np.savez(path, parameter_trace=np.zeros((2, 23)))
    assert experiment.stage_complete(tmp_path, arm, "fit", range(1))
    assert not experiment.stage_complete(tmp_path, arm, "final-eval", range(1))
    prefix = "euler" if arm.startswith("corenflos") else "euler_r20"
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    final = evaluations / f"{prefix}_start000.json"
    final.write_text(json.dumps({
        "evaluation_status": "ok", "eval_particles": 5000, "eval_replicates": 36,
    }))
    assert experiment.stage_complete(tmp_path, arm, "final-eval", range(1))
    assert not experiment.stage_complete(tmp_path, arm, "trace-eval", range(1))
    traces = tmp_path / "trace_evaluations"
    traces.mkdir()
    width = 4 if arm.startswith("corenflos") else 3
    for iteration in range(2):
        destination = traces / f"{prefix}_start000_iter{iteration:0{width}d}.json"
        destination.write_text(json.dumps({
            "evaluation_status": "ok", "eval_particles": 5000, "eval_replicates": 1,
        }))
    assert experiment.stage_complete(tmp_path, arm, "trace-eval", range(1))
    final.write_text(json.dumps({
        "evaluation_status": "ok", "eval_particles": 100, "eval_replicates": 36,
    }))
    with pytest.raises(RuntimeError, match="wrong evaluation effort"):
        experiment.stage_complete(tmp_path, arm, "final-eval", range(1))


def test_particle_completion_does_not_silently_skip_failed_evaluations(tmp_path):
    arm = "corenflos_warm"
    path = experiment.checkpoint(tmp_path, arm, 0)
    path.parent.mkdir()
    np.savez(path, parameter_trace=np.zeros((2, 23)))
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    (evaluations / "euler_start000.json").write_text(
        json.dumps({"evaluation_status": "non-finite"})
    )
    with pytest.raises(RuntimeError, match="failed evaluation requires inspection"):
        experiment.stage_complete(tmp_path, arm, "final-eval", range(1))


def test_missing_or_partial_consolidated_table_is_not_complete(tmp_path):
    assert not experiment.table_complete(tmp_path, "final-eval", range(2))
    pd.DataFrame({"start": [0]}).to_csv(tmp_path / "final_evaluations.csv", index=False)
    assert not experiment.table_complete(tmp_path, "final-eval", range(2))
    pd.DataFrame({"start": [0, 1]}).to_csv(tmp_path / "final_evaluations.csv", index=False)
    assert experiment.table_complete(tmp_path, "final-eval", range(2))


def test_particle_audit_checks_start_identity_and_exports_paired_comparison(tmp_path):
    configs = {}
    for arm, (_, project, reference) in experiment.ARMS.items():
        directory = tmp_path / arm
        directory.mkdir()
        baseline = experiment.ROOT / project / reference
        config = experiment.configuration(arm, directory, 1000, 1)
        configs[arm] = config
        (directory / "configuration.json").write_text(json.dumps(config))
        destination = experiment.checkpoint(directory, arm, 0)
        destination.parent.mkdir()
        with np.load(experiment.checkpoint(baseline, arm, 0), allow_pickle=False) as old:
            np.savez(destination, start=old["start"], parameter_trace=old["parameter_trace"][:1])
        final = pd.read_csv(baseline / "final_evaluations.csv")
        final = final.loc[final["start"].eq(0)].copy()
        final["euler20_loglik"] += 2.0
        final.to_csv(directory / "final_evaluations.csv", index=False)
        pd.DataFrame({"start": [0], "iteration": [0]}).to_csv(
            directory / "optimization_training_traces.csv", index=False,
        )
        trace = final.copy()
        trace["iteration"] = 0
        trace["eval_replicates"] = 1
        trace.to_csv(directory / "optimization_euler_traces.csv", index=False)
        pd.DataFrame({"start": [0], "updates_completed": [0]}).to_csv(
            directory / "fit_summary.csv", index=False,
        )
        traces = directory / "trace_evaluations"
        traces.mkdir()
        filename = "euler_start000_iter0000.json" if arm.startswith("corenflos") else "euler_r20_start000_iter000.json"
        (traces / filename).write_text(json.dumps({
            "evaluation_status": "ok", "eval_particles": 5000, "eval_replicates": 1,
        }))
    report = experiment.audit(tmp_path, configs)
    assert len(report["variants"]) == 4
    assert all(row["median_paired_change_from_j100"] == 2.0 for row in report["variants"])
    assert (tmp_path / "summary.csv").is_file()
    assert len(pd.read_csv(tmp_path / "paired_particle_comparison.csv")) == 4
