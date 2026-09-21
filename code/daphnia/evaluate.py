"""Evaluate saved Daphnia checkpoints, or re-evaluate the best estimates with --best."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import model
import numpy as np
import pandas as pd
import pypomp as pp
import prep
from scipy.special import logsumexp

ARMS = {"alpha0": "IFAD-0", "formal": "IFAD-0.97", "alpha1": "IFAD-1", "mpif680": "MPIF"}
ALPHAS = {"alpha0": 0.0, "formal": 0.97, "alpha1": 1.0}
SEED = 202609190
DATA_SHA256 = "1c67690802183cc3aabd95d552090b1b0c1e22731aa7282904af411502fe9605"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def saved_parameters(path):
    rows = read_csv(path)
    result = []
    for _, group in rows.groupby("start", sort=True):
        shared = group[group.block == "shared"].pivot(index="parameter", columns="unit", values="value")
        specific = group[group.block == "unit_specific"].pivot(index="parameter", columns="unit", values="value")
        result.append({"shared": shared.loc[list(model.SHARED_PARAMETERS), ["shared"]],
                       "unit_specific": specific.loc[list(model.UNIT_PARAMETERS), list(model.UNITS)]})
    return result


def trace_parameters(trace, iteration):
    result = []
    for _, group in trace[trace.iteration == iteration].groupby("theta_idx", sort=True):
        if len(group) != 9 or set(group.unit) != {"shared", *model.UNITS}:
            raise ValueError("Expected one shared row and eight unit rows per start")
        group = group.set_index("unit")
        shared = group.loc["shared", list(model.SHARED_PARAMETERS)].astype(float)
        specific = group.loc[list(model.UNITS), list(model.UNIT_PARAMETERS)].astype(float).T
        duplicated = group.loc[list(model.UNITS), list(model.SHARED_PARAMETERS)].to_numpy(dtype=float)
        np.testing.assert_allclose(duplicated, np.tile(shared.to_numpy(), (8, 1)), rtol=1e-12, atol=0)
        result.append({"shared": shared.to_frame("shared"), "unit_specific": specific})
    if len(result) != 2:
        raise ValueError("Expected two starts per batch")
    for params in result:
        values = np.concatenate([frame.to_numpy().ravel() for frame in params.values()])
        if len(values) != 40 or not np.isfinite(values).all():
            raise ValueError("Invalid reconstructed parameters")
        if not (params["shared"].loc[list(model.FIXED_PARAMETERS)].to_numpy() == 0).all():
            raise ValueError("Fixed sigSn/sigSi changed")
        if not (params["shared"].drop(index=list(model.FIXED_PARAMETERS)).to_numpy() > 0).all() or not (params["unit_specific"].to_numpy() > 0).all():
            raise ValueError("Non-positive free parameter")
    return result


def assert_parameters(left, right):
    if len(left) != len(right):
        raise ValueError("Different numbers of parameter vectors")
    for x, y in zip(left, right):
        for block in ("shared", "unit_specific"):
            np.testing.assert_allclose(x[block], y[block], rtol=1e-12, atol=0)


def aggregate(raw):
    if raw.ndim != 3 or raw.shape[:2] != (2, 8) or raw.shape[-1] < 2 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid ordinary PF replicates: {raw.shape}")
    unit_ll = logsumexp(raw, axis=-1) - np.log(raw.shape[-1])
    weights = np.exp(raw - raw.max(axis=-1, keepdims=True))
    se = weights.std(axis=-1, ddof=1) / np.sqrt(raw.shape[-1]) / weights.mean(axis=-1)
    return unit_ll.sum(axis=-1), np.sqrt(np.square(se).sum(axis=-1))


def load_searches(source):
    configs, finals, timings = {}, {}, {}
    original = model.sample_starts(50, seed=2026091802)
    for arm in ARMS:
        root = source / arm
        configs[arm] = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        final = read_csv(root / "results_final.csv").sort_values("start")
        if not summary["complete"] or final.start.tolist() != list(range(50)):
            raise ValueError(f"Incomplete or duplicate final starts: {arm}")
        if not np.isfinite(final[["logLik", "MCSE"]]).all().all() or (final.MCSE < 0).any():
            raise ValueError(f"Invalid final evaluations: {arm}")
        config = configs[arm]
        for name, expected in {"starts": 50, "batch_size": 2, "particles": 1000, "eval_particles": 2000, "eval_reps": 10}.items():
            if config[name] != expected:
                raise ValueError(f"Unexpected {arm} setting: {name}")
        prep.check_model(config)
        if sha256(model.DATA_PATH) != DATA_SHA256:
            raise ValueError("Observation data changed")
        records = []
        for offset in range(0, 50, 2):
            batch = root / f"batch_{offset:03d}"
            assert_parameters(saved_parameters(batch / "initial_parameters.csv"), original[offset:offset + 2])
            label = "mpif" if arm == "mpif680" else "ifad"
            raw = np.load(batch / f"{label}_pf.npy")
            if raw.shape != (2, 8, 10):
                raise ValueError("Stored final PF shape differs from the evaluation settings")
            ll, mcse = aggregate(raw)
            selected = final.set_index("start").loc[[offset, offset + 1]]
            np.testing.assert_allclose(ll, selected.logLik, rtol=0, atol=1e-10)
            np.testing.assert_allclose(mcse, selected.MCSE, rtol=0, atol=1e-12)
            for stage, filename in [("mif", "mpif_timing.json"), ("train", "train_timing.json")]:
                if stage == "train" and arm == "mpif680":
                    continue
                timing = json.loads((batch / filename).read_text())
                records.append({"batch": offset // 2, "method": stage,
                                "time": timing["seconds"], "iterations": timing["iterations"]})
        timings[arm] = pd.DataFrame(records)
        expected_mif = 680 if arm == "mpif680" else 200
        if not (timings[arm].query("method == 'mif'").iterations == expected_mif).all():
            raise ValueError("Unexpected MPIF iteration count")
        if arm != "mpif680" and not (timings[arm].query("method == 'train'").iterations == 200).all():
            raise ValueError("Unexpected Adam iteration count")
        finals[arm] = final
    return configs, finals, timings


def checkpoint_grid(timings):
    durations = {arm: frame.groupby("method").time.sum().to_dict() for arm, frame in timings.items()}
    ends = {arm: sum(stages.values()) for arm, stages in durations.items()}
    grid = sorted(set(np.linspace(0, max(ends.values()), 21)) | set(ends.values()) |
                  {stages["mif"] for arm, stages in durations.items() if arm != "mpif680"})
    plans = {}
    for arm, stages in durations.items():
        points = []
        for requested in grid:
            if requested > ends[arm] + 1e-8:
                continue
            stage = "mif" if requested <= stages["mif"] + 1e-8 else "train"
            limit = 680 if arm == "mpif680" else 200
            elapsed = requested if stage == "mif" else requested - stages["mif"]
            iteration = min(limit, int(np.floor(elapsed / stages[stage] * limit + 1e-10)))
            actual = stages[stage] * iteration / limit + (stages["mif"] if stage == "train" else 0)
            if not points or (stage, iteration) != (points[-1]["stage"], points[-1]["iteration"]):
                points.append({"checkpoint": len(points), "stage": stage, "iteration": iteration,
                               "time_scaled": actual, "requested_time": float(requested)})
        plans[arm] = points
    return plans


def read_batch(source, arm, offset):
    batch = source / arm / f"batch_{offset:03d}"
    traces = {"mif": read_csv(batch / "mpif_traces.csv.gz")}
    end = 680 if arm == "mpif680" else 200
    assert_parameters(trace_parameters(traces["mif"], end), saved_parameters(batch / "mpif_parameters.csv"))
    if arm != "mpif680":
        traces["train"] = read_csv(batch / "ifad_traces.csv.gz")
        assert_parameters(trace_parameters(traces["mif"], 200), trace_parameters(traces["train"], 0))
        assert_parameters(trace_parameters(traces["train"], 200), saved_parameters(batch / "ifad_parameters.csv"))
    return batch, traces


def pf(parameters, particles, reps, seed):
    panel = model.build_panel(starts=parameters)
    started = time.perf_counter()
    panel.pfilter(J=particles, reps=reps, key=jax.random.key(seed))
    raw = np.asarray(panel.results_history[-1].logLiks.values)
    seconds = time.perf_counter() - started
    if raw.shape != (2, 8, reps) or not np.isfinite(raw).all():
        raise ValueError(f"Non-finite ordinary PF or unexpected shape {raw.shape}")
    ll, mcse = aggregate(raw)
    return ll, mcse, raw, seconds


def evaluate_checkpoints(args):
    configs, finals, timings = load_searches(args.source)
    plans = checkpoint_grid(timings)
    for arm, points in plans.items():
        for offset in range(0, 50, 2):
            batch, traces = read_batch(args.source, arm, offset)
            for point in points:
                if point["iteration"]:
                    trace_parameters(traces[point["stage"]], point["iteration"])
        best = finals[arm].loc[finals[arm].logLik.idxmax()]
        print(f"{arm}: validated 50 starts, {len(points)} checkpoints; "
              f"best={best.logLik:.6f} MCSE={best.MCSE:.6f} median={finals[arm].logLik.median():.6f}", flush=True)
    if args.validate_only:
        return
    if pp.__version__ != "1.0.5" or not jax.config.jax_enable_x64:
        raise RuntimeError("Expected PyPOMP 1.0.5 and float64")
    args.output.mkdir(parents=True, exist_ok=False)
    batch, traces = read_batch(args.source, "formal", 0)
    reconstructed = trace_parameters(traces["train"], 200)
    if args.smoke:
        a = pf(reconstructed, 32, 2, SEED)
        b = pf(saved_parameters(batch / "ifad_parameters.csv"), 32, 2, SEED)
        np.testing.assert_allclose(a[2], b[2], rtol=0, atol=1e-8)
        write_json(args.output / "summary.json", {"smoke_passed": True, "particles": 32, "reps": 2,
                                                   "seconds": a[3] + b[3]})
        print("Reconstructed and saved parameter PF test passed", flush=True)
        return
    if jax.default_backend() != "gpu":
        raise RuntimeError("Full checkpoint evaluation requires the single allocated GPU")
    # Reproduce one stored final evaluation before using fresh evaluation keys.
    reference = pf(reconstructed, 2000, 10, 7100)
    np.testing.assert_allclose(reference[2], np.load(batch / "ifad_pf.npy"), rtol=0, atol=1e-6)
    pilot = [pf(reconstructed, 2000, 10, SEED + i) for i in range(2)]
    projected = sum(len(p) for p in plans.values()) * 25 * max(x[3] for x in pilot) * 1.3
    write_json(args.output / "validation.json", {"all_searches_complete": True,
        "parameter_reconstruction_passed": True, "stored_pf_reproduction_passed": True,
        "gpu_compile_pf_seconds": reference[3], "cached_pf_seconds": [x[3] for x in pilot],
        "projected_evaluation_seconds": projected})
    if projected > 6000:
        raise RuntimeError(f"Projected PF cost {projected:.0f}s exceeds the initial allocation margin; ask before continuing")
    print(f"GPU check passed; projected PF time {projected:.0f}s", flush=True)
    total = sum(len(points) * 25 for points in plans.values())
    completed, started = 0, time.perf_counter()
    for arm, points in plans.items():
        out = args.output / arm
        out.mkdir()
        records, replicates, parameters, seconds = [], [], [], 0.0
        for offset in range(0, 50, 2):
            batch, traces = read_batch(args.source, arm, offset)
            initial = saved_parameters(batch / "initial_parameters.csv")
            for point in points:
                stage, iteration = point["stage"], point["iteration"]
                theta = initial if stage == "mif" and iteration == 0 else trace_parameters(traces[stage], iteration)
                terminal = point == points[-1]
                # Retain the original independent final PF; all other points get fresh PFs.
                if terminal:
                    label = "mpif" if arm == "mpif680" else "ifad"
                    raw = np.load(batch / f"{label}_pf.npy")
                    final = finals[arm].set_index("start").loc[[offset, offset + 1]]
                    ll, mcse = final.logLik.to_numpy(), final.MCSE.to_numpy()
                    seed = (6100 if arm == "mpif680" else 7100) + offset
                else:
                    seed = SEED + 1000 * point["checkpoint"] + offset
                    ll, mcse, raw, duration = pf(theta, 2000, 10, seed)
                    seconds += duration
                for local_id in range(2):
                    start = offset + local_id
                    records.append({**point, "start": start, "logLik": float(ll[local_id]),
                                    "MCSE": float(mcse[local_id]), "seed": seed,
                                    "evaluation": "stored_final" if terminal else "fresh_pf"})
                    for unit_id, unit in enumerate(model.UNITS):
                        for rep in range(10):
                            replicates.append({"start": start, "checkpoint": point["checkpoint"],
                                               "unit": unit, "replicate": rep, "logLik": raw[local_id, unit_id, rep], "seed": seed})
                    for block, frame in theta[local_id].items():
                        for name, row in frame.iterrows():
                            for unit, value in row.items():
                                parameters.append({"start": start, "checkpoint": point["checkpoint"], "block": block,
                                                   "unit": unit, "parameter": name, "value": float(value)})
                completed += 1
            pd.DataFrame(records).to_csv(out / "traces_partial.csv", index=False)
            write_json(args.output / "progress.json", {"arm": arm, "completed_starts_in_arm": offset + 2,
                "completed_batches": completed, "total_batches": total,
                "elapsed_seconds": time.perf_counter() - started, "updated_at": datetime.now(timezone.utc).isoformat()})
            print(f"{arm} {offset + 2}/50; PF checkpoints {completed}/{total}", flush=True)
        pd.DataFrame(records).to_csv(out / "traces.csv.gz", index=False)
        pd.DataFrame(replicates).to_csv(out / "pfilter_logliks.csv", index=False)
        pd.DataFrame(parameters).to_csv(out / "parameters.csv.gz", index=False)
        finals[arm].to_csv(out / "results.csv", index=False)
        timings[arm].to_csv(out / "timings.csv", index=False)
        config = {**configs[arm], "alpha": ALPHAS.get(arm), "learning_rate": 0.01 if arm != "mpif680" else None}
        metadata = {"pypomp_version": pp.__version__, "jax_version": jax.__version__,
                    "run_config": {"eval_particles": 2000, "eval_reps": 10, "seed": SEED},
                    "devices": [d.device_kind for d in jax.devices()]}
        write_json(out / "latest.json", {**metadata, "complete": True, "search_config": config,
            "evaluation_seconds": seconds, "checkpoints": len(points), "starts": 50,
            "source": str(args.source / arm), "source_final_sha256": hashlib.sha256((args.source / arm / "results_final.csv").read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
            "mapped_time_note": "Sum of 25 batch optimization times, including compilation, mapped linearly within each phase; sparse checkpoints; PF evaluation time excluded.",
            "final_evaluation": "Original independent ordinary PF retained; intermediate and initial PFs use fresh keys."})
    write_json(args.output / "summary.json", {"complete": True, "arms": list(ARMS), "starts_per_arm": 50,
        "completed_batches": completed, "total_batches": total, "elapsed_seconds": time.perf_counter() - started})


def select_best(source):
    selected, payloads, parameter_rows = [], [], []
    for index, (arm, method) in enumerate(ARMS.items()):
        root = source / arm
        config = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        final = read_csv(root / "results_final.csv")
        if not summary["complete"] or sorted(final.start.tolist()) != list(range(50)):
            raise ValueError(f"Incomplete source results: {arm}")
        for field, expected in {"eval_particles": 2000, "eval_reps": 10, "batch_size": 2,
                                "pypomp_commit": prep.PYPOMP_COMMIT}.items():
            if config[field] != expected:
                raise ValueError(f"Unexpected {arm} setting: {field}")
        prep.check_model(config)
        if not np.isfinite(final[["logLik", "MCSE"]].to_numpy()).all():
            raise ValueError(f"Non-finite source result: {arm}")
        best = final.sort_values(["logLik", "start"], ascending=[False, True]).iloc[0]
        start = int(best.start)
        offset = start // 2 * 2
        local = start - offset
        label = "mpif" if arm == "mpif680" else "ifad"
        batch = root / f"batch_{offset:03d}"
        parameter_file = batch / f"{label}_parameters.csv"
        rows = read_csv(parameter_file).query("start == @local").copy()
        payload = saved_parameters(parameter_file)[local]
        if len(rows) != 40 or payload["shared"].shape != (24, 1) or payload["unit_specific"].shape != (2, 8):
            raise ValueError(f"Incorrect parameter dimensions: {arm}")
        raw = np.load(batch / f"{label}_pf.npy")
        if raw.shape != (2, 8, 10):
            raise ValueError(f"Invalid stored PF array: {arm}")
        ll, mcse = aggregate(raw)
        np.testing.assert_allclose([ll[local], mcse[local]], [best.logLik, best.MCSE], rtol=0, atol=1e-10)
        selected.append({"arm": arm, "method": method, "start": start,
                         "batch_offset": offset, "local_index": local,
                         "original_logLik": float(best.logLik), "original_MCSE": float(best.MCSE),
                         "seed": 2026092001 + index,
                         "source_parameters": str(parameter_file.resolve()),
                         "parameters_sha256": sha256(parameter_file),
                         "source_results_sha256": sha256(root / "results_final.csv")})
        payloads.append(payload)
        parameter_rows.append(rows.assign(arm=arm, selected_start=start))
    return selected, payloads, pd.concat(parameter_rows, ignore_index=True)


def evaluate_best(args):
    if pp.__version__ != "1.0.5" or not jax.config.jax_enable_x64:
        raise RuntimeError("Expected pinned PyPOMP 1.0.5 and float64")
    if sha256(model.DATA_PATH) != DATA_SHA256:
        raise ValueError("Observation data changed")
    selected, payloads, parameter_rows = select_best(args.source)
    if args.validate_only:
        for record in selected:
            print(f"{record['method']}: selected start {record['start']}, seed {record['seed']}")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    parameter_rows.to_csv(args.output / "selected_parameters.csv", index=False)
    package = Path(pp.__file__).parent
    package_hash = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        package_hash.update(str(path.relative_to(package)).encode())
        package_hash.update(path.read_bytes())
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "selection": "Largest original endpoint logLik in each method; ties use lowest start",
                "eval_particles": 2000, "eval_reps": 10, "units": list(model.UNITS),
                "pypomp_commit": prep.PYPOMP_COMMIT, "pypomp_version": pp.__version__,
                "pypomp_path": pp.__file__, "pypomp_source_sha256": package_hash.hexdigest(),
                "jax_version": jax.__version__, "backend": jax.default_backend(),
                "devices": [d.device_kind for d in jax.devices()], "float64": True,
                "data_sha256": DATA_SHA256, "script_sha256": sha256(Path(__file__)),
                "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("model.py", "prep.py")},
                "selected": selected}
    # Fix all four selections and seeds before re-evaluation.
    write_json(args.output / "manifest.json", manifest)
    eval_args = argparse.Namespace(eval_particles=2000, eval_reps=10)
    results = []
    for record, payload in zip(selected, payloads):
        panel = model.build_panel(starts=[payload])
        prep.check_parameters(panel)
        before = prep.payloads(panel.theta)[0]
        print(f"Evaluating {record['method']} at original start {record['start']}, seed {record['seed']}", flush=True)
        began = time.perf_counter()
        table, raw = prep.evaluate(panel, eval_args, record["seed"])
        seconds = time.perf_counter() - began
        if raw.shape != (1, 8, 10):
            raise ValueError(f"Unexpected fresh PF shape: {raw.shape}")
        after = prep.payloads(panel.theta)[0]
        for block in ("shared", "unit_specific"):
            pd.testing.assert_frame_equal(payload[block], before[block], check_names=False)
            pd.testing.assert_frame_equal(before[block], after[block])
        np.save(args.output / f"{record['arm']}_pf.npy", raw)
        result = {"method": record["method"], "arm": record["arm"], "start": record["start"],
                  "original_logLik": record["original_logLik"], "original_MCSE": record["original_MCSE"],
                  "reevaluated_logLik": float(table.logLik.iloc[0]), "reevaluated_MCSE": float(table.MCSE.iloc[0]),
                  "seed": record["seed"], "seconds_including_compilation": seconds}
        result["difference"] = result["reevaluated_logLik"] - result["original_logLik"]
        results.append(result)
        write_json(args.output / "progress.json", {"completed": len(results), "total": 4, "results": results})
        print(f"{record['method']}: {result['reevaluated_logLik']:.6f} (MCSE {result['reevaluated_MCSE']:.6f}); {seconds:.2f}s", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(args.output / "results.csv", index=False)
    write_json(args.output / "summary.json", {"complete": True, "methods": 4, "optimization_rerun": False,
               "fresh_evaluations_per_method": 1, "eval_particles": 2000, "eval_reps": 10,
               "finished_at": datetime.now(timezone.utc).isoformat(), "results": results})
    print(table.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--best", action="store_true", help="Re-evaluate the previously selected best estimate for each method")
    parser.add_argument("--validate-only", action="store_true", help="Check saved results without running particle filters")
    parser.add_argument("--smoke", action="store_true", help="Test checkpoint reconstruction with a small particle filter")
    args = parser.parse_args()
    if args.best and args.smoke:
        parser.error("--smoke is only available for checkpoint evaluation")
    if args.output is None:
        args.output = args.source / "reevaluation" if args.best else Path("checkpoint_results")
    if args.best:
        evaluate_best(args)
    else:
        evaluate_checkpoints(args)


if __name__ == "__main__":
    main()
