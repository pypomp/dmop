"""Export the Daphnia searches and saved-checkpoint evaluations as plain tables.

Usage:
    python export_results.py --source PATH/TO/SEARCH/results \
        --checkpoints PATH/TO/CHECKPOINT/results

No particle filters or optimization are run. The original files are read only.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RUNS = [("IFAD-0.97", "formal"), ("IFAD-1", "alpha1"),
        ("IFAD-0", "alpha0"), ("MPIF", "mpif680")]
KEYS = ["start", "block", "unit", "parameter"]


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def equal(left, right, keys):
    pd.testing.assert_frame_equal(
        left.sort_values(keys).reset_index(drop=True),
        right.sort_values(keys).reset_index(drop=True), check_exact=True,
    )


def wide(parameters):
    parameters = parameters.copy()
    parameters["name"] = parameters["parameter"]
    local = parameters["block"] == "unit_specific"
    parameters.loc[local, "name"] += "_" + parameters.loc[local, "unit"]
    order = parameters["name"].drop_duplicates()
    result = parameters.pivot(index="start", columns="name", values="value")[order]
    if result.shape != (50, 40) or result.isna().any().any():
        raise ValueError("Expected 50 complete parameter vectors, each with 40 entries")
    if not result[["sigSn", "sigSi"]].eq(0).all().all():
        raise ValueError("sigSn and sigSi must remain fixed at zero")
    return result.rename_axis(columns=None).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--reevaluation", default="reevaluation",
                        help="Subdirectory of --source with fresh best-point evaluations")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "exports")
    args = parser.parse_args()
    finals, traces, timings = [], [], []
    common_initial = None

    for model, arm in RUNS:
        source, checkpoints = args.source / arm, args.checkpoints / arm
        final = read_csv(source / "results_final.csv").sort_values("start")
        if final["start"].tolist() != list(range(50)):
            raise ValueError(f"{arm}: expected starts 0 to 49 exactly once")
        equal(final, read_csv(checkpoints / "results.csv"), ["start"])
        trace = read_csv(checkpoints / "traces.csv.gz")
        if trace.duplicated(["start", "checkpoint"]).any():
            raise ValueError(f"{arm}: duplicate checkpoints")
        for _, group in trace.groupby("checkpoint"):
            if sorted(group["start"]) != list(range(50)):
                raise ValueError(f"{arm}: incomplete checkpoint")
        last = trace.loc[trace["checkpoint"] == trace["checkpoint"].max()]
        equal(final[["start", "logLik", "MCSE"]],
              last[["start", "logLik", "MCSE"]], ["start"])

        endpoint, initial, batch_times = [], [], []
        batches = sorted(source.glob("batch_*"))
        if len(batches) != 25:
            raise ValueError(f"{arm}: expected 25 two-start batches")
        for batch, directory in enumerate(batches):
            offset = int(directory.name.split("_")[1])
            method = "mpif" if arm == "mpif680" else "ifad"
            for name, rows in [(method, endpoint), ("initial", initial)]:
                path = directory / f"{name}_parameters.csv"
                values = read_csv(path)
                values["start"] += offset
                rows.append(values)
            for name, phase in [("mpif", "mif"), ("train", "train")]:
                path = directory / f"{name}_timing.json"
                if name == "train" and arm == "mpif680":
                    continue
                timing = json.loads(path.read_text())
                batch_times.append({"batch": batch, "method": phase,
                                    "time": timing["seconds"], "iterations": timing["iterations"]})

        endpoint, initial = pd.concat(endpoint), pd.concat(initial)
        parameters = read_csv(checkpoints / "parameters.csv.gz")
        equal(endpoint, parameters.loc[parameters["checkpoint"] == trace["checkpoint"].max()]
              .drop(columns="checkpoint"), KEYS)
        equal(initial, parameters.loc[parameters["checkpoint"] == 0]
              .drop(columns="checkpoint"), KEYS)
        if common_initial is None:
            common_initial = initial
        else:
            equal(common_initial, initial, KEYS)
        times = read_csv(checkpoints / "timings.csv")
        equal(pd.DataFrame(batch_times), times, ["batch", "method"])
        estimates = wide(endpoint)
        finals.append(final.merge(estimates, on="start", validate="one_to_one")
                      .rename(columns={"start": "rep"}).assign(model=model))
        traces.append(trace.rename(columns={"start": "rep"}).assign(model=model))
        timings.append(times.rename(columns={"time": "seconds"}).assign(model=model))

        latest = json.loads((checkpoints / "latest.json").read_text())
        # batch=-1 denotes the aggregate saved-checkpoint PF evaluation time.
        timings.append(pd.DataFrame([{
            "model": model, "batch": -1, "method": "pfilter",
            "seconds": latest["evaluation_seconds"], "iterations": latest["checkpoints"],
        }]))

    reeval_dir = args.source / args.reevaluation
    reeval = read_csv(reeval_dir / "results.csv").rename(columns={"method": "model", "start": "rep"})
    if sorted(reeval["model"]) != sorted(model for model, _ in RUNS):
        raise ValueError("Expected one re-evaluation per method")
    final = pd.concat(finals, ignore_index=True)
    for model, _ in RUNS:
        selected = reeval.loc[reeval["model"] == model].iloc[0]
        best = final.loc[final["model"] == model].sort_values(
            ["logLik", "rep"], ascending=[False, True]).iloc[0]
        if (selected["rep"], selected["original_logLik"], selected["original_MCSE"]) != (
                best["rep"], best["logLik"], best["MCSE"]):
            raise ValueError(f"{model}: re-evaluation is not at the original maximizing estimate")
    reeval = reeval.rename(columns={
        "seed": "reevaluation_seed", "seconds_including_compilation": "reevaluation_seconds",
        "difference": "reevaluation_difference",
    })
    columns = ["model", "rep", "reevaluated_logLik", "reevaluated_MCSE",
               "reevaluation_seed", "reevaluation_seconds", "reevaluation_difference"]
    final = final.merge(reeval[columns], on=["model", "rep"], how="left", validate="one_to_one")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in [("final_estimates.csv", final), ("loglik_traces.csv.gz", pd.concat(traces)),
                         ("timings.csv", pd.concat(timings))]:
        columns = ["model"] + [c for c in values.columns if c != "model"]
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        values[columns].to_csv(args.output / name, index=False, compression=compression)
        print(f"{name}: {len(values):,} rows")


if __name__ == "__main__":
    main()
