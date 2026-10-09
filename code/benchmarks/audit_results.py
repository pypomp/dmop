"""Check saved evaluations and Dhaka A/B output selection without refitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]


def likelihood_summary(raw):
    raw = np.asarray(raw)
    if raw.ndim == 1:
        raw = raw[:, None]
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid raw likelihood evaluations")
    weights = np.exp(raw - raw.max(axis=0))
    ll = float((logsumexp(raw, axis=0) - np.log(len(raw))).sum())
    se = float(np.linalg.norm(weights.std(axis=0, ddof=1) / np.sqrt(len(raw)) / weights.mean(axis=0)))
    return ll, se


def audit_batch(folder):
    status = json.loads((folder / "status.json").read_text())
    if not status["complete"]:
        raise ValueError(f"Incomplete batch: {folder}")
    config = json.loads((folder / "configuration.json").read_text())
    final = pd.read_csv(folder / "checkpoints.csv").loc[lambda f: f.final]
    if final.duplicated(["start", "method"]).any():
        raise ValueError("Duplicate final evaluations")
    raw_path = folder / "evaluation_replicates.csv"
    if not raw_path.exists():
        return {"folder": str(folder), "finals": len(final), "raw_replicate_checks": 0}
    draws = pd.read_csv(raw_path)
    for row in final.itertuples():
        raw = draws.loc[draws.start.eq(row.start) & draws.method.eq(row.method)
                        & draws.iteration.eq(row.iteration)]
        if "unit" in raw:
            values = raw.pivot(index="replicate", columns="unit", values="loglik").to_numpy()
        else:
            if raw.replicate.duplicated().any():
                raise ValueError("Duplicate evaluation replicates")
            values = raw.sort_values("replicate").loglik.to_numpy()
        if len(values) != config["eval_reps"]:
            raise ValueError(f"Wrong evaluation count: {folder}/{row.start}/{row.method}")
        np.testing.assert_allclose(likelihood_summary(values), [row.loglik, row.mcse], rtol=1e-10, atol=1e-8)
    return {"folder": str(folder), "finals": len(final), "raw_replicate_checks": len(final)}


def audit_ab(root):
    protocol = json.loads((root / "protocol.json").read_text())
    starts = np.load(root / "starts.npz")
    checked = []
    for regime in ("cold", "warm"):
        for index in range(protocol["starts"]):
            folder = root / regime / f"start{index:03d}"
            status = folder / "status.json"
            if not status.exists() or not json.loads(status.read_text())["complete"]:
                continue
            score = np.load(folder / "score_parameters.npz")
            info = json.loads((folder / "score.json").read_text())
            eligible = np.flatnonzero((score["elapsed_trace"] <= protocol["budget_seconds"])
                & np.isfinite(score["parameter_trace"]).all(axis=1)
                & np.isfinite(score["marginal_loglik_trace"]))
            selected_index = int(eligible[-1]) if len(eligible) else -1
            expected = score["parameter_trace"][selected_index] if selected_index >= 0 else starts[regime][index]
            assert selected_index == info["selected_index"]
            np.testing.assert_array_equal(score["selected"], expected)
            accepted = int(np.count_nonzero(score["accepted_step_size_trace"][:max(selected_index, 0)] > 0))
            saem = np.load(folder / "saem_parameters.npz")
            np.testing.assert_array_equal(saem["selected"], saem["parameters"][-1])
            np.testing.assert_allclose(saem["parameters"][0], starts[regime][index], rtol=0, atol=1e-12)
            trace_path = folder / "saem_trace.csv"
            saem_diagnostics = {"saem_completed_iterations": 0, "saem_q_increases_over_1e_minus_7": 0,
                                "saem_mstep_converged": 0, "saem_total_mstep_evaluations": 0}
            if trace_path.exists() and trace_path.stat().st_size > 1:
                trace = pd.read_csv(trace_path)
                assert len(trace) == len(saem["parameters"]) - 1
                assert (trace.seconds <= protocol["budget_seconds"]).all()
                assert np.isfinite(trace[["q_before", "q_after"]].to_numpy()).all()
                assert (trace.q_after >= trace.q_before - 1e-7).all()
                saem_diagnostics = {
                    "saem_completed_iterations": len(trace),
                    "saem_q_increases_over_1e_minus_7": int((trace.q_after - trace.q_before > 1e-7).sum()),
                    "saem_mstep_converged": int(trace.mstep_converged.sum()),
                    "saem_total_mstep_evaluations": int(trace.mstep_evaluations.sum())}
            final = pd.read_csv(folder / "evaluation.csv").set_index("method")
            draws = pd.read_csv(folder / "evaluation_replicates.csv")
            raw = draws.pivot(index="replicate", columns="method", values="loglik")
            assert set(final.index) == set(raw.columns) == {"initial", "score", "saem"}
            assert len(raw) == protocol["eval_replicates"]
            for method in raw:
                np.testing.assert_allclose(likelihood_summary(raw[method]),
                    final.loc[method, ["loglik", "mcse"]].astype(float), rtol=1e-10, atol=1e-8)
            checked.append({"regime": regime, "start": index, "score_selected_iteration": selected_index,
                            "score_accepted_updates_before_selection": accepted, **saem_diagnostics})
    return {"root": str(root), "checked_pairs": len(checked), "expected_pairs": 2 * protocol["starts"], "pairs": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, nargs="*", default=[])
    parser.add_argument("--dhaka-ab", action="store_true")
    parser.add_argument("--dhaka-ab-root", type=Path, default=ROOT / "ditlevsen/results/saem_ab_scaled")
    parser.add_argument("--output", type=Path, help="Save the same audit printed to stdout")
    args = parser.parse_args()
    result = {"batches": [audit_batch(folder) for folder in args.batch]}
    if args.dhaka_ab:
        result["dhaka_ab"] = audit_ab(args.dhaka_ab_root)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
