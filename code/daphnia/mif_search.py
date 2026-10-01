"""Fit the MPIF-only comparison from the same 50 starts as IFAD."""

import prep


def main():
    args = prep.arguments("MPIF")
    starts = prep.payloads(prep.pp.PanelParameters(
        prep.model.sample_starts(args.starts, seed=2026091802)))
    rows, timings = [], []
    for offset in range(0, args.starts, args.batch_size):
        directory = args.output / f"batch_{offset:03d}"
        directory.mkdir()
        _, table, seconds = prep.warm(starts[offset:offset + args.batch_size], args,
                                     directory, 6000 + offset)
        table["start"] += offset
        rows.append(table)
        timings.append(seconds)
        result = prep.pd.concat(rows, ignore_index=True)
        result.to_csv(args.output / "results_partial.csv", index=False)
        print(f"MPIF: {len(result)}/{args.starts}, best={result.logLik.max():.3f}", flush=True)
    result.to_csv(args.output / "results_final.csv", index=False)
    prep.write_json(args.output / "summary.json", {
        "complete": True, "starts": len(result), "iterations": args.warm_iterations,
        "median_logLik": float(result.logLik.median()),
        "best_logLik": float(result.logLik.max()),
        "total_mpif_seconds": sum(timings), "batch_mpif_seconds": timings,
    })


if __name__ == "__main__":
    main()
