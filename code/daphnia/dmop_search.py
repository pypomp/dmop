"""Fit IFAD from the 50 shared starts with the selected learning rate."""

import prep


def main():
    args = prep.arguments("IFAD")
    starts = prep.payloads(prep.pp.PanelParameters(
        prep.model.sample_starts(args.starts, seed=2026091802)))
    rows = []
    for offset in range(0, args.starts, args.batch_size):
        directory = args.output / f"batch_{offset:03d}"
        directory.mkdir()
        theta, baseline, _ = prep.warm(starts[offset:offset + args.batch_size], args,
                                      directory, 6000 + offset)
        table, _ = prep.train(theta, args, directory, 0.01, args.alpha, 7000 + offset)
        table["mpif_logLik"] = baseline.logLik.to_numpy()
        table["gain_from_mpif"] = table.logLik - table.mpif_logLik
        table["start"] += offset
        rows.append(table)
        result = prep.pd.concat(rows, ignore_index=True)
        result.to_csv(args.output / "results_partial.csv", index=False)
        print(f"IFAD-{args.alpha}: {len(result)}/{args.starts}, "
              f"best={result.logLik.max():.3f}", flush=True)
    result.to_csv(args.output / "results_final.csv", index=False)
    prep.write_json(args.output / "summary.json", {
        "complete": True, "starts": len(result),
        "median_logLik": float(result.logLik.median()),
        "best_logLik": float(result.logLik.max()),
    })


if __name__ == "__main__":
    main()
