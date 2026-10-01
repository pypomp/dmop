import pandas as pd

from ditlevsen.smc_plots import _ditlevsen_trace_summary


def test_trace_summary_keeps_every_update_and_carries_early_failures():
    traces = pd.DataFrame(
        {
            "inference_nstep": [20, 20, 20, 20, 20],
            "start": [0, 0, 0, 1, 1],
            "iteration": [0, 1, 2, 0, 1],
            "elapsed_seconds": [10.0, 20.0, 30.0, 11.0, 22.0],
            "euler20_loglik": [-10.0, -8.0, -9.0, -12.0, -7.0],
        }
    )

    summary = _ditlevsen_trace_summary(traces, nstep=20, budget=800.0)

    assert summary["iteration"].tolist() == [0, 1, 2]
    assert summary["elapsed_seconds"].tolist() == [10.5, 21.0, 26.0]
    assert summary["median"].tolist() == [-11.0, -7.5, -8.0]
    assert summary["maximum"].tolist() == [-10.0, -7.0, -7.0]
