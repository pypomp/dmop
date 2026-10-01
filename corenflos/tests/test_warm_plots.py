from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from plotnine import ggplot

from corenflos import warm_plots


@pytest.fixture
def optimization_frame(monkeypatch):
    args = SimpleNamespace(
        reference=Path("reference"),
        warm_start=Path("checkpoint"),
        corenflos_warm=Path("corenflos_warm"),
        ditlevsen_warm=Path("ditlevsen_warm"),
        corenflos_cold=Path("corenflos_cold"),
        ditlevsen_cold=Path("ditlevsen_cold"),
    )
    reference = pd.DataFrame(
        {
            "method": ["IFAD-0.97", "IFAD-0.97"],
            "effort": ["comparable", "comparable"],
            "stage": ["mif", "train"],
            "elapsed_seconds": [warm_plots.OFFSET, warm_plots.OFFSET + 50],
            "median": [-3770.0, -3750.0],
            "q10": [-3800.0, -3760.0],
            "maximum": [-3750.0, -3745.0],
        }
    )
    files = {
        args.reference / "manuscript_trace_summary.csv": reference,
        args.warm_start / "warm_start_evaluations.csv": pd.DataFrame(
            {"euler20_loglik": [-3760.0, -3780.0]}
        ),
    }
    for directory, offset, loglik in (
        (args.corenflos_warm, warm_plots.OFFSET, -3769.0),
        (args.ditlevsen_warm, warm_plots.OFFSET, -3771.0),
        (args.corenflos_cold, 0.0, -4560.0),
        (args.ditlevsen_cold, 0.0, -4100.0),
    ):
        files[directory / "optimization_euler_traces.csv"] = pd.DataFrame(
            {
                "start": [0, 0],
                "iteration": [0, 1],
                "elapsed_seconds": [offset + 1, offset + 2],
                "euler20_loglik": [loglik - 1, loglik],
            }
        )
    monkeypatch.setattr(warm_plots, "_read", lambda path: files[path].copy())
    return warm_plots._optimization_frame(args)


def test_optimization_frame_keeps_standalone_runs_and_warm_origins(optimization_frame):
    frame = optimization_frame
    assert set(frame["source"]) == {
        "IF2 warm start", "IFAD-0.97", "Corenflos + IF2", "Ditlevsen + IF2",
        "Corenflos", "Ditlevsen",
    }
    for source in ("Corenflos", "Ditlevsen"):
        assert frame.loc[frame["source"].eq(source), "elapsed_seconds"].min() == 1.0
    for source in ("Corenflos + IF2", "Ditlevsen + IF2"):
        origin = frame.loc[
            frame["source"].eq(source) & frame["elapsed_seconds"].eq(warm_plots.OFFSET)
        ]
        assert len(origin) == 1
        assert origin.iloc[0]["median"] == -3770.0


def test_optimization_views_keep_six_sources_and_requested_limits(
    optimization_frame, monkeypatch, tmp_path
):
    saved = {}
    monkeypatch.setattr(
        ggplot, "save", lambda self, filename, **kwargs: saved.update({filename.name: self})
    )
    original = optimization_frame.copy(deep=True)
    warm_plots.plot_optimization(optimization_frame, tmp_path)
    pd.testing.assert_frame_equal(optimization_frame, original)
    focused = saved["optimization_if2warm_elapsed_r20.png"]
    overview = saved["optimization_if2warm_elapsed_full_r20.png"]
    assert focused.coordinates.limits.y == warm_plots.TRACE_LIMITS
    assert overview.coordinates.limits.y == (-4800.0, -3735.0)
    assert set(focused.data["source"]) == {
        "IF2 warm start", "IFAD-0.97", "Corenflos + IF2", "Ditlevsen + IF2",
    }
    assert set(overview.data["source"]) == set(original["source"])
    assert overview.data["source"].notna().all()
    corenflos = overview.data.loc[overview.data["source"].eq("Corenflos")]
    assert corenflos["median"].max() == -4560.0  # Never clamp to the visible axis.
    assert corenflos["median"].between(*overview.coordinates.limits.y).all()
    for plot in (focused, overview):
        assert plot.labels.caption is None
        for aesthetic in ("color", "fill", "linetype"):
            scale = plot.scales.get_scales(aesthetic)
            assert scale.labels == [
                warm_plots.DISPLAY_LABELS.get(source, source) for source in scale.breaks
            ]
        assert all(
            type(layer.geom).__name__ not in ("geom_text", "geom_label")
            for layer in plot.layers
        )


def test_comparison_figures_label_if2_warm_starts(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(
        ggplot, "save", lambda self, filename, **kwargs: saved.update({filename.name: self})
    )
    warm_plots.plot_likelihood(
        pd.DataFrame(
            {"Model": warm_plots.METHODS, "logLik": [-4000.0] * len(warm_plots.METHODS)}
        ),
        tmp_path,
    )
    parameters = pd.DataFrame(
        {parameter: [1.0] * len(warm_plots.METHODS) for parameter in warm_plots.PARAMETERS}
    )
    parameters["source"] = warm_plots.METHODS
    warm_plots.plot_parameters(parameters, tmp_path)
    labels = [
        warm_plots.DISPLAY_LABELS.get(source, source) for source in warm_plots.METHODS
    ]
    likelihood = saved["likelihood_if2warm_comparison_r20.png"]
    assert likelihood.scales.get_scales("x").labels == labels
    parameters = saved["parameter_if2warm_comparison_r20.png"]
    for aesthetic in ("color", "fill"):
        assert parameters.scales.get_scales(aesthetic).labels == labels
