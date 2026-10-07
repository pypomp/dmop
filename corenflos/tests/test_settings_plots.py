import json

import pandas as pd
import pytest
from plotnine import ggplot

from corenflos import settings_plots as s
from corenflos import warm_plots as w


def test_particle_settings_are_shared_within_method_not_selected_by_initialization():
    args, labels = s.sources_for()
    assert args.corenflos_cold == s.LOW["corenflos_cold"]
    assert args.corenflos_warm == s.LOW["corenflos_warm"]
    assert args.ditlevsen_cold == s.HIGH / "ditlevsen_vanilla"
    assert args.ditlevsen_warm == s.HIGH / "ditlevsen_warm"
    assert labels["Corenflos"] == "Corenflos (J=100)"
    assert labels["Corenflos + IF2"] == "Corenflos + IF2 warm start (J=100)"
    assert labels["Ditlevsen"] == "Ditlevsen (J=1,000)"
    assert labels["Ditlevsen + IF2"] == "Ditlevsen + IF2 warm start (J=1,000)"
    with pytest.raises(ValueError, match="Complete standalone and warm-start"):
        s.sources_for(100, 5000)


def test_all_output_view_keeps_outliers_and_settings_labels(monkeypatch, tmp_path):
    _, labels = s.sources_for()
    frame = pd.DataFrame({"Model": w.METHODS * 2,
                          "logLik": [-4000.] * 6 + [-25000., -3980., -24000., -3970., -4000., -3770.]})
    original = frame.copy(deep=True)
    saved = {}
    monkeypatch.setattr(ggplot, "save", lambda self, filename, **kwargs: saved.update({filename.name: self}))
    w.plot_likelihood(frame, tmp_path, display_labels=labels, minimum=None,
                      limits=w.OVERVIEW_LIMITS, save_pdf=True, best_known=None)
    assert set(saved) == {"likelihood_if2warm_comparison_r20.png", "likelihood_if2warm_comparison_r20.pdf"}
    plot = saved["likelihood_if2warm_comparison_r20.png"]
    assert len(plot.data) == len(frame)
    assert plot.data.logLik.min() == -25000
    assert plot.coordinates.limits.y == w.OVERVIEW_LIMITS
    assert plot.scales.get_scales("x").labels == [labels.get(method, method) for method in w.METHODS]
    assert plot.labels.caption is None and plot.labels.title is None
    assert all(type(layer.geom).__name__ not in ("geom_text", "geom_label") for layer in plot.layers)
    pd.testing.assert_frame_equal(frame, original)


def test_uncropped_likelihood_has_no_coordinate_cutoff(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(ggplot, "save", lambda self, filename, **kwargs: saved.append(self))
    w.plot_likelihood(pd.DataFrame({"Model": ["Corenflos"], "logLik": [-25000.]}),
                      tmp_path, minimum=None, best_known=None)
    assert saved[0].coordinates.limits.y is None
    assert saved[0].data.logLik.tolist() == [-25000.]


def test_validate_runs_rejects_wrong_particles_and_missing_start(tmp_path):
    (tmp_path / "configuration.json").write_text(json.dumps({"particles": 100, "starts": 100}))
    with pytest.raises(ValueError, match="Unexpected fitting settings"):
        s.validate_runs(tmp_path, 1000)
    pd.DataFrame({"start": range(99)}).to_csv(tmp_path / "fit_summary.csv", index=False)
    pd.DataFrame({"start": range(100)}).to_csv(tmp_path / "final_evaluations.csv", index=False)
    with pytest.raises(ValueError, match="100 unique starts"):
        s.validate_runs(tmp_path, 100)


def test_objective_mismatch_labels_include_family_particle_setting(monkeypatch, tmp_path):
    _, labels = s.sources_for()
    frame = pd.DataFrame({"source": ["Corenflos + IF2 warm start"] * 2 + ["Ditlevsen + IF2 warm start"] * 2,
                          "training_delta": [1., 2., 15., 17.], "euler_delta": [0., 0., -1., -2.]})
    saved = []
    monkeypatch.setattr(ggplot, "save", lambda self, filename, **kwargs: saved.append(self))
    w.plot_objective_mismatch(frame, tmp_path, display_labels=labels, save_pdf=True)
    assert len(saved) == 4
    for plot in saved:
        assert set(plot.data.source) == {labels["Corenflos + IF2"], labels["Ditlevsen + IF2"]}
        assert plot.labels.caption is None and plot.labels.title is None
        assert all(type(layer.geom).__name__ not in ("geom_text", "geom_label") for layer in plot.layers)
