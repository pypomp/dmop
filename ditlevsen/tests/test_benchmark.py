import numpy as np

from ditlevsen.benchmark import build_parser, make_starts
from ditlevsen.model import default_unconstrained_parameters


def test_benchmark_keeps_inference_and_evaluation_grids_distinct():
    args = build_parser().parse_args(["--nsteps", "1", "5", "20"])
    assert args.nsteps == [1, 5, 20]
    assert args.evaluation_nstep == 20


def test_default_grid_includes_measurement_scale():
    args = build_parser().parse_args([])
    assert args.nsteps == [1, 2, 5, 10, 20]


def test_every_search_start_is_a_reproducible_box_draw():
    starts = make_starts(3, seed=12)
    repeated = make_starts(3, seed=12)
    assert len(starts) == 3
    for left, right in zip(starts, repeated):
        np.testing.assert_allclose(left, right)
    assert not np.allclose(starts[0], default_unconstrained_parameters())
