from argparse import Namespace

import numpy as np

from ditlevsen.particle_sweep import (
    focused_j5000_configurations,
    sweep_configurations,
)
from ditlevsen.smc_benchmark import (
    _assigned_starts,
    _checkpoint_iterations,
    _output_iteration,
)


def test_start_shards_partition_indices():
    shards = [
        list(_assigned_starts(Namespace(starts=10, workers=3, worker_index=index)))
        for index in range(3)
    ]
    assert sorted(value for shard in shards for value in shard) == list(range(10))


def test_elapsed_checkpoints_include_endpoints_without_duplicates():
    elapsed = np.asarray([1.0, 40.0, 101.0, 205.0, 206.0])
    assert _checkpoint_iterations(elapsed, 100.0) == [0, 2, 3, 4]
    assert _checkpoint_iterations(elapsed, 100.0, every_update=True) == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_output_iteration_uses_terminal_or_first_time_crossing():
    elapsed = np.asarray([15.0, 95.0, 207.0, 715.0, 809.0])
    assert _output_iteration(elapsed, None) == 4
    assert _output_iteration(elapsed, 700.0) == 3
    assert _output_iteration(elapsed, 900.0) == 4


def test_particle_sweep_spans_particle_counts_and_unique_rates():
    configurations = sweep_configurations()
    assert {item.particles for item in configurations} == {100, 500, 1000, 5000}
    assert len(configurations) == 8
    assert len({item.label for item in configurations}) == len(configurations)


def test_focused_sweep_keeps_particle_count_and_labels_schedules():
    configurations = focused_j5000_configurations()
    assert {item.particles for item in configurations} == {5000}
    assert len(configurations) == 4
    assert len({item.label for item in configurations}) == len(configurations)
    assert {item.burnin for item in configurations} == {5, 15, 30}
