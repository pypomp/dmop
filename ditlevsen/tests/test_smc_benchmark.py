from argparse import Namespace

import numpy as np

from ditlevsen.smc_benchmark import _assigned_starts, _checkpoint_iterations


def test_start_shards_partition_indices():
    shards = [
        list(_assigned_starts(Namespace(starts=10, workers=3, worker_index=index)))
        for index in range(3)
    ]
    assert sorted(value for shard in shards for value in shard) == list(range(10))


def test_elapsed_checkpoints_include_endpoints_without_duplicates():
    elapsed = np.asarray([1.0, 40.0, 101.0, 205.0, 206.0])
    assert _checkpoint_iterations(elapsed, 100.0) == [0, 2, 3, 4]
