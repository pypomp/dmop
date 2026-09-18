import numpy as np

from ditlevsen.warm_starts import (
    extract_ifad_mif_starts,
    load_starts_file,
    write_ifad_mif_starts,
)


def test_extracts_exact_comparable_ifad_097_boundary() -> None:
    starts, metadata = extract_ifad_mif_starts()

    assert starts.shape == (100, 23)
    assert np.isfinite(starts).all()
    assert metadata["source_method"] == "IFAD-0.97"
    assert metadata["source_effort"] == "comparable"
    assert metadata["if2_particles"] == 5000
    assert metadata["if2_iterations"] == 175
    assert metadata["gradient_iterations"] == 175
    np.testing.assert_allclose(
        metadata["total_elapsed_seconds"],
        metadata["if2_elapsed_seconds"] + metadata["remaining_elapsed_seconds"],
    )


def test_round_trips_start_file(tmp_path) -> None:
    path = tmp_path / "starts.npz"
    write_ifad_mif_starts(path)
    starts = load_starts_file(path, 3)

    assert len(starts) == 3
    assert all(start.shape == (23,) for start in starts)
    assert path.with_suffix(".json").exists()
