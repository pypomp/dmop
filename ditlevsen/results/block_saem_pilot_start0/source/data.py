"""Load the exact data and interpolated covariates used by Pypomp's Dacca model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DaccaData:
    """Dacca observations and covariates on a requested latent-time grid.

    ``step_covariates[k, j]`` contains covariates at the left endpoint of
    substep ``j`` in observation interval ``k``, matching Pypomp's Euler
    implementation.  ``observation_covariates[k]`` contains covariates at the
    measurement time.  Covariate columns are trend, population derivative,
    population, and the six seasonal spline basis values.
    """

    observations: np.ndarray
    observation_times: np.ndarray
    observation_covariates: np.ndarray
    step_times: np.ndarray
    step_covariates: np.ndarray
    initial_covariates: np.ndarray
    nstep: int


def _default_data_dir() -> Path:
    here = Path(__file__).resolve()
    workspace = here.parents[3]
    return workspace / "pypomp" / "pypomp" / "data" / "dacca"


def load_dacca_data(
    nstep: int,
    *,
    max_observations: int | None = None,
    data_dir: str | Path | None = None,
) -> DaccaData:
    """Load observations and linearly interpolate the original covariates.

    The model time unit is a year.  Thus, ``nstep=20`` gives a step of
    ``1 / 240`` years, i.e. one twentieth of a month, exactly as in the DMOP
    manuscript and its Pypomp implementation.
    """

    if nstep < 1:
        raise ValueError("nstep must be a positive integer")

    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    obs_frame = pd.read_csv(root / "dacca.csv")
    cov_frame = pd.read_csv(root / "covars.csv")
    cov_time_frame = pd.read_csv(root / "covart.csv")

    observation_times = obs_frame["time"].to_numpy(dtype=float)
    observations = obs_frame["cholera.deaths"].to_numpy(dtype=float)
    if max_observations is not None:
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        observation_times = observation_times[:max_observations]
        observations = observations[:max_observations]

    # The first column of the two covariate files is an R row-name column.
    covariate_times = cov_time_frame.iloc[:, 1].to_numpy(dtype=float)
    covariates = cov_frame.iloc[:, 1:].to_numpy(dtype=float)
    if covariates.shape[1] != 9:
        raise ValueError(f"expected 9 Dacca covariates, found {covariates.shape[1]}")

    t0 = 1891.0
    interval_starts = np.concatenate(([t0], observation_times[:-1]))
    # Pypomp supplies the forcing at the left endpoint of every Euler step.
    # In particular, with nstep=1 the first month's process uses the 1891.0
    # covariates and the measurement uses the covariates at 1891 + 1/12.
    fractions = np.arange(nstep, dtype=float) / nstep
    step_times = interval_starts[:, None] + (
        observation_times - interval_starts
    )[:, None] * fractions[None, :]

    flat_times = step_times.reshape(-1)
    flat_covariates = np.column_stack(
        [
            np.interp(flat_times, covariate_times, covariates[:, column])
            for column in range(covariates.shape[1])
        ]
    )
    step_covariates = flat_covariates.reshape(
        len(observations), nstep, covariates.shape[1]
    )
    initial_covariates = np.array(
        [
            np.interp(t0, covariate_times, covariates[:, column])
            for column in range(covariates.shape[1])
        ]
    )
    observation_covariates = np.column_stack(
        [
            np.interp(observation_times, covariate_times, covariates[:, column])
            for column in range(covariates.shape[1])
        ]
    )

    return DaccaData(
        observations=observations,
        observation_times=observation_times,
        observation_covariates=observation_covariates,
        step_times=step_times,
        step_covariates=step_covariates,
        initial_covariates=initial_covariates,
        nstep=nstep,
    )
