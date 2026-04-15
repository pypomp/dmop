import os

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pypomp as pp  # noqa: E402
import session_info

print(jax.devices())

session_info.show(dependencies=True)

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")

DEFAULT_SD = 0.02 / 8
RW_SD = pp.RWSigma(
    sigmas={
        "gamma": DEFAULT_SD,
        "m": DEFAULT_SD,
        "rho": 0.0,  # fixed in model code
        "epsilon": DEFAULT_SD,
        "omega": 0.0,  # fixed in model code
        "c": 0.0,  # fixed in model code
        "beta_trend": DEFAULT_SD,
        "sigma": DEFAULT_SD,
        "tau": DEFAULT_SD,
        "bs1": DEFAULT_SD,
        "bs2": DEFAULT_SD,
        "bs3": DEFAULT_SD,
        "bs4": DEFAULT_SD,
        "bs5": DEFAULT_SD,
        "bs6": DEFAULT_SD,
        "omegas1": DEFAULT_SD,
        "omegas2": DEFAULT_SD,
        "omegas3": DEFAULT_SD,
        "omegas4": DEFAULT_SD,
        "omegas5": DEFAULT_SD,
        "omegas6": DEFAULT_SD,
    },
    init_names=[],
)
COOLING_RATE = 0.5

dacca_obj = pp.dacca(dt=None, nstep=20)

params_box = {k: (v * 0.9, v * 1.1) for k, v in dacca_obj.theta[0].items()}
params_box["rho"] = (0.0, 0.0)
# This is the params box from diffPomp
# params_box = {
#     "gamma": (10.0, 40.0),
#     "m": (0.03, 0.60),
#     "rho": (0.0, 0.0),
#     "epsilon": (0.20, 30.0),
#     "omega": (float(jnp.exp(-4.5)), float(jnp.exp(-4.5))),
#     "c": (1.0, 1.0),
#     "beta_trend": (-0.01, 0.00),
#     "sigma": (1.0, 5.0),
#     "tau": (0.10, 0.50),
#     "bs1": (-4.0, 4.0),
#     "bs2": (0.0, 8.0),
#     "bs3": (-4.0, 4.0),
#     "bs4": (0.0, 8.0),
#     "bs5": (0.0, 8.0),
#     "bs6": (0.0, 8.0),
#     "omegas1": (-10.0, 0.0),
#     "omegas2": (-10.0, 0.0),
#     "omegas3": (-10.0, 0.0),
#     "omegas4": (-10.0, 0.0),
#     "omegas5": (-10.0, 0.0),
#     "omegas6": (-10.0, 0.0),
# }

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(params_box, NREPS_FITR, key=subkey)
