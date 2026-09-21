"""SIRJPF2 with 22 shared and 2 x 8 unit-specific estimated parameters.

Reference: pypomp/Daphnia-tutorial, f318330, advanced tutorial Section 2.
The observation-interval wrapper changes the random stream, not the Euler
scheme: the first interval has 24 steps and the remaining intervals have 20.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypomp as pp


REFERENCE_COMMIT = "f3183301273a409e5b079bb0ae3a22246f69ba7a"
DATA_PATH = Path(__file__).with_name("data") / "Mesocosmdata.xls"
UNITS = ("K", "L", "M", "N", "O", "P", "Q", "S")
UNIT_PARAMETERS = ("theta_In", "theta_Ii")
FIXED_PARAMETERS = ("sigSn", "sigSi")
DT = 0.25
STATENAMES = (
    "Sn", "In", "Jn", "Si", "Ii", "Ji", "F", "P",
    "T_Sn", "T_In", "T_Si", "T_Ii", "error_count",
)

# Published starting vector, not the later fitted vector used by the tutorial.
PARAMETERS = {
    "ri": 1.307600e4, "rn": 5.904676e1,
    "f_Si": 1.838259e-5, "f_Sn": 1.105668e-3,
    "probi": 3.110083e1, "probn": 2.565626e-1, "xi": 2.865620e1,
    "theta_Sn": 1.479834e-1, "theta_Si": 3.186040e-2,
    "theta_Ii": 3.531879e-1, "theta_In": 5.489315e-1,
    "theta_P": 2.024991e-2, "theta_Ji": 1.299562e-4,
    "theta_Jn": 1.532613e-4, "sigSn": 0.0, "sigSi": 0.0,
    "sigIn": 3.063207e-4, "sigIi": 2.208698e-2,
    "sigJi": 2.727418e-1, "sigJn": 2.836891e-1,
    "sigF": 1.551729e-1, "sigP": 2.385890e-1,
    "k_Ii": 1.241092, "k_In": 1.005756,
    "k_Si": 4.715556, "k_Sn": 4.282648,
}
SHARED_PARAMETERS = tuple(k for k in PARAMETERS if k not in UNIT_PARAMETERS)


def rinit(theta_, key, covars, t0):
    return {
        "Sn": jnp.array(2.333), "In": jnp.array(0.0),
        "Jn": jnp.array(0.0), "Si": jnp.array(0.667),
        "Ii": jnp.array(0.0), "Ji": jnp.array(0.0),
        "F": jnp.array(16.667), "P": jnp.array(0.0),
        "T_Sn": jnp.array(0.0), "T_In": jnp.array(0.0),
        "T_Si": jnp.array(0.0), "T_Ii": jnp.array(0.0),
        "error_count": jnp.array(0.0),
    }


def euler_step(X_, theta_, key, covars, t, dt):
    """The reference Euler-Maruyama step, including its boundary rules."""
    Sn, Jn, In = X_["Sn"], X_["Jn"], X_["In"]
    Si, Ji, Ii = X_["Si"], X_["Ji"], X_["Ii"]
    F, P = X_["F"], X_["P"]
    error_count = X_["error_count"]
    sigSn, sigIn = theta_["sigSn"], theta_["sigIn"]
    sigSi, sigIi = theta_["sigSi"], theta_["sigIi"]
    sigJn, sigJi = theta_["sigJn"], theta_["sigJi"]
    sigF, sigP = theta_["sigF"], theta_["sigP"]
    theta_Sn, theta_In = theta_["theta_Sn"], theta_["theta_In"]
    theta_Si, theta_Ii = theta_["theta_Si"], theta_["theta_Ii"]
    theta_Jn, theta_Ji = theta_["theta_Jn"], theta_["theta_Ji"]
    theta_P = theta_["theta_P"]
    f_Sn, f_Si = theta_["f_Sn"], theta_["f_Si"]
    rn, ri = theta_["rn"], theta_["ri"]
    probn, probi = theta_["probn"], theta_["probi"]
    xi = theta_["xi"]
    delta, mu_food, lambda_J = 0.013, 0.37, 0.1

    keys = jax.random.split(key, 8)
    sqdt = jnp.sqrt(dt)
    noiSn = sigSn * sqdt * jax.random.normal(keys[0])
    noiIn = sigIn * sqdt * jax.random.normal(keys[1])
    noiSi = sigSi * sqdt * jax.random.normal(keys[2])
    noiIi = sigIi * sqdt * jax.random.normal(keys[3])
    noiJn = sigJn * sqdt * jax.random.normal(keys[4])
    noiJi = sigJi * sqdt * jax.random.normal(keys[5])
    noiF = sigF * sqdt * jax.random.normal(keys[6])
    noiP = sigP * sqdt * jax.random.normal(keys[7])

    Sn_term = (lambda_J * Jn * dt - theta_Sn * Sn * dt
               - probn * f_Sn * Sn * P * dt - delta * Sn * dt + Sn * noiSn)
    Jn_term = (rn * f_Sn * F * Sn * dt - lambda_J * Jn * dt
               - theta_Jn * Jn * dt - delta * Jn * dt + Jn * noiJn)
    In_term = (probn * f_Sn * Sn * P * dt
               - theta_In * In * dt - delta * In * dt + In * noiIn)
    Si_term = (lambda_J * Ji * dt - theta_Si * Si * dt
               - probi * f_Si * Si * P * dt - delta * Si * dt + Si * noiSi)
    Ji_term = (ri * f_Si * F * Si * dt - lambda_J * Ji * dt
               - theta_Ji * Ji * dt - delta * Ji * dt + Ji * noiJi)
    Ii_term = (probi * f_Si * Si * P * dt
               - theta_Ii * Ii * dt - delta * Ii * dt + Ii * noiIi)
    F_term = (-f_Sn * F * (Sn + xi * In + Jn) * dt
              - f_Si * F * (Si + xi * Ii + Ji) * dt
              - delta * F * dt + mu_food * dt + F * noiF)
    P_term = (30.0 * theta_In * In * dt + 30.0 * theta_Ii * Ii * dt
              - f_Sn * (Sn + xi * In) * P * dt
              - f_Si * (Si + xi * Ii) * P * dt
              - theta_P * P * dt - delta * P * dt + P * noiP)

    Sn_new, In_new, Jn_new = Sn + Sn_term, In + In_term, Jn + Jn_term
    Si_new, Ii_new, Ji_new = Si + Si_term, Ii + Ii_term, Ji + Ji_term
    F_new, P_new = F + F_term, P + P_term
    inoculate = (t <= 4.0) & ((t + dt) > 4.0)
    P_new = P_new + jnp.where(inoculate, 25.0, 0.0)

    def viol(x, hi):
        return (x < 0.0) | (x > hi)

    eps = 0.0
    eps += jnp.where(viol(Sn_new, 1e5), 1.0, 0.0)
    eps += jnp.where(viol(Si_new, 1e5), 1e6, 0.0)
    eps += jnp.where(viol(F_new, 1e20), 1e3, 0.0)
    eps += jnp.where(viol(In_new, 1e5), 1e-3, 0.0)
    eps += jnp.where(viol(Ii_new, 1e5), 1e-9, 0.0)
    eps += jnp.where(viol(Jn_new, 1e5), 1e-3, 0.0)
    eps += jnp.where(viol(Ji_new, 1e5), 1e-9, 0.0)
    eps += jnp.where(viol(P_new, 1e20) & (t > 3.9), 1e-6, 0.0)
    Sn_new = jnp.where(viol(Sn_new, 1e5), 0.0, Sn_new)
    In_new = jnp.where(viol(In_new, 1e5), 0.0, In_new)
    Jn_new = jnp.where(viol(Jn_new, 1e5), 0.0, Jn_new)
    Si_new = jnp.where(viol(Si_new, 1e5), 0.0, Si_new)
    Ii_new = jnp.where(viol(Ii_new, 1e5), 0.0, Ii_new)
    Ji_new = jnp.where(viol(Ji_new, 1e5), 0.0, Ji_new)
    F_new = jnp.where(viol(F_new, 1e20), 0.0, F_new)
    P_new = jnp.where(viol(P_new, 1e20) & (t > 3.9), 0.0, P_new)
    return {
        "Sn": Sn_new, "In": In_new, "Jn": Jn_new,
        "Si": Si_new, "Ii": Ii_new, "Ji": Ji_new,
        "F": F_new, "P": P_new,
        "T_Sn": jnp.abs(Sn_new), "T_In": jnp.abs(In_new),
        "T_Si": jnp.abs(Si_new), "T_Ii": jnp.abs(Ii_new),
        "error_count": error_count + eps,
    }


def rproc(X_, theta_, key, covars, t, dt):
    """Integrate one observation interval with a fixed, differentiable loop."""
    nstep = jnp.rint(dt / DT).astype(jnp.int32)

    def advance(i, carry):
        state, step_key = carry
        next_key, subkey = jax.random.split(step_key)
        state = euler_step(state, theta_, subkey, covars, t + i * DT, DT)
        return state, next_key

    def body(i, carry):
        return jax.lax.cond(i < nstep, lambda x: advance(i, x), lambda x: x, carry)

    return jax.lax.fori_loop(0, 24, body, (X_, key))[0]


def nb_logpmf(y, mu, size):
    mu, size = jnp.maximum(mu, 1e-10), jnp.maximum(size, 1e-10)
    return (jax.scipy.special.gammaln(y + size)
            - jax.scipy.special.gammaln(size)
            - jax.scipy.special.gammaln(y + 1.0)
            + size * jnp.log(size / (size + mu))
            + y * jnp.log(mu / (size + mu)))


def dmeas(Y_, X_, theta_, covars, t):
    ll = (nb_logpmf(Y_["dentadult"], X_["T_Sn"], theta_["k_Sn"])
          + nb_logpmf(Y_["dentinf"], X_["T_In"], theta_["k_In"])
          + nb_logpmf(Y_["lumadult"], X_["T_Si"], theta_["k_Si"])
          + nb_logpmf(Y_["luminf"], X_["T_Ii"], theta_["k_Ii"]))
    return jnp.where(X_["error_count"] > 0.0, -150.0, ll)


def to_est(theta):
    return {k: v if k in FIXED_PARAMETERS else jnp.log(jnp.maximum(v, 1e-30))
            for k, v in theta.items()}


def from_est(theta):
    return {k: v if k in FIXED_PARAMETERS else jnp.exp(v) for k, v in theta.items()}


def sample_starts(n, seed, jitter_sd=0.45):
    """Return independent PanelParameters payloads around the published vector."""
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(n):
        shared = pd.DataFrame({"shared": [PARAMETERS[k] for k in SHARED_PARAMETERS]},
                              index=SHARED_PARAMETERS)
        specific = pd.DataFrame({u: [PARAMETERS[k] for k in UNIT_PARAMETERS]
                                 for u in UNITS}, index=UNIT_PARAMETERS)
        for name in shared.index:
            if name not in FIXED_PARAMETERS:
                shared.loc[name, "shared"] *= np.exp(rng.normal(0.0, jitter_sd))
        for name in specific.index:
            specific.loc[name] *= np.exp(rng.normal(0.0, jitter_sd, len(UNITS)))
        starts.append({"shared": shared, "unit_specific": specific})
    return starts


@lru_cache(maxsize=2)
def _unit_models(data_path):
    """Reuse immutable model functions so batches share JAX compilations."""
    data = pd.read_excel(data_path, sheet_name="both species combined")
    data = data.iloc[90:170].copy()
    data.columns = data.columns.str.strip()
    data["day"] = (data["day"] - 1) * 5 + 7
    columns = {"dent.adult": "dentadult", "dent.inf": "dentinf",
               "lum.adult": "lumadult", "lum.adult.inf": "luminf"}
    units = {}
    for unit in UNITS:
        ys = (data.loc[data["rep"] == unit].sort_values("day")
              .set_index("day")[list(columns)].rename(columns=columns).astype(float))
        if not np.array_equal(ys.index.to_numpy(), np.arange(7, 53, 5)):
            raise ValueError(f"Unexpected observation times for unit {unit}")
        units[unit] = pp.Pomp(
            ys=ys, theta=pp.PompParameters(PARAMETERS), statenames=list(STATENAMES),
            t0=1.0, rinit=rinit, rproc=rproc, dmeas=dmeas, nstep=1,
            par_trans=pp.ParTrans(to_est=to_est, from_est=from_est),
            accumvars=("error_count",),
        )
    return units


def build_panel(starts=None, data_path=None):
    """Build a new panel estimate using the same eight unit models."""
    units = _unit_models(Path(data_path or DATA_PATH).resolve())
    theta = sample_starts(1, 0, jitter_sd=0.0) if starts is None else starts
    return pp.PanelPomp(pomp_dict=dict(units), theta=pp.PanelParameters(theta))
