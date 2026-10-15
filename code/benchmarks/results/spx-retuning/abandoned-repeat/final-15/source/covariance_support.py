"""Reproduce local covariance support checks without adding a diagonal floor.

PYTHONPATH=code/benchmarks:corenflos:ditlevsen:../pypomp JAX_PLATFORMS=cpu \
python code/benchmarks/covariance_support.py

Ranks use normalized factor columns, avoiding squaring the condition number.
These finite-state diagnostics accompany the algebra in covariance_support.md;
they are not a proof of global hypoellipticity or a path-space limit theorem.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import daphnia as d
from ditlevsen.data import load_dacca_data
from ditlevsen.model import (default_unconstrained_parameters, diffusion, drift,
                             initial_state, decode_parameters, ESTIMATED_PARAMETER_NAMES)
from ditlevsen.transition import ditlevsen_covariance

jax.config.update("jax_enable_x64", True)
OUT = Path(__file__).parent / "results" / "covariance-support"


def support(factor):
    factor = np.asarray(factor)
    norms = np.linalg.norm(factor, axis=0)
    normalized = factor[:, norms > 0] / norms[norms > 0]
    u, s, _ = np.linalg.svd(normalized, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-10))
    projector = u[:, :rank] @ u[:, :rank].T
    return rank, projector, s.tolist()


@jax.jit
def daphnia_factor(x, z):
    p = d.parameters(z, 0)
    h = .25
    b, B = d.drift(x, p), jax.jacfwd(d.drift)(x, p)
    sig = d.coefficients(p)
    G = jnp.diag(sig*x)
    BG, L0G = B@G, jnp.diag(sig*b)
    return jnp.concatenate((jnp.sqrt(h)*(G+.5*h*(BG+L0G)),
                            jnp.sqrt(h**3/12)*(BG-L0G),
                            h/jnp.sqrt(2)*jnp.diag(sig**2*x),
                            jnp.sqrt(h**3/6)*jnp.diag(sig**3*x)), axis=1)


def main():
    OUT.mkdir(exist_ok=True)
    z = jnp.array(d.pack(d.reference.sample_starts(1, 0, jitter_sd=0.)[0]))
    states = {"interior": np.array([2.4,.1,.5,.8,.1,.3,16.,25.]),
              "initial": np.array([2.333,0.,0.,.667,0.,0.,16.667,0.])}
    records = []
    for label, x in states.items():
        factor = np.asarray(daphnia_factor(x, z))
        Q = np.asarray(d.local_moments(jnp.array(x), d.parameters(z, 0))[1])
        np.testing.assert_allclose(factor@factor.T, Q, atol=1e-12, rtol=1e-12)
        rank, projector, s = support(factor / np.asarray(d.SCALES)[:, None])
        changes = []
        for i, name in enumerate(d.NAMES):
            other = z.at[i].add(np.log(1.1))
            r, P, _ = support(daphnia_factor(x, other) / np.asarray(d.SCALES)[:, None])
            changes.append({"parameter": name, "rank": r,
                            "projector_distance": float(np.linalg.norm(P-projector, 2))})
        records.append({"state": label, "values": x.tolist(), "rank": rank,
                        "factor_singular_values": s, "perturbations": changes})
    rng = np.random.default_rng(20261015)
    random_ranks = []
    for _ in range(20):
        x = states["interior"] * np.exp(rng.normal(0,.3,8))
        theta = z + jnp.array(rng.normal(0,.15,len(z)))
        random_ranks.append(support(daphnia_factor(x,theta)/np.asarray(d.SCALES)[:,None])[0])

    data = load_dacca_data(20, max_observations=1)
    theta = jnp.array(default_unconstrained_parameters())
    x = initial_state(theta)
    cov = jnp.array(data.initial_covariates)
    def plane(t):
        g = diffusion(x,t,cov)
        A = jax.jacfwd(drift)(x,t,cov) @ g
        return support(np.column_stack((g,A)))
    rank, P, s = plane(theta)
    assert rank == 2
    rates = decode_parameters(theta)
    e = np.array([-1.,1.,0.,0.,0.,0.])
    v = np.array([0.,-rates["gamma"]-rates["m"],rates["gamma"],0.,0.,rates["m"]])
    np.testing.assert_allclose(support(np.column_stack((e,v)))[1],P,atol=1e-12)
    joint_change = theta.at[0].add(.1).at[1].add(.1)
    np.testing.assert_allclose(plane(joint_change)[1],P,atol=1e-12)
    perturbations = []
    for i,name in enumerate(ESTIMATED_PARAMETER_NAMES):
        candidate = theta.at[i].add(.1)
        r, other, _ = plane(candidate)
        Q = np.asarray(ditlevsen_covariance(x,candidate,cov,1/240))
        residual = float(np.linalg.norm((np.eye(6)-other)@Q) / np.linalg.norm(Q))
        assert residual < 1e-10
        perturbations.append({"parameter": name, "rank": r,
                              "projector_distance": float(np.linalg.norm(other-P,2)),
                              "covariance_outside_plane_relative_norm": residual})
    result = {"daphnia": records, "daphnia_random_interior_ranks": random_ranks,
              "dhaka": {"state": np.asarray(x).tolist(), "rank": rank,
                        "factor_singular_values": s, "perturbations": perturbations},
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__),Path(d.__file__),
                            Path(__file__).parents[2]/"ditlevsen/ditlevsen/transition.py"]}}
    (OUT / "diagnostics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Daphnia:",[(r["state"],r["rank"]) for r in records],
          "random interior ranks:",sorted(set(random_ranks)))
    print("Dhaka plane changes:",[(r["parameter"],round(r["projector_distance"],8))
                                  for r in perturbations if r["projector_distance"] > 1e-8])


if __name__ == "__main__":
    main()
