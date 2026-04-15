import pickle

from prep import (
    RUN_LEVEL,
    dacca_obj,
    initial_params_list,
    key,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 100)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.2
eta = {
    "gamma": DEFAULT_ETA,
    "epsilon": DEFAULT_ETA,
    "rho": 0.0,
    "m": DEFAULT_ETA,
    "c": 0.0,
    "beta_trend": DEFAULT_ETA,
    **{f"bs{i + 1}": DEFAULT_ETA for i in range(6)},
    "sigma": DEFAULT_ETA,
    "tau": DEFAULT_ETA,
    "omega": DEFAULT_ETA,
    **{f"omegas{i + 1}": DEFAULT_ETA for i in range(6)},
}

# Train step
dacca_obj.train(
    J=NP_FITR,
    M=NTRAIN,
    theta=initial_params_list,
    eta=eta,
    optimizer="Adam",
    n_monitors=1,
    key=key,
    eta_cooling=0.05,
)
print(dacca_obj.results())

# # PFILTER round 2
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Re-evaluate top fit to account for sample max luck
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

with open(f"dmop_results/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
