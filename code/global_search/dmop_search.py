import pickle

from prep import (
    RUN_LEVEL,
    dacca_obj,
    initial_params_list,
    key,
    RW_SD,
    COOLING_RATE,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 200)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 100)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.2
DEFAULT_IVP_ETA = DEFAULT_ETA
eta = {
    "gamma": DEFAULT_ETA,
    "epsilon": DEFAULT_ETA,
    "rho": 0.0,
    "m": DEFAULT_ETA,
    "c": 0.0,
    "alpha": 0.0,
    "delta": 0.0,
    "beta_trend": DEFAULT_ETA,
    **{f"bs{i + 1}": DEFAULT_ETA for i in range(6)},
    "sigma": DEFAULT_ETA,
    "tau": DEFAULT_ETA,
    **{f"omegas{i + 1}": DEFAULT_ETA for i in range(6)},
    "S_0": DEFAULT_IVP_ETA,
    "I_0": DEFAULT_IVP_ETA,
    "Y_0": 0.0,
    "R1_0": DEFAULT_IVP_ETA,
    "R2_0": DEFAULT_IVP_ETA,
    "R3_0": DEFAULT_IVP_ETA,
}

dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=key,
)
print(dacca_obj.results())

dacca_obj.train(
    J=NP_FITR,
    M=NTRAIN,
    eta=eta,
    optimizer="Adam",
    n_monitors=1,
    eta_cooling=0.05,
)
print(dacca_obj.results())

dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

with open(f"dmop_results/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
