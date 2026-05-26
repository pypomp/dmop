import pickle
import numpy as np
import pypomp as pp

from prep import (
    RUN_LEVEL,
    dacca_obj,
    initial_params_list,
    key,
    RW_SD,
    ALPHA,
    N_MONITORS,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR_0 = (2, 5, 100, 60)[RUN_LEVEL - 1]
NFITR_97 = (2, 5, 100, 175)[RUN_LEVEL - 1]
NFITR_1 = (2, 5, 100, 60)[RUN_LEVEL - 1]
NTRAIN_0 = (2, 20, 40, 225)[RUN_LEVEL - 1]
NTRAIN_97 = (2, 20, 40, 175)[RUN_LEVEL - 1]
NTRAIN_1 = (2, 20, 40, 225)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
warmup = (1, 5, 10, 10)[RUN_LEVEL - 1]

M_mif = -1
M_train = -1
beta1 = -1.0
match ALPHA:
    case 0.0:
        M_train = NTRAIN_0
        M_mif = NFITR_0
        beta1 = 0.0
    case 0.97:
        M_train = NTRAIN_97
        M_mif = NFITR_97
        beta1 = 0.9
    case 1.0:
        M_train = NTRAIN_1
        M_mif = NFITR_1
        beta1 = 0.9


def w(v):
    if v == 0.0:
        return 0.0
    return np.concatenate(
        [np.linspace(v * 0.1, v, warmup), np.full(M_train - warmup, v)]
    )


if ALPHA == 0.0:
    DEFAULT_ETA = 0.1  # Tiny learning rate to prevent exploding noisy gradients
else:
    DEFAULT_ETA = 0.1
DEFAULT_IVP_ETA = DEFAULT_ETA / 2
eta = {
    "gamma": w(DEFAULT_ETA * 0.5),
    "epsilon": w(DEFAULT_ETA),
    "rho": 0.0,
    "m": w(DEFAULT_ETA),
    "c": 0.0,
    "alpha": 0.0,
    "delta": 0.0,
    "beta_trend": w(DEFAULT_ETA * 0.5),
    **{f"bs{i + 1}": w(DEFAULT_ETA) for i in range(6)},
    "sigma": w(DEFAULT_ETA * 0.5),
    "tau": w(DEFAULT_ETA * 0.5),
    **{f"omegas{i + 1}": w(DEFAULT_ETA) for i in range(6)},
    "S_0": w(DEFAULT_IVP_ETA),
    "I_0": w(DEFAULT_IVP_ETA),
    "Y_0": 0.0,
    "R1_0": w(DEFAULT_IVP_ETA),
    "R2_0": w(DEFAULT_IVP_ETA),
    "R3_0": w(DEFAULT_IVP_ETA),
}

dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=M_mif,
    a=0.5,
    J=NP_FITR,
    key=key,
    n_monitors=N_MONITORS,
)
print(dacca_obj.results())

dacca_obj.train(
    J=NP_FITR,
    M=M_train,
    eta=pp.LearningRate(eta).cosine_decay(final_factor=0.05, M=M_train),
    alpha=ALPHA,
    optimizer=pp.Adam(beta1=beta1),
    n_monitors=N_MONITORS,
)
print(dacca_obj.results())

dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

with open(
    f"dmop_results/dacca_results_rl{RUN_LEVEL}_alpha{ALPHA}_nm{N_MONITORS}.pkl", "wb"
) as f:
    pickle.dump(dacca_obj, f)
