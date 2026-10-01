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
    LONG,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR_0 = (2, 5, 100, 60)[RUN_LEVEL - 1]
NFITR_97 = (2, 5, 100, 175)[RUN_LEVEL - 1]
NFITR_1 = (2, 5, 100, 60)[RUN_LEVEL - 1]
# The gradient step gained less from the Euler step optimization than the
# filtering step did: about 1.6x against 2.5x to 3x. IFAD nevertheless ended up
# at roughly half the wall clock of the IF2-only runs, because those had their
# iteration count tripled to hold their budget fixed while these were left
# alone, so IFAD simply banked its speedup as a shorter run. The gradient
# counts below spend it instead, sized from the measured per-iteration cost of
# the train step (about 2.1 s to 2.2 s at J=5000) to approach the 946 s of the
# IF2-only run. Measured totals are 948.8 s for IFAD-0, 942.8 s for IFAD-0.97
# and 936.6 s for IFAD-1, so IFAD-0 overshoots by 2.8 s, or 0.3%. That is well
# inside the "similar amount of time" the comparison claims, and IFAD-0 is the
# weakest variant, so no result depends on trimming it further.
#
# Only the gradient counts grow. The IF2 warm-start counts are deliberately
# left alone, because lengthening the warm start pushes the swarm off the ridge
# that the gradient stage needs to start from.
NTRAIN_0 = (2, 20, 40, 410)[RUN_LEVEL - 1]
NTRAIN_97 = (2, 20, 40, 400)[RUN_LEVEL - 1]
NTRAIN_1 = (2, 20, 40, 425)[RUN_LEVEL - 1]
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

if LONG:
    M_train *= 4


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
    rw_sd=RW_SD.geometric_cooling(0.5),
    M=M_mif,
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

suffix = "_long" if LONG else ""
with open(
    f"dmop_results/dacca_results_rl{RUN_LEVEL}_alpha{ALPHA}_nm{N_MONITORS}{suffix}.pkl", "wb"
) as f:
    pickle.dump(dacca_obj, f)
