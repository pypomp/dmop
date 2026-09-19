import pickle

from prep import (
    RUN_LEVEL,
    RW_SD,
    dacca_obj,
    initial_params_list,
    key,
    N_MONITORS,
    LONG,
)

# An Euler step optimization in pypomp cut the cost of an IF2 iteration on this
# model by a measured 2.96x relative to the 0.4.6.0 runs reported in the
# manuscript (1.368 s/iter -> 0.463 s/iter at J=5000, N_MONITORS=0, on the same
# gpu-rtx6000 partition). The IF2-only runs therefore get SPEEDUP times as many
# iterations, which keeps their original wall-clock budget of about 15 minutes.
# Geometric cooling scales the random walk sigmas by a ** (m / 50) at iteration
# m, so the run ends at a total cooling of a ** (M / 50). Taking the SPEEDUP-th
# root of a alongside the SPEEDUP-fold increase in M leaves that end-of-run
# cooling unchanged, and only traverses the same schedule on a finer grid.
SPEEDUP = 3
a = 0.8 ** (1 / SPEEDUP)
NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 650 * SPEEDUP)[RUN_LEVEL - 1]
if LONG:
    NFITR *= 4
    a = 0.9 ** (1 / SPEEDUP)
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]


dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD.geometric_cooling(a),
    M=NFITR,
    J=NP_FITR,
    key=key,
    n_monitors=N_MONITORS,
)

dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

# Save results
suffix = "_long" if LONG else ""
with open(
    f"mif_results/dacca_results_rl{RUN_LEVEL}_nm{N_MONITORS}{suffix}.pkl", "wb"
) as f:
    pickle.dump(dacca_obj, f)
