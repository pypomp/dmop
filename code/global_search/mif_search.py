import pickle

from prep import (
    RUN_LEVEL,
    RW_SD,
    dacca_obj,
    initial_params_list,
    key,
)

NP_FITR = (2, 500, 1000, 6000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 550)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]


dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    a=0.8,
    J=NP_FITR,
    key=key,
)

dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

# Save results
with open(f"mif_results/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
