# R script to benchmark mif2 on the built-in dacca model

library(pomp)

set.seed(631409)

po <- dacca()

params <- coef(po)
param_names <- names(params)

ivps <- param_names[grep("_0$", param_names)]
regular_params <- setdiff(param_names, ivps)

sd_val <- 0.02
ivp_sd_val <- 0.16

rw_list <- list()
for (p in regular_params) {
  rw_list[[p]] <- sd_val
}

for (p in ivps) {
  rw_list[[p]] <- call("ivp", ivp_sd_val)
}

rw_sd_func <- if (exists("rw_sd", envir = asNamespace("pomp"))) {
  get("rw_sd", envir = asNamespace("pomp"))
} else {
  get("rw.sd", envir = asNamespace("pomp"))
}

rw_spec <- do.call(rw_sd_func, rw_list)

start_time <- Sys.time()

fit <- mif2(
  po,
  Nmif = 60,
  Np = 5000,
  cooling.fraction.50 = 0.5,
  rw.sd = rw_spec
)

end_time <- Sys.time()

elapsed <- as.numeric(end_time - start_time, units = "secs")

cat(sprintf("Time taken: %.3f seconds\n", elapsed))
