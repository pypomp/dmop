#!/bin/bash

# Define the alpha values to run
alphas=(0 0.97 1)

# Loop through and submit each job
for alpha in "${alphas[@]}"; do
    echo "Submitting job for alpha=$alpha"
    sbatch --job-name="dmop_alpha_$alpha" --export=ALL,ALPHA="$alpha" dmop_search.sbat
done
