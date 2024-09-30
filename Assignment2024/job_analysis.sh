#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
# source modules.sh

MPIFLAGS="--map-by node"  # Removed --rank-by core
JULIAFLAGS="-O3 --check-bounds=no"

# Set parameters
# Note that the matrix size must be divisible by the number of processors i.e. mod(N, P) = 0
N_values=(100 200 300 400)      # Matrix sizes
P_values=(1 2 4 5 8)                # Number of Processors
stopdiff=1e-20                      # Stopping Criterion
maxiters=1000                       # Maximum Iteration
matrix_id=3                         # Test Matrix Index   

# Generate results for sequential and parallel experiments
for N in "${N_values[@]}"; do
    for P in "${P_values[@]}"; do
        echo "[Running with N=$N and P=$P]"
        mpiexec -np $P $MPIFLAGS julia $JULIAFLAGS --project=. -e "
            include(\"sor_analysis.jl\")
            
            params = Dict{String,Any}(
                \"matrix_id\" => $matrix_id,
                \"N\" => $N,
                \"stopdiff\" => $stopdiff,
                \"maxiters\" => $maxiters
            )

            SORANALYSIS.experiment(params)
        "
    done
done