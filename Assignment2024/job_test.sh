#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
# source modules.sh

function run_test() {
    test_index=$1       # Test index 
    matrix_index=$2     # Matrix index
    N=$3                # Matrix size
    stopdiff=$4         # Stopping criterion
    P=$5                # Number of MPI processes
    maxiters=$6         # Maximum number of iterations

    mpi_command=$(cat <<EOF
        using MPI
        include("sor_test.jl")

        MPI.Init()
        comm = MPI.Comm_dup(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(comm)

        test_result = SORTEST.runtest($matrix_index, $N, $stopdiff, $maxiters)
        if rank == 0
            if test_result
                println("Test $test_index passed 🥳")
            else
                println("Test $test_index failed 😢")
                println("----------------------------------------")
            end
        end
        MPI.Finalize()
EOF
    )

    mpiexec -np $P julia --project=. -e "$mpi_command"
}

# run_test test_index matrix_index N stopdiff P maxiters
run_test 1 1 10 1e-20 1 1000   
run_test 2 1 10 1e-3 5 1000  
run_test 3 2 100 1e-20 2 1000
run_test 4 2 100 1e-3 4 1000
run_test 5 3 10 1e-20 1 1000
run_test 6 3 10 1e-10 2 1000
run_test 7 4 10 1e-20 10 1000
run_test 8 4 100 1e-3 2 1000
run_test 9 4 100 1e-10 4 1000
run_test 10 1 500 1e-12 10 1000


