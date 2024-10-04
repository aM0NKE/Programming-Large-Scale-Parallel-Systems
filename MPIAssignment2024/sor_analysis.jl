module SORANALYSIS

using JSON
using MPI

MPI.Init()

include("sor_seq.jl")
include("sor_par.jl")
include("sor_test.jl")

# Number of runs per test 
const nruns = 5

function case_name(args...)
    name = String(sprint(show, hash(args))[3:end])
end

function experiment(params)
    # Recover params
    matrix_id = params["matrix_id"]
    N = params["N"]
    stopdiff = params["stopdiff"]
    maxiters = params["maxiters"]
    test_matrix = SORTEST.test_matrices[matrix_id]
    name = case_name(matrix_id, N, stopdiff, maxiters)

    # Initialize MPI
    comm = MPI.Comm_dup(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    if nranks == 1
        # Sequential run
        A = test_matrix(N)
        timings = zeros(nruns)
        for irun in 1:nruns
            println("Sequential run $irun of $nruns")
            timings[irun] = @elapsed begin
                actual_diff, actual_iters = sor_seq!(A, stopdiff, maxiters)
            end
        end
        results = copy(params)
        results["P"] = 1
        results["timings"] = timings
        results["min_timings"] = minimum(timings)
        results["actual_iters"] = actual_iters
        results["actual_diff"] = actual_diff
        open("$(name)_seq.json", "w") do f
            JSON.print(f, results)
        end
    else
        # MPI Parallel run
        if rank == 0
            A = test_matrix(N)
        else
            A = test_matrix(0)
        end
        timings = zeros(nruns)
        actual_iters = 0
        actual_diff = 0.0
        for irun in 1:nruns
            if rank == 0
                println("MPI run $irun of $nruns")
            end
            MPI.Barrier(comm)
            timings[irun] = @elapsed begin
                actual_diff, actual_iters = sor_par!(A, stopdiff, maxiters, comm)
            end
        end
        if rank == 0
            results = copy(params)
            results["P"] = nranks
            results["timings"] = timings
            results["min_timings"] = minimum(timings)
            results["actual_iters"] = actual_iters
            results["actual_diff"] = actual_diff
            open("$(name)_np$(nranks).json", "w") do f
                JSON.print(f, results)
            end
        end
    end
end

end # module
