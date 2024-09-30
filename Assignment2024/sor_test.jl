module SORTEST

using MPI
using Printf
using LinearAlgebra

MPI.Init()

include("sor_seq.jl")
include("sor_par.jl")

function print_matrix(M)
    rows, cols = size(M) 
    for i in 1:rows
        for j in 1:cols
            @printf("%10.3f ", M[i, j])
        end
        println()
    end
end

# A matrix where Aij = 1 at boundary
function test_matrix_1(N)
    size = N + 2 # padding boundaries
    A = zeros(size,size)

    for j in 1:size
        for i in 1:size
            if i == 1 || j == 1 || i == size || j == size
                A[i, j] = 1.0
            end
        end
    end
    A
end

# A matrix where Aij = 1 for all values
function test_matrix_2(N)
    size = N + 2
    A = ones(size,size)
    A
end 

# A matrix where Aij = (i+j)/N at boundary 
function test_matrix_3(N)
    size = N + 2
    A = zeros(size, size)

    for j in 1:size
	    for i in 1:size 
            if i == 1 || j == 1 || i == size || j == size
                A[i, j] = (i + j)/size 
            end

        end
    end
    A
end 

# A matrix where Aij = (i+j)/N for all values
function test_matrix_4(N)
    size = N + 2
    A = zeros(size, size)

    for j in 1:size
        for i in 1:size
            A[i, j] = (i + j)/size 
        end
    end
    A
end

# List of test matrices
const test_matrices = [test_matrix_1, test_matrix_2, test_matrix_3, test_matrix_4]

function runtest(matrix_index, N, stopdiff, maxiters)
    comm = MPI.Comm_dup(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)

    test_matrix = test_matrices[matrix_index]

    if rank == 0
        A = test_matrix(N)
    else
        A = test_matrix(0)
    end

    A_par = copy(A)
    max_diff_par, iter_count_par = sor_par!(A_par,stopdiff,maxiters,comm)

    if rank == 0
        max_diff_seq, iter_count_seq = sor_seq!(A,stopdiff,maxiters)

        # Check pass condition
        if A_par ≈ A && iter_count_par == iter_count_seq
            # println("Sequential - Difference: $max_diff_seq , Iterations: $iter_count_seq")
            # println("Parallel   - Difference: $max_diff_par , Iterations: $iter_count_par")
            return true
        else
            println("----------------------------------------")
            println("Sequential - Difference: $max_diff_seq , Iterations: $iter_count_seq")
            println("Parallel   - Difference: $max_diff_par , Iterations: $iter_count_par")
            return false
        end
    end
end

end # module

