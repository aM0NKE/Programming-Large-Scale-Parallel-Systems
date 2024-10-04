using MPI
include("sor_test.jl")

function test1()
    N = 10  
    P = 1 
    maxiters = 1000
    stopdiff = 1e-20
    matrix_index = 1
    run_test(1, matrix_index, N, stopdiff, P, maxiters)
end

function test2()
    N = 10
    P = 5
    maxiters = 1000
    stopdiff = 1e-3
    matrix_index = 1
    run_test(2, matrix_index, N, stopdiff, P, maxiters)
end

function test3()
    N = 100
    P = 2
    maxiters = 1000
    stopdiff = 1e-20
    matrix_index = 2
    run_test(3, matrix_index, N, stopdiff, P, maxiters)
end

function test4()
    N = 100
    P = 4
    maxiters = 1000
    stopdiff = 1e-3
    matrix_index = 2
    run_test(4, matrix_index, N, stopdiff, P, maxiters)
end

function test5()
    N = 10
    P = 1
    maxiters = 1000
    stopdiff = 1e-20
    matrix_index = 3
    run_test(5, matrix_index, N, stopdiff, P, maxiters)
end

function test6()
    N = 10
    P = 2
    maxiters = 1000
    stopdiff = 1e-8
    matrix_index = 3
    run_test(6, matrix_index, N, stopdiff, P, maxiters)
end

function test7()
    N = 10
    P = N
    maxiters = 1000
    stopdiff = 1e-20
    matrix_index = 4
    run_test(7, matrix_index, N, stopdiff, P, maxiters)
end

function test8()
    N = 100
    P = 2
    maxiters = 1000
    stopdiff = 1e-3
    matrix_index = 4
    run_test(8, matrix_index, N, stopdiff, P, maxiters)
end

function test9()
    N = 100
    P = 4
    maxiters = 1000
    stopdiff = 1e-10
    matrix_index = 4
    run_test(9, matrix_index, N, stopdiff, P, maxiters)
end

function test10()
    N = 500
    P = 10
    maxiters = 1000
    stopdiff = 1e-12
    matrix_index = 1
    run_test(10, matrix_index, N, stopdiff, P, maxiters)
end

function run_test(test_index, matrix_index, N, stopdiff, P, maxiters)
    
    mpi_command =  """ 
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
    """

    run(`$(mpiexec()) -np $P julia --project=. -e $mpi_command`);
end


tests = [
        test1, 
        test2, 
        test3, 
        test4, 
        test5, 
        test6, 
        test7, 
        test8, 
        test9, 
        test10
    ]
    
for (i, test) in enumerate(tests)
    test()
end 