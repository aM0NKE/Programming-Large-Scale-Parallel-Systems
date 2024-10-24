using MPI

function sor_par!(A, stopdiff, maxiters, comm)
    # INITIALIZE
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    # Timers
    comm_time = 0.0
    comp_time = 0.0

    # BROADCAST matrix dimensions N from master to all workers
    ni, nj = size(A)
    ni_global = Ref(ni)
    t_comm = MPI.Wtime()
    MPI.Bcast!(ni_global, comm; root=0)
    comm_time += MPI.Wtime() - t_comm
    ni, nj = ni_global[], ni_global[]
    load = div(ni - 2, nranks)  # Base load

    # SCATTER matrix A from master to all workers [Option: Scatter!]
    # Set boundary conditions for u first and last rank
    u = zeros(eltype(A), load + 2, nj) # load with boundary
    if nranks > 1 
        t_comm = MPI.Wtime()
        if rank == 0
            u[1, :] = A[1, :]
            MPI.Send(A[end, :], comm, dest=nranks - 1)
        elseif rank == nranks - 1
            rec = zeros(nj)
            MPI.Recv!(rec, comm, source=0)
            u[end, :] = rec
        end
        comm_time += MPI.Wtime() - t_comm
    else
        u[1, :] = A[1, :]
        u[end, :] = A[end, :]
    end
    # Scatter A to workers
    recv = zeros(ni, load) # recieve load without boundary
    t_comm = MPI.Wtime()
    MPI.Scatter!(collect(transpose(A[2:end-1, :])), recv, comm; root=0)
    comm_time += MPI.Wtime() - t_comm  # End of scatter communication
    # Add recieved to u
    u[2:end-1, :] .= collect(transpose(recv))

    # PERFORM RED BLACK SOR
    max_diff = 0
    iter_count = 0
    for iter in 1:maxiters
        mydiff = zero(eltype(u))
        for color in (0, 1) # Black/Red
            # DATA TRANSFER
            t_comm = MPI.Wtime()
            if rank > 0  # Everything except master
                MPI.Sendrecv!(view(u, 2, :), view(u, 1, :), comm; dest=rank - 1, source=rank - 1)
            end
            if rank < nranks - 1  # Everything except last rank
                MPI.Sendrecv!(view(u, load + 1, :), view(u, load + 2, :), comm; dest=rank + 1, source=rank + 1)
            end
            comm_time += MPI.Wtime() - t_comm  # End of sendrecv communication

            # COMPUTE SOR
            t_comp = MPI.Wtime()
            for j in 2:(nj - 1)  # Columns (excluding boundary)
                for i in 2:(load + 1)  # Rows (excluding boundary)
                    global_i = (rank * load + i) - 1  # Global i corresponding to local i
                    color_ij = (global_i + (j - 1)) % 2  # 1 if red, 0 if black
                    if color != color_ij continue end  # Skip if color does not match
                    # Compute SOR
                    old_value = u[i, j]
                    u[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
                    # Update max difference
                    mydiff = max(mydiff, abs(u[i, j] - old_value))
                end
            end
            comp_time += MPI.Wtime() - t_comp  # End of computation phase
        end
        # FIND GLOBAL DIFFERENCE
        diff_ref = Ref(mydiff)
        t_comm = MPI.Wtime()
        MPI.Allreduce!(diff_ref, max, comm)
        comm_time += MPI.Wtime() - t_comm  # End of allreduce communication
        diff = diff_ref[]
        # STOPPING CRITERION
        iter_count += 1
        if diff < stopdiff
            max_diff = diff
            break
        end
    end

    # GATHER results from workers to master [Option: Gather!]
    snd = collect(transpose(u[2:end-1, :]))
    if rank == 0
        rcv = zeros(eltype(A), nj, ni - 2)
    else
        rcv = nothing
    end
    t_comm = MPI.Wtime()
    MPI.Gather!(snd, rcv, comm; root=0)
    comm_time += MPI.Wtime() - t_comm  # End of gather communication
    if rank == 0 
        A[2:end-1, :] .= collect(transpose(rcv))
    end

    # Output results
    total_comm_time = MPI.Reduce(comm_time, MPI.SUM, 0, comm)
    total_comp_time = MPI.Reduce(comp_time, MPI.SUM, 0, comm)
    if rank == 0
        println("Total Communication Time: ", total_comm_time)
        println("Total Computation Time: ", total_comp_time)
        println("Communication/Computation Overhead Ratio: ", total_comm_time / total_comp_time)
    end
    return max_diff, iter_count
end