using MPI

function sor_par!(A, stopdiff, maxiters, comm)
    # INITIALIZE
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    # nreqs = 2 * ((rank != 0) + (rank != nranks - 1))
    # reqs = MPI.MultiRequest(nreqs)

    # BROADCAST matrix dimensions N from master to all workers
    ni, nj = size(A)
    ni_global = Ref(ni)
    MPI.Bcast!(ni_global, comm; root=0)
    ni, nj = ni_global[], ni_global[]
    load = div(ni - 2, nranks)  # Base load

    # SCATTER matrix A from master to all workers [Option: Scatter!]
    # Set boundary conditions for u first and last rank
    u = zeros(eltype(A), load + 2, nj) # load with boundary
    if nranks > 1 
        if rank == 0
            u[1, :] = A[1, :]
            MPI.Send(A[end, :], comm, dest=nranks - 1)
        elseif rank == nranks - 1
            rec = zeros(nj)
            MPI.Recv!(rec, comm, source=0)
            u[end, :] = rec
        end
    else
        u[1, :] = A[1, :]
        u[end, :] = A[end, :]
    end
    # Scatter A to workers
    recv = zeros(ni, load) # recieve load without boundary
    MPI.Scatter!(collect(transpose(A[2:end-1, :])), recv, comm; root=0)
    # Add recieved to u
    u[2:end-1, :] = collect(transpose(recv))

    # PERFORM RED BLACK SOR
    max_diff = 0
    iter_count = 0
    for iter in 1:maxiters
        mydiff = zero(eltype(u))
        for color in (0, 1) # Black/Red
            # DATA TRANSFER
            # ireq = 0
            if rank > 0  # Everything except master
                # MPI.Irecv!(view(u, 1, :), comm, reqs[ireq += 1]; source=rank - 1)  # Receive in first row of u (boundary)
                # MPI.Isend(view(u, 2, :), comm, reqs[ireq += 1]; dest=rank - 1)  # Send first row of load
                MPI.Sendrecv!(view(u, 2, :), view(u, 1, :), comm; dest=rank - 1, source=rank - 1, sendtag=0, recvtag=0)
            end
            if rank < nranks - 1  # Everything except last rank
                # MPI.Irecv!(view(u, load + 2, :), comm, reqs[ireq += 1]; source=rank + 1)  # Receive in last row of u (boundary)
                # MPI.Isend(view(u, load + 1, :), comm, reqs[ireq += 1]; dest=rank + 1)  # Send last row of load
                MPI.Sendrecv!(view(u, load + 1, :), view(u, load + 2, :), comm; dest=rank + 1, source=rank + 1, sendtag=0, recvtag=0)
            end
            # MPI.Waitall(reqs)      
            # COMPUTE SOR
            for j in 2:(nj - 1)  # Columns (excluding boundary)
                for i in 2:(load + 1)  # Rows (excluding boundary)
                    global_i = (rank * load + i) - 1  # Global i corresponding to local i
                    global_j = j - 1  # Global j corresponding to local j
                    color_ij = (global_i + global_j) % 2  # 1 if red, 0 if black
                    if color != color_ij continue end  # Skip if color does not match
                    # Compute SOR
                    old_value = u[i, j]
                    u[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
                    # Update max difference
                    mydiff = max(mydiff, abs(u[i, j] - old_value))
                end
            end
        end
        # FIND GLOBAL DIFFERENCE
        diff_ref = Ref(mydiff)
        MPI.Allreduce!(diff_ref, max, comm)
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
    MPI.Gather!(snd, rcv, comm; root=0)
    if rank == 0 
        A[2:end-1, :] = collect(transpose(rcv))
    end
    return max_diff, iter_count
end

# DEAD CODE:
    # # Scatter matrix A from master to all workers [Option: Snd/Recv]
    # if rank == 0
    #     # Send matrix dimensions and local matrices to workers
    #     for other_rank in 1:(nranks - 1)
    #         local_matrix = view(A, other_rank * load + 1:(other_rank + 1) * load + 2, :)
    #         MPI.Send(local_matrix, comm, dest=other_rank)
    #     end
    #     # Define local matrix for master
    #     u = view(A, 1:(load + 2), :)
    # else
    #     # Receive local matrix from master
    #     u = zeros(eltype(A), load + 2, nj)
    #     MPI.Recv!(u, comm, source=0)
    # end

    # # Gather results from workers to master [Option: Snd/Recv]
    # if rank != 0  # Not Master
    #     MPI.Send(view(u, 2:(load + 1), :), comm, dest=0)
    # else  # Master
    #     lb = 2
    #     ub = load + 1
    #     A[lb:ub, :] = view(u, lb:ub, :)
    #     for other_rank in 1:(nranks - 1)
    #         lb += load
    #         ub += load
    #         u_rcv = view(A, lb:ub, :)
    #         MPI.Recv!(u_rcv, comm, source=other_rank)
    #     end
    #     println("[METHOD 1]", A, "\n")
    # end