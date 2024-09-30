using MPI

function sor_par!(A, stopdiff, maxiters, comm)
    # INITIALIZE
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    nreqs = 2*((rank != 0) + (rank != (nranks-1)))
    reqs = MPI.MultiRequest(nreqs)

    if rank == 0
        ni, nj = size(A) # Size includes boundary
        load = div(ni-2, nranks) # Number of rows per rank
        # Send N to workers
        for other_rank in 1:(nranks-1)
            MPI.Send([ni], comm, dest=other_rank)
        end
        
        # Define local matrix for master
        u = view(A, 1:(load+2), :) # Include boundary rows
        u_new = copy(u)
        
        # Send local matrices to workers
        for other_rank in 1:(nranks-1)
            lb = other_rank * load + 1
            ub = lb + load + 1
            local_matrix = view(A, lb:ub, :)
            MPI.Send(local_matrix, comm, dest=other_rank)
        end
    else
        # Workers receive N from master
        N = zeros(Int, 1)
        MPI.Recv!(N, comm, source=0)
        ni, nj = N[1], N[1]
        load = div(ni-2, nranks)

        # Receive local matrix from master
        u = zeros(eltype(A), load + 2, nj) # Local matrix size for worker
        MPI.Recv!(u, comm, source=0)
        u_new = copy(u)
        
        # Free memory for A on the workers
        A = zeros(0)
    end

    # PERFORM RED BLACK SOR
    max_diff = 0
    iter_count = 0
    for iter in 1:maxiters
        mydiff = zero(eltype(u)) 
        u_old = copy(u) # Copy u to u_old (needed to compute diff)

        for color in (0, 1) # Black/Red

            # DATA TRANSFER
            ireq = 0
            if rank != 0 # Everything except master
                neigh_rank = rank-1
                u_snd = view(u, 2:2, :) # Send first row of load
                u_rcv = view(u, 1:1, :) # Recieve in first row of u (boundary)
                dest = neigh_rank
                source = neigh_rank
                ireq += 1
                MPI.Isend(u_snd, comm, reqs[ireq]; dest)
                ireq += 1
                MPI.Irecv!(u_rcv, comm, reqs[ireq]; source)
            end
            if rank != (nranks-1) # Everything except last rank
                neigh_rank = rank+1
                u_snd = view(u, (load+1):(load+1), :) # Send last row of load
                u_rcv = view(u, (load+2):(load+2), :) # Recieve in last row of u (boundary)
                dest = neigh_rank
                source = neigh_rank
                ireq += 1
                MPI.Isend(u_snd, comm, reqs[ireq]; dest)
                ireq += 1
                MPI.Irecv!(u_rcv, comm, reqs[ireq]; source)
            end
            MPI.Waitall(reqs)
            
            # COMPUTE SOR
            for j in 2:(nj-1) # Columns (excluding boundary)
                for i in 2:(load+1) # Rows (excluding boundary)
                    # Check if color matches
                    global_i = (rank * load + i) - 1 # Global i corresponding to local i
                    global_j = j - 1 # Global j corresponding to local j
                    color_ij = (global_i + global_j) % 2 # 1 if red, 0 if black
                    if color != color_ij # Skip if color does not match
                        continue
                    end

                    # Compute SOR
                    u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])                    
                    
                    # Update max difference
                    diff_ij = abs(u_new[i, j] - u_old[i, j])
                    mydiff = max(mydiff, diff_ij)
                end
            end
            u = u_new
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

    # GATHER RESULTS
    if rank != 0 # Not Master
        u_snd = view(u, 2:(load+1), :)
        MPI.Send(u_snd, comm, dest=0)
    else # Master
        lb = 2
        ub = load + 1
        A[lb:ub, :] = view(u, lb:ub, :)
        for other_rank in 1:(nranks-1)
            lb += load
            ub += load
            u_rcv = view(A, lb:ub, :)
            MPI.Recv!(u_rcv, comm, source=other_rank)
        end
    end
    return max_diff, iter_count
end
