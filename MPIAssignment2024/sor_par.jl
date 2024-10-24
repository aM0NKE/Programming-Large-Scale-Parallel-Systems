using MPI

function sor_par!(A, stopdiff, maxiters, comm)
    # INITIALIZE
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    # BROADCAST
    ni, nj = size(A)
    ni_global = Ref(ni)
    MPI.Bcast!(ni_global, comm; root=0)
    ni, nj = ni_global[], ni_global[]
    load = div(ni - 2, nranks)
    # SCATTER
    u = zeros(eltype(A), load + 2, nj)
    if nranks > 1 
        if rank == 0
            u[1, :] = A[1, :]
            sndbuf = A[end, :]
            MPI.Send(sndbuf, comm, dest=nranks - 1)
        elseif rank == nranks - 1
            rcvbuf = zeros(nj)
            MPI.Recv!(rcvbuf, comm, source=0)
            u[end, :] = rcvbuf
        end
    else
        u[1, :] = A[1, :]
        u[end, :] = A[end, :]
    end
    rcvbuf = zeros(ni, load)
    MPI.Scatter!(collect(transpose(A[2:end-1, :])), rcvbuf, comm; root=0)
    u[2:end-1, :] .= collect(transpose(rcvbuf))
    # PERFORM RED BLACK SOR
    max_diff = 0
    iter_count = 0
    for iter in 1:maxiters
        mydiff = zero(eltype(u))
        for color in (0, 1)
            # DATA TRANSFER
            if rank > 0
                MPI.Sendrecv!(view(u, 2, :), view(u, 1, :), comm; dest=rank - 1, source=rank - 1)
            end
            if rank < nranks - 1
                MPI.Sendrecv!(view(u, load + 1, :), view(u, load + 2, :), comm; dest=rank + 1, source=rank + 1)
            end
            # COMPUTE SOR
            for j in 2:(nj - 1)
                for i in 2:(load + 1)
                    # Check color
                    global_i = (rank * load + i) - 1
                    color_ij = (global_i + (j - 1)) % 2
                    if color != color_ij continue end
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
    # GATHER
    snd = collect(transpose(u[2:end-1, :]))
    if rank == 0
        rcv = zeros(eltype(A), nj, ni - 2)
    else
        rcv = nothing
    end
    MPI.Gather!(snd, rcv, comm; root=0)
    if rank == 0 
        A[2:end-1, :] .= collect(transpose(rcv))
    end
    return max_diff, iter_count
end