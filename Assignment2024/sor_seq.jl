using Printf

function sor_seq!(A,stopdiff,maxiters)
    ni,nj = size(A)
    iter_count = 0
    max_diff = 0
    for iter in 1:maxiters
        diff = zero(eltype(A))
        for color in (0,1)
            for j in 2:(nj-1)
                color_j = mod(j,2)
                for i in 2:(ni-1)
                    color_i = mod(i,2)
                    color_ij = xor(color_i,color_j)
                    if color != color_ij
                        continue
                    end
                    Aij_old = A[i,j]
                    Aij_new = 0.25*(A[i-1,j]+A[i+1,j]+A[i,j-1]+A[i,j+1]) 
                    diff_Aij = abs(Aij_new - Aij_old)
                    diff = max(diff,diff_Aij)
                    A[i,j] = Aij_new
                end
            end
        end
        iter_count += 1
        if diff < stopdiff
            max_diff = diff
            break
        end
    end
    return max_diff, iter_count
end
