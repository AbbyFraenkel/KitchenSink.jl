

@doc """
    jacobi_preconditioner(A::Matrix{Float64})::Matrix{Float64}

Constructs a Jacobi preconditioner for the given matrix A.
"""
function JacobiPreconditioner(A::Matrix{Float64})::Matrix{Float64}
    M = diagm(diag(A))
    return inv(M)
end

@doc """
    ilu_preconditioner(A::Matrix{Float64})::IncompleteLU

Constructs an incomplete LU preconditioner for the given matrix A.
"""

function ILUPreconditioner(A::Matrix{Float64})::IncompleteLU
    sparse_A = sparse(A)
    F = incomplete_factorize(sparse_A)
    return F
end
