module Preconditioners

using LinearAlgebra, SparseArrays, AlgebraicMultigrid, IncompleteLU
using ..KSTypes, ..CommonMethods

export jacobi_preconditioner, ilu_preconditioner, amg_preconditioner

"""
    jacobi_preconditioner(A::AbstractMatrix{T}) where T<:Real

Create a Jacobi preconditioner for the matrix A.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.

# Returns
- The Jacobi preconditioner as a diagonal matrix.

# Throws
- `ArgumentError`: If the matrix A is not symmetric or not square.
"""
function jacobi_preconditioner(A::AbstractMatrix{T}) where T<:Real
    if !issymmetric(A)
        throw(ArgumentError("Matrix A must be symmetric for Jacobi preconditioner"))
    end
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end

    return Diagonal(one(T) ./ diag(A))
end

"""
    ilu_preconditioner(A::SparseMatrixCSC{T}) where T<:Real

Create an Incomplete LU (ILU) preconditioner for the sparse matrix A.

# Arguments
- `A::SparseMatrixCSC{T}`: The sparse system matrix.

# Returns
- The ILU preconditioner.

# Throws
- `ArgumentError`: If the matrix A is not square.
"""
function ilu_preconditioner(A::SparseMatrixCSC{T}) where T<:Real
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end

    IncompleteLU.ilu(A, τ = 0.1)
end

"""
    amg_preconditioner(A::SparseMatrixCSC{T}) where T<:Real

Create an Algebraic Multigrid (AMG) preconditioner for the sparse matrix A.

# Arguments
- `A::SparseMatrixCSC{T}`: The sparse system matrix.

# Returns
- The AMG preconditioner.

# Throws
- `ArgumentError`: If the matrix A is not square or not symmetric.
"""
function amg_preconditioner(A::SparseMatrixCSC{T}) where T<:Real
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end
    if !issymmetric(A)
        throw(ArgumentError("Matrix A must be symmetric for AMG preconditioner"))
    end
    AlgebraicMultigrid.ruge_stuben(A)
end

end # module Preconditioners
