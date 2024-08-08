# src/Preconditioners/Preconditioners.jl
module Preconditioners

using LinearAlgebra, SparseArrays, IncompleteLU, AlgebraicMultigrid
using ..KSTypes

export jacobi_preconditioner, ilu_preconditioner, amg_preconditioner

"""
    jacobi_preconditioner(A::AbstractMatrix{T}) where T<:Real

Create a Jacobi preconditioner for the matrix A.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.

# Returns
- `Diagonal{T,Vector{T}}`: The Jacobi preconditioner as a diagonal matrix.

# Throws
- `ArgumentError`: If the matrix is not square or symmetric.

# Examples
```julia
A = Symmetric([4.0 1.0; 1.0 3.0])
preconditioner = jacobi_preconditioner(A)
```
"""
function jacobi_preconditioner(A::AbstractMatrix{T}) where {T<:Real}
    if !issymmetric(A)
        throw(ArgumentError("Matrix A must be symmetric for Jacobi preconditioner"))
    end
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end

    Diagonal(one(T) ./ diag(A))
end

"""
    ilu_preconditioner(A::SparseMatrixCSC{T}) where T<:Real

Create an Incomplete LU (ILU) preconditioner for the sparse matrix A.

# Arguments
- `A::SparseMatrixCSC{T}`: The sparse system matrix.

# Returns
- `IncompleteLU.ILUFactorization{T}`: The ILU preconditioner.

# Throws
- `ArgumentError`: If the matrix is not square.

# Examples
```julia
A = sparse([4.0 1.0; 1.0 3.0])
preconditioner = ilu_preconditioner(A)
```
"""
function ilu_preconditioner(A::SparseMatrixCSC{T}) where {T<:Real}
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end

    IncompleteLU.ilu(A, τ=0.1)
end

"""
    amg_preconditioner(A::SparseMatrixCSC{T}) where T<:Real

Create an Algebraic Multigrid (AMG) preconditioner for the sparse matrix A.

# Arguments
- `A::SparseMatrixCSC{T}`: The sparse system matrix.

# Returns
- `AlgebraicMultigrid.MultiLevel{T}`: The AMG preconditioner.

# Throws
- `ArgumentError`: If the matrix is not square or not symmetric.

# Examples
```julia
A = sparse(Symmetric([4.0 1.0; 1.0 3.0]))
preconditioner = amg_preconditioner(A)
```
"""
function amg_preconditioner(A::SparseMatrixCSC{T}) where {T<:Real}
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Matrix A must be square"))
    end
    if !issymmetric(A)
        throw(ArgumentError("Matrix A must be symmetric for AMG preconditioner"))
    end

    AlgebraicMultigrid.ruge_stuben(A)
end

end # module Preconditioners
