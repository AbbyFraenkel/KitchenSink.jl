"""
    solve_direct(A::AbstractMatrix{T}, b::AbstractVector{T}, method::Symbol)::Vector{T} where {T<:Real}

Solve a linear system using a direct solver.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `method::Symbol`: The direct method to use (:lu, :qr, etc.).

# Returns
- `Vector{T}`: The solution vector.

# Examples
```julia
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
x = solve_direct(A, b, :lu)
```

See also: [`KSDirectSolver`](@ref)
"""
function solve_direct(A::AbstractMatrix{T}, b::AbstractVector{T}, method::Symbol)::Vector{Float64} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    length(b) == size(A, 1) || throw(ArgumentError("Dimensions of A and b must match. Got A size $(size(A)) and b length $(length(b))."))

    # Convert A and b to Float64 to avoid inexact errors
    A = convert(Matrix{Float64}, A)
    b = convert(Vector{Float64}, b)

    if method == :lu
        return A \ b
    elseif method == :qr
        return qr(A) \ b
    elseif method == :cholesky
        return cholesky(A) \ b
    else
        throw(ArgumentError("Unsupported direct method: $method. Supported methods are :lu, :qr, and :cholesky."))
    end
end
"""
    solve_direct_multiple_rhs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, method::Symbol)::Matrix{T} where {T<:Real}

Solve a system of linear equations AX = B for multiple right-hand sides using a direct solver.

# Arguments
- `A::AbstractMatrix{T}`: The coefficient matrix of size `n x n`.
- `B::AbstractMatrix{T}`: The right-hand side matrix of size `n x m`.
- `method::Symbol`: The direct method to use (:lu, :qr, etc.).

# Returns
- `Matrix{T}`: The solution matrix X of size `n x m`.

# Examples
```julia
A = [4.0 1.0; 1.0 3.0]
B = [1.0 2.0; 2.0 3.0]
X = solve_direct_multiple_rhs(A, B, :lu)
```

See also: [`solve_direct`](@ref)
"""
function solve_direct_multiple_rhs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, method::Symbol)::Matrix{T} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    size(A, 1) == size(B, 1) || throw(ArgumentError("Dimensions of A and B must match. Got A size $(size(A)) and B size $(size(B))."))

    if method == :lu
        return A \ B
    elseif method == :qr
        return qr(A) \ B
    elseif method == :cholesky
        return cholesky(A) \ B
    else
        throw(ArgumentError("Unsupported direct method: $method. Supported methods are :lu, :qr, and :cholesky."))
    end
end
