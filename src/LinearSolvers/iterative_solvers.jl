"""
    solve_iterative(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSIterativeSolver)::Vector{T} where {T<:Real}

Solve a linear system Ax = b using an iterative solver method.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `solver::KSIterativeSolver`: The iterative solver configuration.

# Returns
- `Vector{T}`: The solution vector.

# Examples
```julia
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
solver = KSIterativeSolver(:cg, 1000, 1e-6, nothing)
x = solve_iterative(A, b, solver)
```

See also: [`KSIterativeSolver`](@ref), [`solve_direct`](@ref)
"""

function solve_iterative(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSIterativeSolver)::Vector{T} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    length(b) == size(A, 1) || throw(ArgumentError("Dimensions of A and b must match. Got A size $(size(A)) and b length $(length(b))."))

    kwargs = Dict(:maxiter => solver.max_iter, :abstol => solver.tolerance, :reltol => solver.tolerance)
    if !isnothing(solver.preconditioner)
        kwargs[:Pl] = solver.preconditioner
    end

    if solver.method == :cg
        return IterativeSolvers.cg(A, b; kwargs...)
    elseif solver.method == :gmres
        return IterativeSolvers.gmres(A, b; kwargs...)
    elseif solver.method == :bicgstab
        return IterativeSolvers.bicgstab(A, b; kwargs...)
    else
        throw(ArgumentError("Unsupported iterative method: $(solver.method). Supported methods are :cg, :gmres, and :bicgstab."))
    end
end

"""
    solve_iterative_multiple_rhs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, solver::KSIterativeSolver)::Matrix{T} where {T<:Real}

Solve a system of linear equations AX = B for multiple right-hand sides using an iterative solver.

# Arguments
- `A::AbstractMatrix{T}`: The coefficient matrix of size `n x n`.
- `B::AbstractMatrix{T}`: The right-hand side matrix of size `n x m`.
- `solver::KSIterativeSolver`: The iterative solver configuration.

# Returns
- `Matrix{T}`: The solution matrix X of size `n x m`.

# Examples
```julia
A = [4.0 1.0; 1.0 3.0]
B = [1.0 2.0; 2.0 3.0]
solver = KSIterativeSolver(:cg, 1000, 1e-6, nothing)
X = solve_iterative_multiple_rhs(A, B, solver)
```

See also: [`solve_iterative`](@ref), [`KSIterativeSolver`](@ref)
"""
function solve_iterative_multiple_rhs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, solver::KSIterativeSolver)::Matrix{T} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    size(A, 1) == size(B, 1) || throw(ArgumentError("Dimensions of A and b must match. Got A size $(size(A)) and B size $(size(B))."))

    X = similar(B)
    Threads.@threads for i in 1:size(B, 2)
        X[:, i] = solve_iterative(A, B[:, i], solver)
    end
    return X
end

"""
    iterative_refinement(A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}; max_iter::Int=5, tol::Real=1e-12)::Vector{T} where {T<:Real}

Perform iterative refinement to improve the accuracy of a solution to the linear system Ax = b.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `x::AbstractVector{T}`: The initial solution vector.
- `b::AbstractVector{T}`: The right-hand side vector.
- `max_iter::Int=5`: Maximum number of refinement iterations.
- `tol::Real=1e-12`: Tolerance for convergence.

# Returns
- `Vector{T}`: The refined solution vector.

# Examples
```julia
A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
x_initial = A \\ b
x_refined = iterative_refinement(A, x_initial, b)
```

See also: [`solve_iterative`](@ref), [`solve_direct`](@ref)
"""
function iterative_refinement(A::AbstractMatrix{T}, x::AbstractVector{T}, b::AbstractVector{T}; max_iter::Int=5, tol::Real=1e-12)::Vector{T} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    length(x) == size(A, 1) || throw(ArgumentError("Dimensions of A and x must match. Got A size $(size(A)) and x length $(length(x))."))
    length(b) == size(A, 1) || throw(ArgumentError("Dimensions of A and b must match. Got A size $(size(A)) and b length $(length(b))."))

    for iter in 1:max_iter
        r = b - A * x  # Compute residual
        if norm(r) < tol
            @info "Converged in $iter iterations with residual norm $(norm(r))."
            return x  # Converged
        end
        dx = A \ r  # Solve for correction
        x += dx  # Update solution
    end
    @warn "Reached maximum iterations ($max_iter) with residual norm $(norm(b - A * x))."
    return x  # Return the refined solution
end
