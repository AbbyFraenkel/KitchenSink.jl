"""
    solve_amg(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSAMGSolver; verbose=false)::Vector{T} where {T<:Real}

Solve a linear system using Algebraic Multigrid (AMG) solver.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `solver::KSAMGSolver`: The AMG solver configuration.
- `verbose::Bool=false`: Whether to print verbose output.

# Returns
- `Vector{T}`: The solution vector.

# Examples
```julia
A = sprand(1000, 1000, 0.01)
A = A + A' + 1000I  # Make diagonally dominant
b = rand(1000)
solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
x = solve_amg(A, b, solver)
```

See also: [`KSAMGSolver`](@ref), [`solve_direct`](@ref), [`solve_iterative`](@ref)
"""
function solve_amg(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSAMGSolver; verbose=false)::Vector{T} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    length(b) == size(A, 1) || throw(ArgumentError("Dimensions of A and b must match. Got A size $(size(A)) and b length $(length(b))."))

    smoother = select_smoother(solver.smoother)
    A_sparse = sparse(A)
    ml = AlgebraicMultigrid.ruge_stuben(A_sparse)
    x, conv = AlgebraicMultigrid.solve(ml, b; maxiter=solver.max_iter, abstol=solver.tolerance, smoother=smoother)

    if verbose
        if conv
            println("AMG solver converged.")
        else
            @warn "AMG solver did not converge within the specified tolerance."
        end
    end

    return x
end

"""
    solve_amg_multiple_rhs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, solver::KSAMGSolver; verbose=false)::Matrix{T} where {T<:Real}

Solve a system of linear equations AX = B for multiple right-hand sides using Algebraic Multigrid (AMG) solver.

# Arguments
- `A::AbstractMatrix{T}`: The coefficient matrix of size `n x n`.
- `B::AbstractMatrix{T}`: The right-hand side matrix of size `n x m`.
- `solver::KSAMGSolver`: The AMG solver configuration.
- `verbose::Bool=false`: Whether to print verbose output.

# Returns
- `Matrix{T}`: The solution matrix X of size `n x m`.

# Examples
```julia
A = sprand(1000, 1000, 0.01)
A = A + A' + 1000I  # Make diagonally dominant
B = rand(1000, 5)
solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
X = solve_amg_multiple_rhs(A, B, solver)
```

See also: [`solve_amg`](@ref), [`KSAMGSolver`](@ref)
"""
function solve_amg_multiple_rhs(A::AbstractMatrix{T}, B::AbstractMatrix{T}, solver::KSAMGSolver; verbose=false)::Matrix{T} where {T<:Real}
    size(A, 1) == size(A, 2) || throw(ArgumentError("Matrix A must be square. Got size $(size(A))."))
    size(A, 1) == size(B, 1) || throw(ArgumentError("Dimensions of A and B must match. Got A size $(size(A)) and B size $(size(B))."))

    X = similar(B)
    smoother = select_smoother(solver.smoother)
    A_sparse = sparse(A)
    ml = ruge_stuben(A_sparse)
    Threads.@threads for i in eachindex(B, 2)
        X[:, i], conv = AlgebraicMultigrid.solve(ml, B[:, i]; maxiter=solver.max_iter, tol=solver.tolerance, smoother=smoother)
        if verbose && !conv
            @warn "AMG solver did not converge within the specified tolerance for column $i"
        end
    end
    return X
end

# Helper function to select smoother
function select_smoother(smoother::Symbol)
    if smoother == :jacobi
        return Jacobi()
    elseif smoother == :gauss_seidel
        return GaussSeidel()
    else
        throw(ArgumentError("Unsupported smoother: $smoother. Supported smoothers are :jacobi and :gauss_seidel."))
    end
end
