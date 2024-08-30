module LinearSolvers

using LinearAlgebra, SparseArrays, IterativeSolvers, AlgebraicMultigrid
using ..KSTypes, ..CommonMethods, ..Preconditioners

export solve_linear_system, select_optimal_solver


"""
    solve_linear_system(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::AbstractKSLinearSolver) where T <: Real

Solve a linear system Ax = b using the specified solver configuration.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `solver::AbstractKSLinearSolver`: The solver configuration.

# Returns
- `AbstractVector{T}`: The solution vector.

# Throws
- `DimensionMismatch`: If the dimensions of A and b are incompatible.
- `ArgumentError`: If an unsupported solver type is provided.
"""
function solve_linear_system(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::AbstractKSLinearSolver) where T <: Real
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("Matrix A must be square"))
    size(A, 1) == length(b) || throw(DimensionMismatch("Dimensions of A and b must match"))

    if solver isa KSDirectSolver
        return solve_direct(A, b, solver.method)
    elseif solver isa KSIterativeSolver
        return solve_iterative(A, b, solver)
    elseif solver isa KSAMGSolver
        return solve_amg(A, b, solver)
    else
        throw(ArgumentError("Unsupported solver type: $(typeof(solver))"))
    end
end

"""
    solve_direct(A::AbstractMatrix{T}, b::AbstractVector{T}, method::Symbol) where T <: Real

Solve a linear system using a direct method.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `method::Symbol`: The direct method to use (:lu or :qr).

# Returns
- `AbstractVector{T}`: The solution vector.

# Throws
- `ArgumentError`: If an unsupported direct method is provided.
"""
function solve_direct(A::AbstractMatrix{T}, b::AbstractVector{T}, method::Symbol) where T <: Real
    if method == :lu
        return A \ b
    elseif method == :qr
        return qr(A) \ b
    else
        throw(ArgumentError("Unsupported direct method: $method"))
    end
end

"""
    solve_iterative(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSIterativeSolver) where T <: Real

Solve a linear system using an iterative method.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `solver::KSIterativeSolver`: The iterative solver configuration.

# Returns
- `AbstractVector{T}`: The solution vector.

# Throws
- `ArgumentError`: If an unsupported iterative method is provided.
"""
function solve_iterative(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSIterativeSolver) where T <: Real
    if solver.method == :cg
        return cg(A, b; maxiter=solver.max_iter, tol=solver.tolerance, Pl=solver.preconditioner)
    elseif solver.method == :gmres
        return gmres(A, b; maxiter=solver.max_iter, tol=solver.tolerance, Pl=solver.preconditioner)
    else
        throw(ArgumentError("Unsupported iterative method: $(solver.method)"))
    end
end

"""
    solve_amg(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSAMGSolver; verbose::Bool=false) where T <: Real

Solve a linear system using the Algebraic Multigrid (AMG) method.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `solver::KSAMGSolver`: The AMG solver configuration.
- `verbose::Bool`: Whether to print verbose output (default: false).

# Returns
- `AbstractVector{T}`: The solution vector.

# Throws
- `ArgumentError`: If an unsupported smoother is provided.
"""
function solve_amg(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::KSAMGSolver; verbose::Bool=false) where T <: Real
    smoother = select_smoother(solver.smoother)
    A_sparse = sparse(A)
    ml = AlgebraicMultigrid.ruge_stuben(A_sparse)
    x, conv = AlgebraicMultigrid.solve(ml, b; maxiter=solver.max_iter, abstol=solver.tolerance, smoother=smoother)

    if verbose
        if conv
            @info "AMG solver converged."
        else
            @warn "AMG solver did not converge within the specified tolerance."
        end
    end

    return x
end

"""
    select_smoother(smoother::Symbol)

Select the appropriate smoother function for AlgebraicMultigrid.

# Arguments
- `smoother::Symbol`: The smoother type (:jacobi, :gauss_seidel, or :sor).

# Returns
- `Function`: The smoother function.

# Throws
- `ArgumentError`: If an unsupported smoother is provided.
"""
function select_smoother(smoother::Symbol)
    if smoother == :jacobi
        return AlgebraicMultigrid.Jacobi()
    elseif smoother == :gauss_seidel
        return AlgebraicMultigrid.GaussSeidel()
    elseif smoother == :sor
        return AlgebraicMultigrid.SOR(1.0)
    else
        throw(ArgumentError("Unsupported smoother: $smoother"))
    end
end

"""
    select_optimal_solver(A::AbstractMatrix{T}, problem_type::Symbol)::AbstractKSLinearSolver where T <: Real

Select the optimal solver based on the properties of the input matrix A and the problem type.

# Arguments
- `A::AbstractMatrix{T}`: The input matrix.
- `problem_type::Symbol`: The type of the problem (e.g., :general, :spd, :elliptic).

# Returns
- `AbstractKSLinearSolver`: The selected optimal solver.

# Throws
- `ArgumentError`: If an unsupported problem type is provided.
"""
function select_optimal_solver(A::AbstractMatrix{T}, problem_type::Symbol)::AbstractKSLinearSolver where T <: Real
    n = size(A, 1)
    nnz = count(!iszero, A)
    density = nnz / (n * n)

    if n < 1000 && density > 0.1
        return KSDirectSolver(:lu)
    elseif issymmetric(A) && isposdef(A)
        return KSIterativeSolver(:cg, 1000, 1e-8, Preconditioners.amg_preconditioner(A))
    elseif problem_type == :elliptic
        return KSAMGSolver(1000, 1e-8, :gauss_seidel)
    elseif problem_type == :general
        return KSIterativeSolver(:gmres, 1000, 1e-8, Preconditioners.ilu_preconditioner(sparse(A)))
    else
        throw(ArgumentError("Unsupported problem type: $problem_type"))
    end
end

end # module LinearSolvers
