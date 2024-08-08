module LinearSolvers

using LinearAlgebra, SparseArrays, IterativeSolvers, AlgebraicMultigrid, BenchmarkTools, DataFrames
using ..KSTypes, ..Preprocessing

export solve_linear_system, select_optimal_solver, solve_with_logging, benchmark_solvers

export solve_direct, solve_direct_multiple_rhs
include("direct_solvers.jl")

export solve_iterative, solve_iterative_multiple_rhs, iterative_refinement
include("iterative_solvers.jl")

export solve_amg, solve_amg_multiple_rhs
include("amg_solvers.jl")

"""
    solve_linear_system(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::AbstractKSLinearSolver)::AbstractVector{T} where T

Solve a linear system Ax = b using the specified solver configuration.

# Arguments
- `A::AbstractMatrix{T}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `solver::AbstractKSLinearSolver`: The solver configuration.

# Returns
- `AbstractVector{T}`: The solution vector.

# Example
```julia
using KitchenSink.LinearSolvers
using KitchenSink.KSTypes

A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
solver = KSDirectSolver(:lu)
x = solve_linear_system(A, b, solver)
```
"""
function solve_linear_system(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::AbstractKSLinearSolver)::AbstractVector{T} where {T}
    if solver isa KSDirectSolver
        return solve_direct(A, b, solver.method)
    elseif solver isa KSIterativeSolver
        return solve_iterative(A, b, solver)
    elseif solver isa KSAMGSolver
        return solve_amg(A, b, solver)
    else
        error("Unknown solver type")
    end
end

"""
    select_optimal_solver(A::AbstractMatrix{T}, problem_type::Symbol)::Union{KSDirectSolver,KSIterativeSolver,KSAMGSolver} where {T<:Real}

Select the optimal solver based on the properties of the input matrix A and the problem type.

# Arguments
- `A::AbstractMatrix{T}`: The input matrix.
- `problem_type::Symbol`: The type of the problem (e.g., :general, :spd, :elliptic).

# Returns
- `Union{KSDirectSolver,KSIterativeSolver,KSAMGSolver}`: The selected optimal solver.

# Example
```julia
using KitchenSink.LinearSolvers
using KitchenSink.KSTypes

A = sprand(1000, 1000, 0.01)
problem_type = :general
solver = select_optimal_solver(A, problem_type)
```
"""
function select_optimal_solver(A::AbstractMatrix{T}, problem_type::Symbol)::Union{KSDirectSolver,KSIterativeSolver,KSAMGSolver} where {T<:Real}
    n = size(A, 1)
    nnz = count(!iszero, A)
    density = nnz / (n * n)

    if n < 1000 && density > 0.1
        return KSDirectSolver(:lu)
    elseif issymmetric(A) && isposdef(A)
        return KSIterativeSolver(:cg, 1000, 1e-8, amg_preconditioner(A))
    elseif problem_type == :elliptic
        return KSAMGSolver(1000, 1e-8, :gauss_seidel)
    else
        return KSIterativeSolver(:gmres, 1000, 1e-8, ilu_preconditioner(sparse(A)))
    end
end

"""
    solve_with_logging(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::Union{KSDirectSolver, KSIterativeSolver, KSAMGSolver})::Vector{T} where {T<:Real}

Solve a linear system Ax = b using the specified solver and log the solve time and residual.

# Arguments
- `A::AbstractMatrix{T}`: The coefficient matrix of the linear system.
- `b::AbstractVector{T}`: The right-hand side vector of the linear system.
- `solver::Union{KSDirectSolver, KSIterativeSolver, KSAMGSolver}`: The solver to use for solving the linear system.

# Returns
- `Vector{T}`: The solution vector x of the linear system.

# Example
```julia
using KitchenSink.LinearSolvers
using KitchenSink.KSTypes

A = [4.0 1.0; 1.0 3.0]
b = [1.0, 2.0]
solver = KSDirectSolver(:lu)
x = solve_with_logging(A, b, solver)
```
"""
function solve_with_logging(A::AbstractMatrix{T}, b::AbstractVector{T}, solver::Union{KSDirectSolver,KSIterativeSolver,KSAMGSolver})::Vector{T} where {T<:Real}
    @info "Starting linear system solve" size(A) solver

    time = @elapsed begin
        x = solve_linear_system(A, b, solver)
    end

    residual = norm(b - A * x)
    @info "Linear system solved" time residual

    return x
end
end
