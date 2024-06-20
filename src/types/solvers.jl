# using DocStringExtensions
@doc """
    Solver

An abstract type representing a solver.
"""
abstract type AbstractSolver end

@doc """
    TimeSteppingSolver

A structure representing a time-stepping solver.

# Fields
- `dt::AbstractFloat`: Time step size.
- `max_steps::Int`: Maximum number of steps.
"""
mutable struct TimeSteppingSolver{T<:AbstractFloat} <: AbstractSolver
    dt::T
    max_steps::Int
end

@doc """
    TimeSteppingSolver(dt::AbstractFloat, max_steps::Int)

Create a new `TimeSteppingSolver` instance.

# Arguments
- `dt::AbstractFloat`: Time step size.
- `max_steps::Int`: Maximum number of steps.

# Returns
- A `TimeSteppingSolver` instance.

# Errors
- Throws an `ArgumentError` if `dt` or `max_steps` are not positive.
"""
function TimeSteppingSolver(dt::T, max_steps::Int) where {T<:AbstractFloat}
    check_positive(dt, "Time step size")
    check_positive(max_steps, "Maximum steps")

    instance = TimeSteppingSolver{T}(dt, max_steps)
    check_fields(instance, "TimeSteppingSolver")

    return instance
end

@doc """
    OptimalControlSolver

A structure representing an optimal control solver.

# Fields
- `tolerance::AbstractFloat`: Tolerance level.
- `max_iterations::Int`: Maximum number of iterations.
"""
mutable struct OptimalControlSolver{T<:AbstractFloat} <: AbstractSolver
    tolerance::T
    max_iterations::Int
end

@doc """
    OptimalControlSolver(tolerance::AbstractFloat, max_iterations::Int)

Create a new `OptimalControlSolver` instance.

# Arguments
- `tolerance::AbstractFloat`: Tolerance level.
- `max_iterations::Int`: Maximum number of iterations.

# Returns
- An `OptimalControlSolver` instance.

# Errors
- Throws an `ArgumentError` if `tolerance` or `max_iterations` are not positive.
"""
function OptimalControlSolver(tolerance::T, max_iterations::Int) where {T <: AbstractFloat}
    check_positive(tolerance, "Tolerance")
    check_positive(max_iterations, "Maximum iterations")

    instance = OptimalControlSolver{T}(tolerance, max_iterations)
    check_fields(instance, "OptimalControlSolver")

    return instance
end

@doc """
    SolverLog

A structure representing a solver log.

# Fields
- `iteration::Int`: Iteration number.
- `residual::Float64`: Residual value.
- `error_estimate::Float64`: Error estimate.
"""
struct SolverLog{T<:AbstractFloat}
    iteration::Int
    residual::T
    error_estimate::T
end

@doc """
    SolverLog(iteration::Int, residual::Float64, error_estimate::Float64)

Create a new `SolverLog` instance.

# Arguments
- `iteration::Int`: Iteration number.
- `residual::Float64`: Residual value.
- `error_estimate::Float64`: Error estimate.

# Returns
- A `SolverLog` instance.

# Errors
- Throws an `ArgumentError` if `iteration` is not positive or if `residual` or `error_estimate` are negative.
"""
function SolverLog(iteration::Int, residual::T, error_estimate::T) where {T<:AbstractFloat}
    check_positive(iteration, "Iteration number")
    check_non_negative(residual, "Residual value")
    check_non_negative(error_estimate, "Error estimate")

    instance = SolverLog{T}(iteration, residual, error_estimate)
    check_fields(instance, "SolverLog")

    return instance
end

@doc """
    SolverMonitor

A structure representing a solver monitor.

# Fields
- `logs::Vector{SolverLog}`: Vector of solver logs.
"""
struct SolverMonitor{T<:AbstractVector}
    logs::T
end

@doc """
    SolverMonitor(logs::Vector{SolverLog})

Create a new `SolverMonitor` instance.

# Arguments
- `logs::Vector{SolverLog}`: Vector of solver logs.

# Returns
- A `SolverMonitor` instance.

# Errors
- Throws an `ArgumentError` if `logs` are empty.
"""
function SolverMonitor(logs::T) where {T<:AbstractVector}
    # check_non_empty_elements(logs, "Solver logs")

    instance = SolverMonitor{T}(logs)
    check_fields(instance, "SolverMonitor")

    return instance
end

@doc """
    ParallelOptions

A structure representing parallel options for solvers.

# Fields
- `use_parallel::Bool`: Flag indicating whether to use parallel computation.
- `num_threads::Int`: Number of threads to use.
- `solver_type::Symbol`: Type of solver (:mpi, :openmp, etc.).
"""
mutable struct ParallelOptions{T<:AbstractFloat}
    use_parallel::Bool
    num_threads::T
    solver_type::Symbol
end

@doc """
    ParallelOptions(use_parallel::Bool, num_threads::Int, solver_type::Symbol)

Create a new `ParallelOptions` instance.

# Arguments
- `use_parallel::Bool`: Flag indicating whether to use parallel computation.
- `num_threads::Int`: Number of threads to use.
- `solver_type::Symbol`: Type of solver (:mpi, :openmp, etc.).

# Returns
- A `ParallelOptions` instance.

# Errors
- Throws an `ArgumentError` if `num_threads` is not positive.
"""
function ParallelOptions(use_parallel::Bool, num_threads::T, solver_type::Symbol) where {T<:AbstractFloat}
    check_positive(num_threads, "Number of threads")

    instance = ParallelOptions{T}(use_parallel, num_threads, solver_type)
    check_fields(instance, "ParallelOptions")

    return instance
end

@doc """
    DerivativeRecovery{T<:AbstractFloat}

A structure representing derivative recovery.

# Fields
- `recovery_method::Function`: Method for recovering derivatives.
- `recovered_derivatives::Vector{T}`: Recovered derivatives.
"""
struct DerivativeRecovery{T<:AbstractFloat}
    recovery_method::Function
    recovered_derivatives::Vector{T}
end

@doc """
    DerivativeRecovery(recovery_method::Function, recovered_derivatives::Vector{T}) where {T<:AbstractFloat}

Create a new `DerivativeRecovery` instance.

# Arguments
- `recovery_method::Function`: Method for recovering derivatives.
- `recovered_derivatives::Vector{T}`: Recovered derivatives.

# Returns
- A `DerivativeRecovery` instance.

# Errors
- Throws an `ArgumentError` if `recovered_derivatives` are empty or `recovery_method` is `nothing`.
"""
function DerivativeRecovery(recovery_method::Function, recovered_derivatives::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(recovered_derivatives, "Recovered derivatives")
    check_is_nothing(recovery_method, "Recovery method")

    instance = DerivativeRecovery{T}(recovery_method, recovered_derivatives)
    check_fields(instance, "DerivativeRecovery")

    return instance
end
