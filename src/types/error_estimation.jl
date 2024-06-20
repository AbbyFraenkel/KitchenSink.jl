# using DocStringExtensions
@doc """
    ResidualErrorEstimator{T<:AbstractFloat}

A structure representing a residual error estimator.

# Fields
- `residuals::Vector{T}`: Vector of residuals.
"""
struct ResidualErrorEstimator{T<:AbstractFloat} <: AbstractErrorEstimator
    residuals::Vector{T}
end

@doc """
    ResidualErrorEstimator(residuals::Vector{T}) where {T<:AbstractFloat}

Create a new `ResidualErrorEstimator` instance.

# Arguments
- `residuals::Vector{T}`: Vector of residuals.

# Returns
- A `ResidualErrorEstimator` instance.

# Errors
- Throws an `ArgumentError` if `residuals` are empty.
"""
function ResidualErrorEstimator(residuals::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(residuals, "Residuals")

    instance = ResidualErrorEstimator{T}(residuals)
    check_fields(instance, "ResidualErrorEstimator")

    return instance
end

@doc """
    GoalOrientedErrorEstimator{T<:AbstractFloat}

A structure representing a goal-oriented error estimator.

# Fields
- `goal_functional::Function`: Goal functional.
- `error_indicators::Vector{T}`: Vector of error indicators.
"""
struct GoalOrientedErrorEstimator{T<:AbstractFloat} <: AbstractErrorEstimator
    goal_functional::Function
    error_indicators::Vector{T}
end

@doc """
    GoalOrientedErrorEstimator(goal_functional::Function, error_indicators::Vector{T}) where {T<:AbstractFloat}

Create a new `GoalOrientedErrorEstimator` instance.

# Arguments
- `goal_functional::Function`: Goal functional.
- `error_indicators::Vector{T}`: Vector of error indicators.

# Returns
- A `GoalOrientedErrorEstimator` instance.

# Errors
- Throws an `ArgumentError` if `goal_functional` is `nothing` or `error_indicators` are empty.
"""
function GoalOrientedErrorEstimator(goal_functional::Function, error_indicators::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(error_indicators, "Error indicators")
    check_is_nothing(goal_functional, "Goal functional")

    instance = GoalOrientedErrorEstimator{T}(goal_functional, error_indicators)
    check_fields(instance, "GoalOrientedErrorEstimator")

    return instance
end
