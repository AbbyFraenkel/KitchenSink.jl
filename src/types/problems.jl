# Mutable Structures

@doc """
    Equation{T<:AbstractFloat}

A structure representing a differential equation.

# Fields
- `equation::Function`: The differential equation function.
"""
mutable struct Equation{T}
    equation::Function
end

@doc """
    Equation(equation::Function) where {T<:AbstractFloat}

Create a new `Equation` instance.

# Arguments
- `equation::Function`: The differential equation function.

# Returns
- An `Equation` instance.
"""
function Equation(equation::Function)
    instance = Equation{T}(equation)
    check_fields(instance, "Equation")
    return instance
end

@doc """
    Domain{T<:AbstractFloat}

A structure representing the domain of the problem.

# Fields
- `coordinates::Vector{Vector{T}}`: Coordinates defining the domain.
"""
mutable struct Domain{T<:AbstractFloat}
    coordinates::Vector{Vector{T}}
end

@doc """
    Domain(coordinates::Vector{Vector{T}}) where {T<:AbstractFloat}

Create a new `Domain` instance.

# Arguments
- `coordinates::Vector{Vector{T}}`: Coordinates defining the domain.

# Returns
- A `Domain` instance.

# Errors
- Throws an `ArgumentError` if `coordinates` are empty.
"""
function Domain(coordinates::Vector{Vector{T}}) where {T<:AbstractFloat}
    check_non_empty_elements(coordinates, "Domain coordinates")

    instance = Domain{T}(coordinates)
    check_fields(instance, "Domain")

    return instance
end

@doc """
    DirichletCondition{T<:AbstractFloat}

A structure representing a Dirichlet boundary condition.

# Fields
- `value::T`: The Dirichlet value.
- `location::Vector{T}`: The location where the condition is applied.
"""
struct DirichletCondition{T<:AbstractFloat} <: AbstractBoundaryCondition
    value::T
    location::Vector{T}
end

@doc """
    DirichletCondition(value::T, location::Vector{T}) where {T<:AbstractFloat}

Create a new `DirichletCondition` instance.

# Arguments
- `value::T`: The Dirichlet value.
- `location::Vector{T}`: The location where the condition is applied.

# Returns
- A `DirichletCondition` instance.

# Errors
- Throws an `ArgumentError` if `location` is empty.
"""
function DirichletCondition(value::T, location::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(location, "Location")

    instance = DirichletCondition{T}(value, location)
    check_fields(instance, "DirichletCondition")

    return instance
end

@doc """
    NeumannCondition{T<:AbstractFloat}

A structure representing a Neumann boundary condition.

# Fields
- `flux::T`: The Neumann flux.
- `location::Vector{T}`: The location where the condition is applied.
"""
struct NeumannCondition{T<:AbstractFloat} <: AbstractBoundaryCondition
    flux::T
    location::Vector{T}
end

@doc """
    NeumannCondition(flux::T, location::Vector{T}) where {T<:AbstractFloat}

Create a new `NeumannCondition` instance.

# Arguments
- `flux::T`: The Neumann flux.
- `location::Vector{T}`: The location where the condition is applied.

# Returns
- A `NeumannCondition` instance.

# Errors
- Throws an `ArgumentError` if `location` is empty.
"""
function NeumannCondition(flux::T, location::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(location, "Location")

    instance = NeumannCondition{T}(flux, location)
    check_fields(instance, "NeumannCondition")

    return instance
end

@doc """
    RobinCondition{T<:AbstractFloat}

A structure representing a Robin boundary condition.

# Fields
- `alpha::T`: The Robin alpha coefficient.
- `beta::T`: The Robin beta coefficient.
- `location::Vector{T}`: The location where the condition is applied.
"""
struct RobinCondition{T<:AbstractFloat} <: AbstractBoundaryCondition
    alpha::T
    beta::T
    location::Vector{T}
end

@doc """
    RobinCondition(alpha::T, beta::T, location::Vector{T}) where {T<:AbstractFloat}

Create a new `RobinCondition` instance.

# Arguments
- `alpha::T`: The Robin alpha coefficient.
- `beta::T`: The Robin beta coefficient.
- `location::Vector{T}`: The location where the condition is applied.

# Returns
- A `RobinCondition` instance.

# Errors
- Throws an `ArgumentError` if `location` is empty.
"""
function RobinCondition(alpha::T, beta::T, location::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(location, "Location")

    instance = RobinCondition{T}(alpha, beta, location)
    check_fields(instance, "RobinCondition")

    return instance
end

@doc """
    BoundaryCondition{T<:AbstractFloat}

A structure representing a boundary condition.

# Fields
- `condition::AbstractBoundaryCondition`: The boundary condition.
"""
mutable struct BoundaryCondition{T<:AbstractFloat}
    condition::AbstractBoundaryCondition
end

@doc """
    BoundaryCondition(condition::AbstractBoundaryCondition) where {T<:AbstractFloat}

Create a new `BoundaryCondition` instance.

# Arguments
- `condition::AbstractBoundaryCondition`: The boundary condition.

# Returns
- A `BoundaryCondition` instance.
"""
function BoundaryCondition(condition::AbstractBoundaryCondition) where {T<:AbstractFloat}
    instance = BoundaryCondition{T}(condition)
    check_fields(instance, "BoundaryCondition")

    return instance
end

@doc """
    Weights{T<:AbstractFloat}

A structure representing the integration weights.

# Fields
- `values::Vector{T}`: Vector of integration weights.
"""
mutable struct Weights{T<:AbstractFloat}
    values::Vector{T}
end

@doc """
    Weights(values::Vector{T}) where {T<:AbstractFloat}

Create a new `Weights` instance.

# Arguments
- `values::Vector{T}`: Vector of integration weights.

# Returns
- A `Weights` instance.

# Errors
- Throws an `ArgumentError` if `values` are empty.
"""
function Weights(values::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(values, "Integration weights")

    instance = Weights{T}(values)
    check_fields(instance, "Weights")

    return instance
end

@doc """
    TimeSpan{T<:AbstractFloat}

A structure representing the time span of a problem.

# Fields
- `span::Vector{T}`: Vector representing the time span.
"""
mutable struct TimeSpan{T<:AbstractFloat}
    span::Vector{T}
end

@doc """
    TimeSpan(span::Vector{T}) where {T<:AbstractFloat}

Create a new `TimeSpan` instance.

# Arguments
- `span::Vector{T}`: Vector representing the time span.

# Returns
- A `TimeSpan` instance.

# Errors
- Throws an `ArgumentError` if `span` is empty.
"""
function TimeSpan(span::Vector{T}) where {T<:AbstractFloat}
    check_non_empty_elements(span, "Time span")

    instance = TimeSpan{T}(span)
    check_fields(instance, "TimeSpan")

    return instance
end

@doc """
    DifferentialVars

A structure representing differential variables.

# Fields
- `vars::Vector{Bool}`: Vector of boolean values indicating differential variables.
"""
mutable struct DifferentialVars
    vars::Vector{Bool}
end

@doc """
    DifferentialVars(vars::Vector{Bool})

Create a new `DifferentialVars` instance.

# Arguments
- `vars::Vector{Bool}`: Vector of boolean values indicating differential variables.

# Returns
- A `DifferentialVars` instance.

# Errors
- Throws an `ArgumentError` if `vars` are empty.
"""
function DifferentialVars(vars::Vector{Bool})
    isempty(vars) && throw(ArgumentError("vars cannot be empty"))

    instance = DifferentialVars(vars)
    return instance
end
