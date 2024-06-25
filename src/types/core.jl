# Abstract Types
# using DocStringExtensions

@doc """
    AbstractSystem

An abstract type representing a system.
"""
abstract type AbstractSystem end

@doc """
    AbstractMesh

An abstract type representing a mesh.
"""
abstract type AbstractMesh end

@doc """
    AbstractProblem

An abstract type representing a problem.
"""
abstract type AbstractProblem <: AbstractSystem end

@doc """
    AbstractElement

An abstract type representing an element.
"""
abstract type AbstractElement <: AbstractMesh end

@doc """
    AbstractTreeNode

An abstract type representing a tree node.
"""
abstract type AbstractTreeNode <: AbstractElement end

@doc """
    AbstractConstraint

An abstract type representing a constraint.
"""
abstract type AbstractConstraint end

@doc """
    AbstractBoundaryCondition

An abstract type representing a boundary condition.
"""
abstract type AbstractBoundaryCondition end

@doc """
    AbstractObjectiveFunction

An abstract type representing an objective function.
"""
abstract type AbstractObjectiveFunction end

@doc """
    AbstractCostFunction

An abstract type representing a cost function.
"""
abstract type AbstractCostFunction end

@doc """
    AbstractTensorProductMask

An abstract type representing a tensor product mask.
"""
abstract type AbstractTensorProductMask end

@doc """
    AbstractLocationMatrix

An abstract type representing a location matrix.
"""
abstract type AbstractLocationMatrix end

@doc """
    AbstractRefinementStrategy

An abstract type representing a refinement strategy.
"""
abstract type AbstractRefinementStrategy end

@doc """
    AbstractErrorEstimator

An abstract type representing an error estimator.
"""
abstract type AbstractErrorEstimator end

@doc """
    AbstractIntegrationKernel

An abstract type representing an integration kernel.
"""
abstract type AbstractIntegrationKernel end

@doc """
    AbstractBasisFunction

An abstract type representing a basis function.
"""
abstract type AbstractBasisFunction end

@doc """
    AbstractHeiricialBasisFunction

An abstract type representing a hierarchical basis function.
"""
abstract type AbstractHeiricialBasisFunction <: AbstractBasisFunction end

@doc """
    TensorProductMask{T<:AbstractFloat}

A structure representing the tensor product mask used in hierarchical polynomial basis functions.

# Fields
- `masks::Vector{SparseMatrixCSC{T, Int}}`: Vector of SparseMatrixCSC representing the tensor product masks.
"""

mutable struct TensorProductMask{T<:AbstractFloat} <: AbstractTensorProductMask
    masks::Vector{SparseMatrixCSC{T,Int}}
end

function TensorProductMask(masks::Vector{SparseMatrixCSC{T,Int}}) where {T<:AbstractFloat}
    check_empty(masks, "Tensor product masks")
    check_is_nothing(masks, "Tensor product masks")

    instance = TensorProductMask{T}(masks)
    check_fields(instance, "TensorProductMask")

    return instance
end

@doc """
    LocationMatrix{T<:AbstractFloat}

A structure representing the location matrix for mapping local element degrees of freedom to global degrees of freedom.

# Fields
- `matrix::SparseMatrixCSC{T}`: Sparse matrix representing the location matrix.
"""
mutable struct LocationMatrix{T<:AbstractFloat} <: AbstractLocationMatrix
    matrix::SparseMatrixCSC{T}
end

function LocationMatrix(matrix::SparseMatrixCSC{T}) where {T<:AbstractFloat}
    check_is_nothing(matrix, "Location matrix")
    check_empty(matrix, "Location matrix")

    instance = LocationMatrix{T}(matrix)
    check_fields(instance, "LocationMatrix")

    return instance
end

@doc """
    HierarchicalBasisFunction{T<:AbstractFloat}

A structure representing a hierarchical basis function with an associated decay rate.

# Fields
- `id::Int`: Identifier for the basis function.
- `func::Function`: The basis function itself.
- `decay_rate::T`: Decay rate of the basis function.
"""
mutable struct HierarchicalBasisFunction{T<:AbstractFloat}
    id::Int
    func::Function
    decay_rate::T
end

function HierarchicalBasisFunction(id::Int, func::Function, decay_rate::T) where {T<:AbstractFloat}
    check_positive(id, "ID")
    check_non_negative(decay_rate, "Decay rate")

    instance = HierarchicalBasisFunction{T}(id, func, decay_rate)
    check_fields(instance, "HierarchicalBasisFunction")

    return instance
end

@doc """
    DecayRate{T<:AbstractFloat}

A structure representing the decay rate of hierarchical basis functions.

# Fields
- `rates::Vector{T}`: Vector of decay rates.
"""
mutable struct DecayRate{T<:AbstractFloat}
    rates::Vector{T}
end

function DecayRate(rates::Vector{T}) where {T<:AbstractFloat}
    check_empty(rates, "Decay rates")
    check_is_nothing(rates, "Decay rates")

    instance = DecayRate{T}(rates)
    check_fields(instance, "DecayRate")

    return instance
end
@doc """
    AbstractDerivativeMatrix

An abstract type representing a derivative matrix.
"""
abstract type AbstractDerivativeMatrix end

# Mutable Structures

@doc """
    DerivativeMatrix{T<:AbstractFloat}

A structure representing a derivative matrix used for numerical differentiation.

# Fields
- `matrix::Matrix{T}`: Matrix representing the derivative matrix.
- `order::Int`: The order of the derivative (e.g., 1 for first derivative, 2 for second derivative, etc.).
"""
mutable struct DerivativeMatrix{T<:AbstractFloat} <: AbstractDerivativeMatrix
    matrix::Matrix{T}
    order::Int
end

function DerivativeMatrix(matrix::Matrix{T}, order::Int) where {T<:AbstractFloat}
    check_is_nothing(matrix, "Derivative matrix")
    check_empty(matrix, "Derivative matrix")
    check_positive(order, "Order of derivative")

    instance = DerivativeMatrix{T}(matrix, order)
    check_fields(instance, "DerivativeMatrix")

    return instance
end
