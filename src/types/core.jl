# Abstract Types

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
    AbstractHierarchicalBasisFunction

An abstract type representing a hierarchical basis function.
"""
abstract type AbstractHierarchicalBasisFunction <: AbstractBasisFunction end

@doc """
    AbstractPoint

An abstract type representing a point, either a node or a collocation point.
"""
abstract type AbstractPoint end

@doc """
    AbstractNode

An abstract type representing a node.
"""
abstract type AbstractNode <: AbstractPoint end

@doc """
    AbstractCollocationPoint

An abstract type representing a collocation point.
"""
abstract type AbstractCollocationPoint <: AbstractPoint end

# Concrete Types

@doc """
    Node

A structure representing a node in the mesh.

# Fields
- `id::Int`: Identifier for the node.
- `coordinates::Vector{Float64}`: Coordinates of the node.
"""
struct Node <: AbstractNode
    id::Int
    coordinates::Vector{Float64}
    # other node-specific fields
end

@doc """
    CollocationPoint

A structure representing a collocation point in the mesh.

# Fields
- `id::Int`: Identifier for the collocation point.
- `coordinates::Vector{Float64}`: Coordinates of the collocation point.
"""
struct CollocationPoint <: AbstractCollocationPoint
    id::Int
    coordinates::Vector{Float64}
    # other collocation-point-specific fields
end

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


@doc """
    Mesh{T<:AbstractFloat}

A structure representing a mesh.

# Fields
- `elements::Vector{Element{T}}`: List of elements in the mesh.
- `neighbors::Connectivity`: Connectivity information.
- `levels::Vector{Int}`: Levels of elements.
- `is_leaf::Vector{Bool}`: Leaf status of elements.
- `degrees::Vector{Int}`: Degrees of elements.
- `connectivity::Connectivity`: Connectivity information.
- `parallel::Bool`: Flag for parallel computation.
"""
mutable struct Mesh{T<:AbstractFloat} <: AbstractMesh
    elements::Vector{Element{T}}
    neighbors::Connectivity
    levels::Vector{Int}
    is_leaf::Vector{Bool}
    degrees::Vector{Int}
    connectivity::Connectivity
    parallel::Bool
end

@doc """
    Mesh(elements::Vector{Element{T}}, neighbors::Connectivity, levels::Vector{Int}, is_leaf::Vector{Bool}, degrees::Vector{Int}, connectivity::Connectivity, parallel::Bool) where {T<:AbstractFloat}

Create a new `Mesh` instance.

# Arguments
- `elements::Vector{Element{T}}`: List of elements in the mesh.
- `neighbors::Connectivity`: Connectivity information.
- `levels::Vector{Int}`: Levels of elements.
- `is_leaf::Vector{Bool}`: Leaf status of elements.
- `degrees::Vector{Int}`: Degrees of elements.
- `connectivity::Connectivity`: Connectivity information.
- `parallel::Bool`: Flag for parallel computation.

# Returns
- A `Mesh` instance.

# Errors
- Throws an `ArgumentError` if `elements` are empty.
"""
function Mesh(elements::Vector{Element{T}}, neighbors::Connectivity, levels::Vector{Int}, is_leaf::Vector{Bool}, degrees::Vector{Int}, connectivity::Connectivity, parallel::Bool) where {T<:AbstractFloat}
    check_non_empty_elements(elements, "Elements")

    instance = Mesh{T}(elements, neighbors, levels, is_leaf, degrees, connectivity, parallel)
    check_fields(instance, "Mesh")

    return instance
end

@doc """
    FiniteElement

A structure representing a finite element, including nodes and collocation points.

# Fields
- `id::Int`: Identifier for the finite element.
- `nodes::Vector{Node}`: Vector of nodes in the element.
- `collocation_points::Vector{CollocationPoint}`: Vector of collocation points in the element.
"""
mutable struct FiniteElement <: AbstractElement
    id::Int
    nodes::Vector{Node}
    collocation_points::Vector{CollocationPoint}
end

function FiniteElement(id::Int, nodes::Vector{Node}, collocation_points::Vector{CollocationPoint})
    check_positive(id, "ID")
    check_non_empty_elements(nodes, "Nodes")
    check_non_empty_elements(collocation_points, "Collocation points")

    instance = FiniteElement(id, nodes, collocation_points)
    check_fields(instance, "FiniteElement")

    return instance
end

@doc """
    HPRefinement

A structure representing combined h-refinement and p-refinement strategies.

# Fields
- `marked_nodes::Vector{Int}`: Indices of nodes marked for h-refinement.
- `marked_collocation_points::Vector{Int}`: Indices of collocation points marked for h-refinement.
- `new_orders::Vector{Int}`: New orders of elements for p-refinement.
"""
mutable struct HPRefinement <: AbstractRefinementStrategy
    marked_nodes::Vector{Int}
    marked_collocation_points::Vector{Int}
    new_orders::Vector{Int}
end

function HPRefinement(marked_nodes::Vector{Int}, marked_collocation_points::Vector{Int}, new_orders::Vector{Int})
    check_non_empty_elements(marked_nodes, "Marked nodes")
    check_non_empty_elements(marked_collocation_points, "Marked collocation points")
    check_non_empty_elements(new_orders, "New orders")

    instance = HPRefinement(marked_nodes, marked_collocation_points, new_orders)
    check_fields(instance, "HPRefinement")

    return instance
end
