# Structures and Functions

@doc """
    Connectivity

A structure representing the connectivity information of elements in a mesh.

# Fields
- `matrix::Vector{Vector{Int}}`: Connectivity matrix.
"""
mutable struct Connectivity{T<:Int}
    matrix::Vector{Vector{T}}
end

@doc """
    Connectivity(matrix::Vector{Vector{Int}})

Create a new `Connectivity` instance.

# Arguments
- `matrix::Vector{Vector{Int}}`: Connectivity matrix.

# Returns
- A `Connectivity` instance.

# Errors
- Throws an `ArgumentError` if `matrix` contains empty elements.
"""
function Connectivity(matrix::Vector{Vector{T}}) where {T<:Int}
    check_non_empty_elements(matrix, "Connectivity")

    instance = Connectivity{T}(matrix)
    check_fields(instance, "Connectivity")

    return instance
end


@doc """
    BasisFunction

A structure representing a basis function.

# Fields
- `id::Int`: Identifier for the basis function.
- `basis_function::Function`: The basis function itself.
- `removable::Bool`: Indicates if the basis function is removable.
"""
mutable struct BasisFunction <: AbstractBasisFunction
    id::Int
    basis_function::Function
    removable::Bool

    # Inner constructor
    function BasisFunction(id::Int, basis_function::Function, removable::Bool)
        check_positive(id, "Basis function ID")

        instance = new(id, basis_function, removable)
        check_fields(instance, "BasisFunction")

        return instance
    end
end

@doc """
    Element{T<:AbstractFloat}

A structure representing an element in a mesh.

# Fields
- `nodes::Vector{Node{T}}`: List of nodes in the element.
- `collocation_points::Vector{CollocationPoint{T}}`: List of collocation points in the element.
- `basis_functions::Vector{BasisFunction}`: List of basis functions in the element.
- `active_basis_indices::Vector{Int}`: Active basis function indices.
- `removable_basis_indices::Vector{Int}`: Removable basis function indices.
- `parent::Union{Nothing, Element{T}}`: Parent element.
- `level::Int`: Level of the element.
- `tensor_product_masks::TensorProductMask{T}`: Tensor product masks.
- `location_matrix::LocationMatrix{T}`: Location matrix.
- `error_estimate::T`: Error estimate of the element.
- `boundary_information::Dict`: Boundary information.
"""
mutable struct Element{T<:AbstractFloat} <: AbstractElement
    nodes::Vector{Node{T}}
    weights::Vector{Vector{T}}
    collocation_points::Vector{CollocationPoint{T}}
    basis_functions::Vector{BasisFunction}
    active_basis_indices::Vector{Int}
    removable_basis_indices::Vector{Int}
    parent::Union{Nothing,Element{T}}
    level::Int
    tensor_product_masks::TensorProductMask{T}
    location_matrix::LocationMatrix{T}
    error_estimate::T
    boundary_information::Dict
end

# ToDo: compare element structures on implementation

# mutable struct Element{T,N}
#     nodes::Vector{Vector{T}}
#     weights::Vector{Vector{T}}
#     p_order::Int
#     tensor_data::Array{Float64,N}
#     active_basis::BitVector
#     solution::Array{Float64,N}
#     residuals::Array{Float64,N}
#     refined::Bool
#     connectivity::Vector{Tuple{Int,Int}}
# end

@doc """
    Element(nodes::Vector{Node{T}}, collocation_points::Vector{CollocationPoint{T}}, basis_functions::Vector{BasisFunction}, active_basis_indices::Vector{Int}, removable_basis_indices::Vector{Int}, parent::Union{Nothing, Element{T}}, level::Int, tensor_product_masks::TensorProductMask{T}, location_matrix::LocationMatrix{T}, error_estimate::T, boundary_information::Dict) where {T<:AbstractFloat}

Create a new `Element` instance.

# Arguments
- `nodes::Vector{Node{T}}`: List of nodes in the element.
- `collocation_points::Vector{CollocationPoint{T}}`: List of collocation points in the element.
- `basis_functions::Vector{BasisFunction}`: List of basis functions in the element.
- `active_basis_indices::Vector{Int}`: Active basis function indices.
- `removable_basis_indices::Vector{Int}`: Removable basis function indices.
- `parent::Union{Nothing, Element{T}}`: Parent element.
- `level::Int`: Level of the element.
- `tensor_product_masks::TensorProductMask{T}`: Tensor product masks.
- `location_matrix::LocationMatrix{T}`: Location matrix.
- `error_estimate::T`: Error estimate of the element.
- `boundary_information::Dict`: Boundary information.

# Returns
- An `Element` instance.

# Errors
- Throws an `ArgumentError` if `nodes` or `basis_functions` are empty.
"""
function Element(
    nodes::Vector{Node{T}},
    collocation_points::Vector{CollocationPoint{T}},
    basis_functions::Vector{BasisFunction},
    active_basis_indices::Vector{Int},
    removable_basis_indices::Vector{Int},
    parent::Union{Nothing,Element{T}},
    level::Int,
    tensor_product_masks::TensorProductMask{T},
    location_matrix::LocationMatrix{T},
    error_estimate::T,
    boundary_information::Dict,
) where {T<:AbstractFloat}
    check_non_empty_elements(nodes, "Nodes")
    check_non_empty_elements(collocation_points, "Collocation points")
    check_non_empty_elements(basis_functions, "Basis functions")

    instance = Element{T}(
        nodes,
        collocation_points,
        basis_functions,
        active_basis_indices,
        removable_basis_indices,
        parent,
        level,
        tensor_product_masks,
        location_matrix,
        error_estimate,
        boundary_information,
    )
    check_fields(instance, "Element", ["parent"])

    return instance
end

@doc """
    HierarchicalElement{T<:AbstractFloat}

A structure representing a hierarchical element in a mesh.

# Fields
- `nodes::Vector{Node{T}}`: List of nodes in the hierarchical element.
- `collocation_points::Vector{CollocationPoint{T}}`: List of collocation points in the hierarchical element.
- `basis_functions::Vector{BasisFunction}`: List of basis functions in the hierarchical element.
- `active_basis_indices::Vector{Int}`: Active basis function indices.
- `removable_basis_indices::Vector{Int}`: Removable basis function indices.
- `parent::Union{Nothing, HierarchicalElement{T}}`: Parent hierarchical element.
- `level::Int`: Level of the hierarchical element.
- `refinement_level::Int`: Refinement level of the hierarchical element.
- `tensor_product_masks::SparseMatrixCSC{Int}`: Sparse matrix representing tensor product masks.
- `location_matrix::SparseMatrixCSC{Int}`: Sparse matrix representing location matrix.
- `children::Vector{HierarchicalElement{T}}`: Child hierarchical elements.
- `is_leaf::Bool`: Indicates if the hierarchical element is a leaf.
- `neighbors::Vector{Int}`: Neighboring elements.
"""
mutable struct HierarchicalElement{T<:AbstractFloat} <: AbstractElement
    nodes::Vector{Node{T}}
    collocation_points::Vector{CollocationPoint{T}}
    basis_functions::Vector{BasisFunction}
    active_basis_indices::Vector{Int}
    removable_basis_indices::Vector{Int}
    parent::Union{Nothing,HierarchicalElement{T}}
    level::Int
    refinement_level::Int
    tensor_product_masks::SparseMatrixCSC{Int}
    location_matrix::SparseMatrixCSC{Int}
    children::Vector{HierarchicalElement{T}}
    is_leaf::Bool
    neighbors::Vector{Int}
end

@doc """
    HierarchicalElement(nodes::Vector{Node{T}}, collocation_points::Vector{CollocationPoint{T}}, basis_functions::Vector{BasisFunction}, active_basis_indices::Vector{Int}, removable_basis_indices::Vector{Int}, parent::Union{Nothing, HierarchicalElement{T}}, level::Int, refinement_level::Int, tensor_product_masks::SparseMatrixCSC{Int}, location_matrix::SparseMatrixCSC{Int}, children::Vector{HierarchicalElement{T}}, is_leaf::Bool, neighbors::Vector{Int}) where {T<:AbstractFloat}

Create a new `HierarchicalElement` instance.

# Arguments
- `nodes::Vector{Node{T}}`: List of nodes in the hierarchical element.
- `collocation_points::Vector{CollocationPoint{T}}`: List of collocation points in the hierarchical element.
- `basis_functions::Vector{BasisFunction}`: List of basis functions in the hierarchical element.
- `active_basis_indices::Vector{Int}`: Active basis function indices.
- `removable_basis_indices::Vector{Int}`: Removable basis function indices.
- `parent::Union{Nothing, HierarchicalElement{T}}`: Parent hierarchical element.
- `level::Int`: Level of the hierarchical element.
- `refinement_level::Int`: Refinement level of the hierarchical element.
- `tensor_product_masks::SparseMatrixCSC{Int}`: Sparse matrix representing tensor product masks.
- `location_matrix::SparseMatrixCSC{Int}`: Sparse matrix representing location matrix.
- `children::Vector{HierarchicalElement{T}}`: Child hierarchical elements.
- `is_leaf::Bool`: Indicates if the hierarchical element is a leaf.
- `neighbors::Vector{Int}`: Neighboring elements.

# Returns
- A `HierarchicalElement` instance.

# Errors
- Throws an `ArgumentError` if `nodes`, `collocation_points`, or `basis_functions` are empty.
"""
function HierarchicalElement(
    nodes::Vector{Node{T}},
    collocation_points::Vector{CollocationPoint{T}},
    basis_functions::Vector{BasisFunction},
    active_basis_indices::Vector{Int},
    removable_basis_indices::Vector{Int},
    parent::Union{Nothing,HierarchicalElement{T}},
    level::Int,
    refinement_level::Int,
    tensor_product_masks::SparseMatrixCSC{Int},
    location_matrix::SparseMatrixCSC{Int},
    children::Vector{HierarchicalElement{T}},
    is_leaf::Bool,
    neighbors::Vector{Int},
) where {T<:AbstractFloat}
    check_non_empty_elements(nodes, "Nodes")
    check_non_empty_elements(collocation_points, "Collocation points")
    check_non_empty_elements(basis_functions, "Basis functions")

    instance = HierarchicalElement{T}(
        nodes,
        collocation_points,
        basis_functions,
        active_basis_indices,
        removable_basis_indices,
        parent,
        level,
        refinement_level,
        tensor_product_masks,
        location_matrix,
        children,
        is_leaf,
        neighbors,
    )
    check_fields(instance, "HierarchicalElement")

    return instance
end


@doc """
    HierarchicalGridLayer{T<:AbstractFloat}

A structure representing a hierarchical grid layer.

# Fields
- `nodes::Vector{Node{T}}`: List of nodes in the hierarchical grid layer.
- `collocation_points::Vector{CollocationPoint{T}}`: List of collocation points in the hierarchical grid layer.
- `elements::Vector{HierarchicalElement{T}}`: List of hierarchical elements.
- `connectivity::SparseMatrixCSC{Int}`: Connectivity information.
- `levels::Vector{Int}`: Levels of elements.
- `is_leaf::Vector{Bool}`: Leaf status of elements.
- `degrees::Vector{Int}`: Degrees of elements.
- `parent::Union{Nothing, HierarchicalGridLayer{T}}`: Parent hierarchical grid layer.
- `children::Vector{HierarchicalGridLayer{T}}`: Child hierarchical grid layers.
- `location_matrices::Vector{SparseMatrixCSC{Int}}`: Location matrices.
- `hierarchical_relations::Dict{Int, Vector{Int}}`: Hierarchical relations.
"""
mutable struct HierarchicalGridLayer{T<:AbstractFloat} <: AbstractMesh
    nodes::Vector{Node{T}}
    collocation_points::Vector{CollocationPoint{T}}
    elements::Vector{HierarchicalElement{T}}
    connectivity::SparseMatrixCSC{Int}
    levels::Vector{Int}
    is_leaf::Vector{Bool}
    degrees::Vector{Int}
    parent::Union{Nothing,HierarchicalGridLayer{T}}
    children::Vector{HierarchicalGridLayer{T}}
    location_matrices::Vector{SparseMatrixCSC{Int}}
    hierarchical_relations::Dict{Int,Vector{Int}}
end

@doc """
    HierarchicalGridLayer(nodes::Vector{Node{T}}, collocation_points::Vector{CollocationPoint{T}}, elements::Vector{HierarchicalElement{T}}, connectivity::SparseMatrixCSC{Int}, levels::Vector{Int}, is_leaf::Vector{Bool}, degrees::Vector{Int}, parent::Union{Nothing, HierarchicalGridLayer{T}}, children::Vector{HierarchicalGridLayer{T}}, location_matrices::Vector{SparseMatrixCSC{Int}}, hierarchical_relations::Dict{Int, Vector{Int}}) where {T<:AbstractFloat}

Create a new `HierarchicalGridLayer` instance.

# Arguments
- `nodes::Vector{Node{T}}`: List of nodes in the hierarchical grid layer.
- `collocation_points::Vector{CollocationPoint{T}}`: List of collocation points in the hierarchical grid layer.
- `elements::Vector{HierarchicalElement{T}}`: List of hierarchical elements.
- `connectivity::SparseMatrixCSC{Int}`: Connectivity information.
- `levels::Vector{Int}`: Levels of elements.
- `is_leaf::Vector{Bool}`: Leaf status of elements.
- `degrees::Vector{Int}`: Degrees of elements.
- `parent::Union{Nothing, HierarchicalGridLayer{T}}`: Parent hierarchical grid layer.
- `children::Vector{HierarchicalGridLayer{T}}`: Child hierarchical grid layers.
- `location_matrices::Vector{SparseMatrixCSC{Int}}`: Location matrices.
- `hierarchical_relations::Dict{Int, Vector{Int}}`: Hierarchical relations.

# Returns
- A `HierarchicalGridLayer` instance.

# Errors
- Throws an `ArgumentError` if `nodes`, `collocation_points`, or `elements` are empty.
"""
function HierarchicalGridLayer(
    nodes::Vector{Node{T}},
    collocation_points::Vector{CollocationPoint{T}},
    elements::Vector{HierarchicalElement{T}},
    connectivity::SparseMatrixCSC{Int},
    levels::Vector{Int},
    is_leaf::Vector{Bool},
    degrees::Vector{Int},
    parent::Union{Nothing,HierarchicalGridLayer{T}},
    children::Vector{HierarchicalGridLayer{T}},
    location_matrices::Vector{SparseMatrixCSC{Int}},
    hierarchical_relations::Dict{Int,Vector{Int}},
) where {T<:AbstractFloat}
    check_non_empty_elements(nodes, "Nodes")
    check_non_empty_elements(collocation_points, "Collocation points")
    check_non_empty_elements(elements, "Elements")

    instance = HierarchicalGridLayer{T}(
        nodes,
        collocation_points,
        elements,
        connectivity,
        levels,
        is_leaf,
        degrees,
        parent,
        children,
        location_matrices,
        hierarchical_relations,
    )
    check_fields(instance, "HierarchicalGridLayer")

    return instance
end

@doc """
    HRefinement

A structure representing h-refinement strategy for mesh elements.

# Fields
- `marked_elements::Vector{Int}`: Indices of elements marked for refinement.
"""
struct HRefinement <: AbstractRefinementStrategy
    marked_elements::Vector{Int}
end

@doc """
    HRefinement(marked_elements::Vector{Int})

Create a new `HRefinement` instance.

# Arguments
- `marked_elements::Vector{Int}`: Indices of elements marked for refinement.

# Returns
- An `HRefinement` instance.

# Errors
- Throws an `ArgumentError` if `marked_elements` are empty.
"""
function HRefinement(marked_elements::Vector{Int})
    check_non_empty_elements(marked_elements, "Marked elements")

    instance = HRefinement(marked_elements)
    check_fields(instance, "HRefinement")

    return instance
end

@doc """
    PRefinement

A structure representing p-refinement strategy.

# Fields
- `marked_elements::Vector{Int}`: Indices of elements marked for refinement.
- `new_orders::Vector{Int}`: New orders of elements after refinement.
"""
struct PRefinement <: AbstractRefinementStrategy
    marked_elements::Vector{Int}
    new_orders::Vector{Int}
end

@doc """
    PRefinement(marked_elements::Vector{Int}, new_orders::Vector{Int})

Create a new `PRefinement` instance.

# Arguments
- `marked_elements::Vector{Int}`: Indices of elements marked for refinement.
- `new_orders::Vector{Int}`: New orders of elements after refinement.

# Returns
- A `PRefinement` instance.

# Errors
- Throws an `ArgumentError` if `marked_elements` or `new_orders` are empty.
"""
function PRefinement(marked_elements::Vector{Int}, new_orders::Vector{Int})
    check_non_empty_elements(marked_elements, "Marked elements")
    check_non_empty_elements(new_orders, "New orders")

    instance = PRefinement(marked_elements, new_orders)
    check_fields(instance, "PRefinement")

    return instance
end

@doc """
    HPRefinement

A structure representing combined h-refinement and p-refinement strategies.

# Fields
- `marked_elements::Vector{Int}`: Indices of elements marked for h-refinement.
- `new_orders::Vector{Int}`: New orders of elements for p-refinement.
"""
struct HPRefinement <: AbstractRefinementStrategy
    marked_elements::Vector{Int}
    new_orders::Vector{Int}
end

@doc """
    HPRefinement(marked_elements::Vector{Int}, new_orders::Vector{Int})

Create a new `HPRefinement` instance.

# Arguments
- `marked_elements::Vector{Int}`: Indices of elements marked for h-refinement.
- `new_orders::Vector{Int}`: New orders of elements for p-refinement.

# Returns
- An `HPRefinement` instance.

# Errors
- Throws an `ArgumentError` if `marked_elements` or `new_orders` are empty.
"""
function HPRefinement(marked_elements::Vector{Int}, new_orders::Vector{Int})
    check_non_empty_elements(marked_elements, "Marked elements for h-refinement")
    check_non_empty_elements(new_orders, "New orders for p-refinement")

    instance = HPRefinement(marked_elements, new_orders)
    check_fields(instance, "HPRefinement")

    return instance
end
