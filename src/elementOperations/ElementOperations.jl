module ElementOperations

using LinearAlgebra
using FastGaussQuadrature
using ..Types

export initialize_element, create_initial_connectivity


"""
    initialize_element(N::Int, p_order::Int, dims::Int, a::Vector{Float64}, b::Vector{Float64}, level::Int)::Element

Initialize an element with given parameters.

# Arguments
- `N::Int`: Number of nodes in each dimension.
- `p_order::Int`: Polynomial order.
- `dims::Int`: Number of dimensions.
- `a::Vector{Float64}`: Lower bounds of the element.
- `b::Vector{Float64}`: Upper bounds of the element.
- `level::Int`: Level of the element in the tree.

# Returns
- `Element`: Initialized element.
"""
function initialize_element(N, p_order, dims, a::Vector{Float64}, b::Vector{Float64})::Element
    # Degree of freedom (DOF) check
    expected_dof = (p_order + 1)^dims
    actual_dof = N^dims
    if actual_dof != expected_dof
        throw(ArgumentError("Incorrect number of degrees of freedom. Expected: $expected_dof, Actual: $actual_dof"))
    end

    nodes = Vector{Vector{Float64}}(undef, dims)
    weights = Vector{Vector{Float64}}(undef, dims)

    for i in 1:dims
        nodes[i], weights[i] = shifted_adapted_gauss_legendre(N, a[i], b[i])
    end
    tensor_data = zeros(Float64, ntuple(_ -> N, dims)...)
    active_basis = trues(N)
    solution = zeros(Float64, ntuple(_ -> N, dims)...)
    residuals = zeros(Float64, ntuple(_ -> N, dims)...)
    return Element(nodes, weights, p_order, tensor_data, active_basis, solution, residuals)
end
"""
    create_initial_connectivity(N::Int, dims::Int)

Create initial connectivity for an element.

# Arguments
- `N::Int`: Number of nodes in each dimension.
- `dims::Int`: Number of dimensions.

# Returns
- `Vector{Tuple{Int, Vararg{Int}}}`: Connectivity of the element.
"""
function create_initial_connectivity(N::Int, dims::Int)
    connectivity = Vector{Tuple{Int,Vararg{Int}}}()

    function add_connectivity(indices)
        push!(connectivity, (indices..., indices[1:(end-1)]..., indices[end] + 1))
    end

    function iterate_dims(dims::Int, indices::Vector{Int})
        if dims == 0
            add_connectivity(indices)
        else
            for i in 1:N
                iterate_dims(dims - 1, [indices...; i])
            end
        end
    end

    iterate_dims(dims, [])
    return connectivity
end
"""
    initialize_tree_node(N, p_order, dims, a, b, level, max_level)

Initialize a tree node with given parameters.

# Arguments
- `N::Int`: Number of nodes in each dimension.
- `p_order::Int`: Polynomial order.
- `dims::Int`: Number of dimensions.
- `a`: Lower bounds of the node.
- `b`: Upper bounds of the node.
- `level::Int`: Level of the node in the tree.
- `max_level::Int`: Maximum level of the tree.

# Returns
- `TreeNode`: Initialized tree node.
"""
function initialize_tree_node(N, p_order, dims, a, b, level, max_level)
    element = initialize_element(N, p_order, dims, a, b)
    children = TreeNode{Float64}[]
    bounds = (a, b)

    node = TreeNode(element, children, bounds)

    if level < max_level
        mid = [(a[i] + b[i]) / 2 for i in 1:dims]
        push!(
            children, initialize_tree_node(N, p_order, dims, a, mid, level + 1, max_level))
        push!(
            children, initialize_tree_node(N, p_order, dims, mid, b, level + 1, max_level))
    end

    return node
end
"""
    build_tree(N::Int, p_order::Int, dims::Int, max_level::Int)

Build a tree with given parameters.

# Arguments
- `N::Int`: Number of nodes in each dimension.
- `p_order::Int`: Polynomial order.
- `dims::Int`: Number of dimensions.
- `max_level::Int`: Maximum level of the tree.

# Returns
- `TreeNode`: Root of the built tree.
"""
function build_tree(N::Int, p_order::Int, dims::Int, max_level::Int)
    a = zeros(Float64, dims)
    b = ones(Float64, dims)
    return initialize_tree_node(N, p_order, dims, a, b, 0, max_level)
end
"""
    collect_elements(root::TreeNode)

Collect all elements in the tree starting from the given root.

# Arguments
- `root::TreeNode`: Root of the tree.

# Returns
- `Vector{Element{Float64, N}}`: Vector of collected elements.
"""
function collect_elements(root::TreeNode)
    elements = []
    if isempty(root.children)
        push!(elements, root.element)
    else
        for child in root.children
            append!(elements, collect_elements(child))
        end
    end
    return elements
end

end
