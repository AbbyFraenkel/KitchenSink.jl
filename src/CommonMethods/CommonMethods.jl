module CommonMethods

using LinearAlgebra, SparseArrays, StaticArrays
using ..KSTypes, ..SpectralMethods

export create_mesh, create_elements, generate_mesh_nodes, update_tensor_product_masks!
export update_location_matrices!, assemble_system_matrix, assemble_local_system_matrix
export assemble_rhs_vector, assemble_local_rhs_vector, element_size, get_active_indices
export inner_product, total_dofs, element_dofs, create_ocfe_discretization
export update_location_matrix!, get_element_nodes, transform_standard_element, STANDARD_ELEMENT_CACHE
export deactivate_internal_boundary!, scale_standard_element, cache_scaled_element
export create_or_get_standard_element, create_standard_element

# Thread-safe cache for StandardElement instances
global STANDARD_ELEMENT_CACHE = Dict{Tuple{NTuple{N, Int} where N, Int}, StandardElement{T, N} where {T <: Real, N}}()
global STANDARD_ELEMENT_LOCK = ReentrantLock()


"""
    create_mesh(domain::AbstractVector{<:Tuple{T, T}},
                num_elements::AbstractVector{Int},
                polynomial_degree::AbstractVector{Int}) where {T <: Real}

Creates the initial mesh, including elements, tensor product masks, and location matrices.
Uses caching and parallel processing for improved performance.
"""
function create_mesh(domain::AbstractVector{<:Tuple{T, T}},
                     num_elements::AbstractVector{Int},
                     polynomial_degree::AbstractVector{Int}) where {T <: Real}
    N = length(domain)
    @assert length(num_elements) == N "Number of elements must match domain dimensionality"
    @assert length(polynomial_degree) == N "Polynomial degree must match domain dimensionality"
    @assert all(x -> x > 0, num_elements) "Number of elements must be positive integers"
    @assert all(x -> x > 0, polynomial_degree) "Polynomial degree must be positive integers"

    nodes, node_map = generate_mesh_nodes(domain, num_elements, polynomial_degree)
    elements = create_elements(node_map, num_elements, polynomial_degree)

    # Pre-compute standard elements
    unique_configs = unique(zip(polynomial_degree, [1 for _ in 1:length(elements)])) # Example level=1
    for config in unique_configs
        create_or_get_standard_element(config...)
    end

    tensor_product_masks = [BitArray{N}(trues(elem.polynomial_degree .+ 1)) for elem in elements]
    location_matrices = [Dict{Int, Int}() for _ in elements]

    mesh = KSMesh{T, N}(elements, tensor_product_masks, location_matrices, zero(T))
    update_tensor_product_masks!(mesh)
    update_location_matrices!(mesh)
    return mesh
end


"""
    create_or_get_standard_element(polynomial_degree::Tuple{Vararg{Int}}, level::Int)

Creates a StandardElement or retrieves it from cache in a thread-safe manner.
"""
function create_or_get_standard_element(polynomial_degree::Tuple{Vararg{Int}}, level::Int)
    key = (polynomial_degree, level)
    lock(STANDARD_ELEMENT_LOCK) do
        if !haskey(STANDARD_ELEMENT_CACHE, key)
            STANDARD_ELEMENT_CACHE[key] = create_standard_element(polynomial_degree, level)
        end
    end
    return STANDARD_ELEMENT_CACHE[key]
end

"""
    create_standard_element(polynomial_degree::Tuple{Vararg{Int}}, level::Int)

Creates a StandardElement for the given polynomial degree and refinement level.
Uses caching to avoid redundant computations.
"""
function create_standard_element(polynomial_degree::Tuple{Vararg{Int}}, level::Int)
    key = (polynomial_degree, level)
    if haskey(STANDARD_ELEMENT_CACHE, key)
        return STANDARD_ELEMENT_CACHE[key]
    end

    T = Float64  # You might want to make this parametric

    # Create Legendre nodes and weights
    nodes_with_boundary, weights_with_boundary, collocation_points, quad_weights = SpectralMethods.gauss_legendre_with_boundary_nd(maximum(polynomial_degree), length(polynomial_degree))

    # Create differentiation matrices
    diff_matrices = [SpectralMethods.derivative_matrix!(getindex.(collocation_points, i)) for i in 1:length(polynomial_degree)]

    # Create quadrature matrices (if needed)
    quad_matrices = [SpectralMethods.quadrature_matrix(getindex.(quad_weights, i)) for i in 1:length(polynomial_degree)]

    std_elem = StandardElement{T, length(polynomial_degree)}(nodes_with_boundary,
                                     collocation_points,
                                     quad_weights,
                                     diff_matrices,
                                     quad_matrices,
                                     level)

    STANDARD_ELEMENT_CACHE[key] = std_elem
    return std_elem
end

"""
	create_elements(node_map::Dict{NTuple{N, T}, Int},
					num_elements::AbstractVector{Int},
					polynomial_degree::AbstractVector{Int}) where {T <: Real, N}

Creates the elements of a structured mesh based on the provided nodes and precomputed data.
"""
function create_elements(node_map::Dict{NTuple{N, T}, Int}, num_elements::AbstractVector{Int}, polynomial_degree::AbstractVector{Int}) where {T <: Real, N}
    @assert length(num_elements) == length(polynomial_degree) == N "Dimensions mismatch"

    total_elements = prod(num_elements)
    elements = Vector{KSElement{T, N}}(undef, total_elements)

    for idx in 1:total_elements
        element_index = CartesianIndices(Tuple(num_elements))[idx]
        level = 1
        polynomial_degree_vector = collect(polynomial_degree)  # Convert tuple to vector
        element_nodes = get_element_nodes(node_map, element_index, polynomial_degree_vector)
        elements[idx] = KSElement{T, N}(id = idx,
                                        level = level,
                                        polynomial_degree = Tuple(polynomial_degree),
                                        nodes = element_nodes)
    end

    return elements
end
"""
	generate_mesh_nodes(domain::AbstractVector{<:Tuple{T, T}},
						num_elements::AbstractVector{Int},
						polynomial_degree::AbstractVector{Int}) where {T <: Real}

Generates the nodes of a structured mesh using Legendre nodes and weights.
"""
function generate_mesh_nodes(domain::AbstractVector{<:Tuple{T, T}},
                             num_elements::AbstractVector{Int},
                             polynomial_degree::AbstractVector{Int}) where {T <: Real}
    N = length(domain)
    @assert length(num_elements) == N && length(polynomial_degree) == N "Dimensions mismatch"

    nodes_per_dim = num_elements .* polynomial_degree .+ 1
    ranges = [range(d[1], d[2], length = n) for (d, n) in zip(domain, nodes_per_dim)]

    nodes = vec([Tuple(coords) for coords in Iterators.product(ranges...)])
    node_map = Dict(node => i for (i, node) in enumerate(nodes))

    return nodes, node_map
end

"""
	update_tensor_product_masks!(mesh::KSMesh{T, N}) where {T <: Real, N}

Updates the tensor product masks for all elements in the mesh.
"""
function update_tensor_product_masks!(mesh::KSMesh{T, N}) where {T <: Real, N}
    for (i, element) in enumerate(mesh.elements)
        mask = mesh.tensor_product_masks[i]
        fill!(mask, true)

        if element.neighbors !== nothing
            for (dir, neighbor) in enumerate(element.neighbors)
                if neighbor !== nothing && neighbor.level > element.level
                    deactivate_internal_boundary!(mask, dir, neighbor.level - element.level)
                end
            end
        end
    end

    # Second pass to handle equal-level neighbors
    for a in 1:N
        for (i, element) in enumerate(mesh.elements)
            if element.neighbors !== nothing && element.neighbors[a] !== nothing
                neighbor_id = element.neighbors[a]
                neighbor = mesh.elements[neighbor_id]
                if element.level == neighbor.level
                    mesh.tensor_product_masks[i] .|= mesh.tensor_product_masks[neighbor_id]
                end
            end
        end
    end
end

"""
	update_location_matrices!(mesh::KSMesh{T, N}) where {T <: Real, N}


Updates the location matrices for all elements in the mesh, mapping local indices to global indices.
"""

function update_location_matrices!(mesh::KSMesh{T, N}) where {T <: Real, N}
    global_index = 1

    for (i, element) in enumerate(mesh.elements)
        active_indices = get_active_indices(element, mesh)
        update_location_matrix!(mesh.location_matrices[i], active_indices, global_index)
        global_index += length(active_indices)
    end
end
"""
	assemble_system_matrix(mesh::KSMesh{T, N}) where {T <: Real, N}

Assembles the global system matrix for the mesh using precomputed differentiation matrices
from `StandardElement`. Uses parallel processing for improved performance.
"""
function assemble_system_matrix(mesh::KSMesh{T, N}) where {T <: Real, N}
    n_dofs = total_dofs(mesh)
    A = spzeros(T, n_dofs, n_dofs)

    local_matrices = Vector{Tuple{Vector{Int}, SparseMatrixCSC{T, Int}}}(undef, length(mesh.elements))

    for i in eachindex(mesh.elements)
        indices = get_active_indices(mesh.elements[i], mesh)
        A_local = assemble_local_system_matrix(mesh.elements[i], mesh)
        local_matrices[i] = (indices, A_local)
    end

    for (indices, A_local) in local_matrices
        A[indices, indices] += A_local
    end

    return A
end

"""
	assemble_local_system_matrix(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}

Assembles the local system matrix for a given element using precomputed differentiation matrices
from `StandardElement`.
"""
function assemble_local_system_matrix(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
    std_elem = create_or_get_standard_element(element.polynomial_degree, element.level)
    diff_matrices = std_elem.differentiation_matrices

    n_local_dofs = length(std_elem.collocation_points)
    local_matrix = spzeros(T, n_local_dofs, n_local_dofs)
    for i in 1:N
        local_matrix += diff_matrices[i]' * diff_matrices[i]
    end

    return local_matrix
end

"""
	assemble_rhs_vector(mesh::KSMesh{T, N}, f::Function) where {T <: Real, N}

Assembles the global right-hand side (RHS) vector for the mesh.
Uses parallel processing for improved performance.
"""
function assemble_rhs_vector(mesh::KSMesh{T, N}, f::Function) where {T <: Real, N}
    n = total_dofs(mesh)
    b = zeros(T, n)

    local_vectors = Vector{Tuple{Vector{Int}, Vector{T}}}(undef, length(mesh.elements))

    for i in eachindex(mesh.elements)
        indices = get_active_indices(mesh.elements[i], mesh)
        b_local = assemble_local_rhs_vector(mesh.elements[i], mesh, f)
        local_vectors[i] = (indices, b_local)
    end

    for (indices, b_local) in local_vectors
        b[indices] .+= b_local
    end

    return b
end

"""
	assemble_local_rhs_vector(element::KSElement{T, N}, mesh::KSMesh{T, N}, f::Function) where {T <: Real, N}

Assembles the local right-hand side (RHS) vector for a given element using precomputed basis functions
and collocation points from `StandardElement`.
"""
function assemble_local_rhs_vector(element::KSElement{T, N}, mesh::KSMesh{T, N}, f::Function) where {T <: Real, N}
    std_elem = create_or_get_standard_element(element.polynomial_degree, element.level)
    n = length(std_elem.collocation_points)
    b_local = zeros(T, n)

    for (i, coord) in enumerate(std_elem.collocation_points)
        b_local[i] = f(coord...) * inner_product(std_elem.collocation_weights, ones(T, n), element, mesh)
    end

    return b_local
end

"""
	element_size(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}

Computes the size of the specified element based on its nodes.
"""
function element_size(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
    std_elem = create_or_get_standard_element(element.polynomial_degree, element.level)
    min_coords = reduce(min, std_elem.collocation_points)
    max_coords = reduce(max, std_elem.collocation_points)
    lengths = max_coords .- min_coords
    max_length = maximum(lengths)

    return lengths, max_length
end

"""
	get_active_indices(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}

Retrieves the global indices of active degrees of freedom for the specified element.
"""
function get_active_indices(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
    element_id = findfirst(el -> el.id == element.id, mesh.elements)
    if isnothing(element_id)
        throw(ArgumentError("Element with ID $(element.id) not found in mesh."))
    end
    return collect(values(mesh.location_matrices[element_id]))
end

"""
	inner_product(f::AbstractVector{T}, g::AbstractVector{T}, element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}

Computes the inner product of two vectors over the collocation points of the specified element.
"""
function inner_product(f::AbstractVector{T}, g::AbstractVector{T}, element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
    std_elem = create_or_get_standard_element(element.polynomial_degree, element.level)
    if length(f) != length(g) || length(f) != length(std_elem.collocation_points)
        throw(DimensionMismatch("Input vectors must match the number of collocation points."))
    end
    weights = std_elem.collocation_weights
    return sum(f .* g .* weights)
end

"""
	total_dofs(mesh::KSMesh{T, N}) where {T <: Real, N}

Computes the total number of degrees of freedom (DOFs) in the mesh.
"""
function total_dofs(mesh::KSMesh{T, N}) where {T <: Real, N}
    return sum(length, mesh.location_matrices)
end

"""
	create_ocfe_discretization(mesh::KSMesh{T, N}, problem::AbstractKSProblem) where {T <: Real, N}

Creates the orthogonal collocation on finite elements discretization for a given mesh and problem.
"""
function create_ocfe_discretization(mesh::KSMesh{T, N}, problem::AbstractKSProblem) where {T <: Real, N}
    A = assemble_system_matrix(mesh)
    b = assemble_rhs_vector(mesh, problem.equation)
    return A, b
end

"""
	update_location_matrix!(location_matrix::Dict{Int, Int}, active_indices::AbstractVector{Int}, global_index::Threads.Atomic{Int})

Updates the location matrix for an element, mapping local indices to global indices.
Uses atomic operations for thread safety.
"""
function update_location_matrix!(location_matrix::Dict{Int, Int}, active_indices::AbstractVector{Int}, global_index::Int)
    empty!(location_matrix)
    for (local_index, _) in enumerate(active_indices)
        location_matrix[local_index] = global_index
        global_index += 1
    end
    return global_index
end


"""
    get_element_nodes(node_map::Dict{NTuple{N, T}, Int}, element_index::CartesianIndex{N}, polynomial_degree::AbstractVector{Int}) where {T <: Real, N}

Retrieves the nodes for a specific element based on its index and polynomial degree.
"""
function get_element_nodes(node_map::Dict{NTuple{N, T}, Int}, element_index::CartesianIndex{N}, polynomial_degree::AbstractVector{Int}) where {T <: Real, N}
    @assert length(polynomial_degree) == N "Polynomial degree vector length must match the number of dimensions"
    element_nodes = Vector{NTuple{N, T}}()
    for index in CartesianIndices(Tuple(polynomial_degree .+ 1))
        node_index = Tuple(CartesianIndex(Tuple(element_index) .+ Tuple(index) .- Tuple(ones(Int, N))))
        push!(element_nodes, Tuple(map(T, node_map[node_index])))
    end
    return element_nodes
end

"""
    transform_standard_element(std_elem::StandardElement{T, N}, physical_domain::NTuple{N, Tuple{T, T}}, level::Int) where {T <: Real, N}

Transforms a StandardElement to the physical domain for a given element.
"""
function transform_standard_element(std_elem::StandardElement{T, N}, physical_domain::NTuple{N, Tuple{T, T}}, level::Int) where {T <: Real, N}
    # Compute scaling factors
    scaling = [(physical_domain[i][2] - physical_domain[i][1]) / 2 for i in 1:N]

    # Transform points
    transformed_points = [NTuple{N, T}([scaling[j] * (point[j] + 1) + physical_domain[j][1] for j in 1:N]) for point in std_elem.collocation_points]

    # Scale differentiation matrices
    scaled_diff_matrices = [matrix ./ scaling[i] for (i, matrix) in enumerate(std_elem.differentiation_matrices)]

    # Scale quadrature matrices
    scaled_quad_matrices = [matrix .* scaling[i] for (i, matrix) in enumerate(std_elem.quadrature_matrices)]

    return StandardElement(transformed_points, scaled_diff_matrices, scaled_quad_matrices)
end

"""
	deactivate_internal_boundary!(mask::AbstractArray{Bool, N}, direction::Int, level_difference::Int) where {N}

Deactivates the internal boundary in the tensor-product mask for a specified element.
"""
function deactivate_internal_boundary!(mask::AbstractArray{Bool, N}, direction::Int, level_difference::Int) where {N}
    size_shift = level_difference * 2

    ranges = ntuple(d -> 1:size(mask, d), N)
    adjusted_range = 1:(size(mask, direction) - size_shift)
    ranges = Base.setindex(ranges, adjusted_range, direction)

    for idx in CartesianIndices(ranges)
        mask[idx] = false
    end
end

"""
	scale_standard_element(std_elem::StandardElement{T, N}, element_size::NTuple{N, T}) where {T <: Real, N}

Scales a StandardElement's differentiation matrices and collocation points to match a given element's size.
"""
function scale_standard_element(std_elem::StandardElement{T, N}, element_size::NTuple{N, T}) where {T <: Real, N}
    # Scale differentiation matrices
    scaled_diff_matrices = [matrix ./ element_size[i] for (i, matrix) in enumerate(std_elem.differentiation_matrices)]

    # Scale collocation points
    scaled_collocation_points = [NTuple{N, T}([point[i] * element_size[i] for i in 1:N]) for point in std_elem.collocation_points]

    # Create a scaled element
    return StandardElement{T, N}(points_with_boundary = std_elem.points_with_boundary,
                                 collocation_points = scaled_collocation_points,
                                 collocation_weights = std_elem.collocation_weights,
                                 differentiation_matrices = scaled_diff_matrices,
                                 quadrature_matrices = std_elem.quadrature_matrices,
                                 level = std_elem.level)
end
"""
	cache_scaled_element(std_elem::StandardElement{T, N}, element_size::NTuple{N, T}) where {T <: Real, N}

Caches the scaled version of a StandardElement to avoid redundant scaling computations.
"""
function cache_scaled_element(std_elem::StandardElement{T, N}, element_size::NTuple{N, T}) where {T <: Real, N}
    key = (element_size, std_elem.level)
    lock(STANDARD_ELEMENT_LOCK) do
        if !haskey(STANDARD_ELEMENT_CACHE, key)
            STANDARD_ELEMENT_CACHE[key] = scale_standard_element(std_elem, element_size)
        end
    end
    return STANDARD_ELEMENT_CACHE[key]
end

"""
	element_dofs(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}

Get the number of degrees of freedom (DOFs) associated with a specific element.
"""
function element_dofs(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
    element_id = findfirst(isequal(element), mesh.elements)
    if element_id === nothing
        throw(ArgumentError("Element not found in the mesh."))
    end
    return length(mesh.location_matrices[element_id])
end
end # module CommonMethods
