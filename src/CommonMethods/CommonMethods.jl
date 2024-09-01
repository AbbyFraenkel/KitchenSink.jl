module CommonMethods

using LinearAlgebra, SparseArrays, StaticArrays
using ..KSTypes, ..SpectralMethods, ..CoordinateSystems

export create_mesh, update_tensor_product_masks!, update_location_matrices!
export assemble_system_matrix, assemble_local_system_matrix
export assemble_rhs_vector, assemble_local_rhs_vector, get_active_indices
export inner_product, total_dofs, element_dofs, create_ocfe_discretization
export update_location_matrix!, create_or_get_standard_element
export deactivate_internal_boundary!, cache_scaled_element
# Thread-safe cache for StandardElement instances
global STANDARD_ELEMENT_CACHE = Dict{Tuple{NTuple{N, Int} where N, Int}, StandardElement{T, N} where {T <: Real, N}}()
global STANDARD_ELEMENT_LOCK = ReentrantLock()

"""
	create_mesh(domain::AbstractVector{<:Tuple{T, T}},
				num_elements::AbstractVector{Int},
				polynomial_degree::AbstractVector{Int},
				coord_system::AbstractKSCoordinateSystem;
				use_caching::Bool = true) where {T <: Real}

Creates the initial mesh, including elements, tensor product masks, and location matrices.
Optionally uses caching for improved performance.
"""

function create_mesh(domain::Union{AbstractVector{<:Tuple{T, T}}, Tuple{Vararg{Tuple{T, T}}}},
					 num_elements::AbstractVector{<:Integer},
					 polynomial_degree::AbstractVector{<:Integer},
					 coord_system::AbstractKSCoordinateSystem;
					 use_caching::Bool = true) where {T <: Real}
	N = length(domain)
	@assert length(num_elements)==N "Number of elements must match domain dimensionality"
	@assert length(polynomial_degree)==N "Polynomial degree must match domain dimensionality"
	@assert all(x -> x > 0, num_elements) "Number of elements must be positive integers"
	@assert all(x -> x > 0, polynomial_degree) "Polynomial degree must be positive integers"

	if use_caching
		mesh = create_mesh_with_caching(domain, num_elements, polynomial_degree, coord_system)
	else
		mesh = SpectralMethods.create_ocfe_mesh(coord_system, num_elements, polynomial_degree, 0, N)
	end

	# Ensure mesh is a valid KSMesh
	@assert mesh isa KSMesh{T, N} "Expected mesh to be of type KSMesh{T, N}"

	# Update tensor product masks and location matrices
	update_tensor_product_masks!(mesh)
	update_location_matrices!(mesh)

	return mesh
end

"""
Helper function to create mesh with caching
"""
function create_mesh_with_caching(domain::Union{AbstractVector{<:Tuple{T, T}}, Tuple{Vararg{Tuple{T, T}}}},
								  num_elements::AbstractVector{<:Integer},
								  polynomial_degree::AbstractVector{<:Integer},
								  coord_system::AbstractKSCoordinateSystem) where {T <: Real}
	N = length(domain)

	# Create base mesh using create_ocfe_mesh
	mesh = SpectralMethods.create_ocfe_mesh(coord_system, num_elements, polynomial_degree, 0, N)

	# Pre-compute and cache standard elements
	unique_configs = unique(zip(polynomial_degree, [1 for _ in 1:length(mesh.elements)])) # Example level=1
	for config in unique_configs
		# Wrap polynomial_degree in a tuple
		create_or_get_standard_element((config[1],), config[2])
	end

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
Uses SpectralMethods functions for node and matrix generation.
"""
function create_standard_element(polynomial_degree::Tuple{Vararg{Int}}, level::Int)
	T = Float64  # You might want to make this parametric

	# Create Legendre nodes and weights using SpectralMethods
	nodes_with_boundary, weights_with_boundary, collocation_points, quad_weights = SpectralMethods.gauss_legendre_with_boundary_nd(maximum(polynomial_degree), length(polynomial_degree))

	# Create differentiation matrices using SpectralMethods
	diff_matrices = [SpectralMethods.derivative_matrix!(getindex.(nodes_with_boundary, i))
					 for i in 1:length(polynomial_degree)]

	# Create quadrature matrices using SpectralMethods
	quad_matrices = [SpectralMethods.quadrature_matrix!(getindex.(quad_weights, i))
					 for i in 1:length(polynomial_degree)]

	return StandardElement{T, length(polynomial_degree)}(nodes_with_boundary,
														 collocation_points,
														 quad_weights,
														 diff_matrices,
														 quad_matrices,
														 level)
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
	assemble_system_matrix(mesh::KSMesh{T, N}, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}

Assembles the global system matrix for the mesh using precomputed differentiation matrices
from `StandardElement`. Uses parallel processing for improved performance.
"""
function assemble_system_matrix(mesh::KSMesh{T, N}, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}
    n_dofs = total_dofs(mesh)
    A = spzeros(T, n_dofs, n_dofs)  # keep this as the default behavior
    local_matrices = Vector{Tuple{Vector{Int}, AbstractMatrix{T}}}(undef, length(mesh.elements))

    for i in eachindex(mesh.elements)
        indices = get_active_indices(mesh.elements[i], mesh)
        A_local = assemble_local_system_matrix(mesh.elements[i], mesh, coord_system)
        local_matrices[i] = (indices, A_local)
    end

    for (indices, A_local) in local_matrices
        A[indices, indices] += A_local
    end

    return A
end


"""
	assemble_local_system_matrix(element::KSElement{T, N}, mesh::KSMesh{T, N}, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}

Assembles the local system matrix for a given element using precomputed differentiation matrices
from `StandardElement` and taking into account the coordinate system.
"""
function assemble_local_system_matrix(element::KSElement{T, N}, mesh::KSMesh{T, N}, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}
	std_elem = create_or_get_standard_element(element.polynomial_degree, element.level)
	diff_matrices = std_elem.differentiation_matrices

	n_local_dofs = length(std_elem.collocation_points)
	local_matrix = spzeros(T, n_local_dofs, n_local_dofs)

	# Compute Jacobian for the element
	jacobian = CoordinateSystems.compute_jacobian(element.nodes[1], coord_system)

	for i in 1:N
		transformed_diff_matrix = jacobian * diff_matrices[i]
		local_matrix += transformed_diff_matrix' * transformed_diff_matrix
	end

	return local_matrix
end

"""
	assemble_rhs_vector(mesh::KSMesh{T, N}, f::Function, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}

Assembles the global right-hand side (RHS) vector for the mesh.
Uses parallel processing for improved performance and takes into account the coordinate system.
"""
function assemble_rhs_vector(mesh::KSMesh{T, N}, f::Function, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}
	n = total_dofs(mesh)
	b = zeros(T, n)

	local_vectors = Vector{Tuple{Vector{Int}, Vector{T}}}(undef, length(mesh.elements))

	for i in eachindex(mesh.elements)
		indices = get_active_indices(mesh.elements[i], mesh)
		b_local = assemble_local_rhs_vector(mesh.elements[i], mesh, f, coord_system)
		local_vectors[i] = (indices, b_local)
	end

	for (indices, b_local) in local_vectors
		b[indices] .+= b_local
	end

	return b
end

"""
	assemble_local_rhs_vector(element::KSElement{T, N}, mesh::KSMesh{T, N}, f::Function, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}

Assembles the local right-hand side (RHS) vector for a given element using precomputed basis functions
and collocation points from `StandardElement`, taking into account the coordinate system.
"""
function assemble_local_rhs_vector(element::KSElement{T, N}, mesh::KSMesh{T, N}, f::Function, coord_system::AbstractKSCoordinateSystem) where {T <: Real, N}
	std_elem = create_or_get_standard_element(element.polynomial_degree, element.level)
	n = length(std_elem.collocation_points)
	b_local = zeros(T, n)

	# Transform collocation points to physical coordinates
	physical_points = [CoordinateSystems.map_from_reference_element(coord, coord_system) for coord in std_elem.collocation_points]

	for (i, coord) in enumerate(physical_points)
		b_local[i] = f(coord...) * inner_product(std_elem.collocation_weights, ones(T, n), element, mesh)
	end

	return b_local
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
	A = assemble_system_matrix(mesh, problem.coord_system)
	b = assemble_rhs_vector(mesh, problem.equation, problem.coord_system)
	return A, b
end

"""
	update_location_matrix!(location_matrix::Dict{Int, Int}, active_indices::AbstractVector{Int}, global_index::Int)

Updates the location matrix for an element, mapping local indices to global indices.
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
	scale_standard_element(std_elem::StandardElement{T, N}, element_size::NTuple{N, T}) where {T <: Real, N}

Scales a StandardElement's differentiation matrices and collocation points to match a given element's size.
"""
function create_standard_element(polynomial_degree::Tuple{Vararg{Int}}, level::Int)
    T = Float64  # You might want to make this parametric

    # Create Legendre nodes and weights using SpectralMethods
    nodes_with_boundary, weights_with_boundary, collocation_points, quad_weights = SpectralMethods.gauss_legendre_with_boundary_nd(maximum(polynomial_degree), length(polynomial_degree))

    # Create differentiation matrices using SpectralMethods
    diff_matrices = [SpectralMethods.derivative_matrix!(getindex.(nodes_with_boundary, i))
                     for i in 1:length(polynomial_degree)]

    # Create quadrature matrices using SpectralMethods
    quad_matrices = [SpectralMethods.quadrature_matrix!(getindex.(quad_weights, i))
                     for i in 1:length(polynomial_degree)]

    return StandardElement{T, length(polynomial_degree)}(nodes_with_boundary,
                                                         collocation_points,
                                                         weights_with_boundary,
                                                         diff_matrices,
                                                         quad_matrices,
                                                         level)
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

# Helper function to check if a value is close to zero
is_near_zero(x::Number) = abs(x) < eps(typeof(x))

end # module CommonMethods
