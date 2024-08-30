module Preprocessing

using LinearAlgebra, SparseArrays, StaticArrays
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..ErrorEstimation
using Base.Threads

export preprocess_mesh, generate_initial_mesh, refine_mesh, estimate_mesh_error
export create_OCFE_discretization, adapt_mesh!, update_tensor_product_masks_with_trunk!
export update_mesh_connectivity!

"""
	preprocess_mesh(problem::AbstractKSProblem,
					domain::AbstractVector{Tuple{T, T}},
					coord_system::AbstractKSCoordinateSystem,
					num_elements::AbstractVector{Int};
					polynomial_degree::Union{Int, Nothing} = nothing,
					max_levels::Int,
					tolerance::T) where {T <: Real}

Preprocess the mesh for the given problem by creating an initial mesh and refining it based on error estimates.
"""
function preprocess_mesh(problem::AbstractKSProblem,
						 domain::AbstractVector{Tuple{T, T}},
						 coord_system::AbstractKSCoordinateSystem,
						 num_elements::AbstractVector{Int};
						 polynomial_degree::Union{Int, Nothing} = nothing,
						 max_levels::Int,
						 tolerance::T) where {T <: Real}

	# Check dimensionality
	check_dimensionality(domain, coord_system)

	# Generate the initial mesh
	mesh = generate_initial_mesh(domain, coord_system, num_elements, polynomial_degree)

	# Refine the mesh iteratively based on error estimates
	for _ in 1:max_levels
		error_estimates = estimate_mesh_error(mesh, problem)
		mesh = refine_mesh(mesh, error_estimates, tolerance)
	end

	return mesh
end

"""
	generate_initial_mesh(domain::AbstractVector{Tuple{T, T}},
						  coord_system::AbstractKSCoordinateSystem,
						  num_elements::AbstractVector{Int},
						  polynomial_degree::Union{Int, Nothing} = nothing) where {T <: Real}

Generate the initial mesh for the given problem domain and coordinate system.
"""
function generate_initial_mesh(domain::AbstractVector{Tuple{T, T}},
							   coord_system::AbstractKSCoordinateSystem,
							   num_elements::AbstractVector{Int},
							   polynomial_degree::Union{Int, Nothing} = nothing) where {T <: Real}

	# Check dimensionality
	check_dimensionality(domain, coord_system)

	if polynomial_degree === nothing
		polynomial_degree = 1
	end

	# Generate mesh nodes and elements
	nodes = CommonMethods.generate_mesh_nodes(domain, coord_system, num_elements)
	elements = CommonMethods.create_elements(nodes, num_elements, polynomial_degree, coord_system)

	# Create and cache StandardElement
	degree_tuple = ntuple(_ -> polynomial_degree, N)
	std_elem = SpectralMethods.create_standard_element(degree_tuple)
	SpectralMethods.standard_element_cache[degree_tuple] = std_elem

	# Construct the initial mesh
	mesh = KSMesh{T, N}(elements,
						[trues(tuple(fill(polynomial_degree + 1, length(domain))...)) for _ in elements],
						[Dict{Int, Int}() for _ in elements],
						KSBasisFunction[],  # Assuming basis functions are initially empty or passed in
						zero(T))

	update_tensor_product_masks!(mesh)
	update_location_matrices!(mesh)

	return mesh
end

"""
	refine_mesh(mesh::KSMesh{T, N},
				error_estimates::AbstractVector{T},
				tolerance::T) where {T <: Real, N}

Refine the mesh based on the provided error estimates and tolerance.
"""
function refine_mesh(mesh::KSMesh{T, N},
					 error_estimates::AbstractVector{T},
					 tolerance::T) where {T <: Real, N}
	new_elements = Vector{KSElement{T, N}}()

	# Refine each element based on the error estimate
	for (i, element) in enumerate(mesh.elements)
		if error_estimates[i] > tolerance
			refined_elements = refine_element!(element)
			append!(new_elements, refined_elements)
		else
			push!(new_elements, element)
		end
	end

	# Construct the refined mesh
	refined_mesh = KSMesh{T, N}(new_elements,
								[trues(tuple(fill(el.polynomial_degree + 1, length(mesh.elements[1].collocation_points))...)) for el in new_elements],
								[Dict{Int, Int}() for _ in new_elements],
								mesh.basis_functions,  # Reusing or updating basis functions as necessary
								zero(T))

	update_tensor_product_masks!(refined_mesh)
	update_location_matrices!(refined_mesh)

	return refined_mesh
end

"""
	refine_element!(element::KSElement{T, N}) -> Vector{KSElement{T, N}}

Refine the given element.
"""
function refine_element!(element::KSElement{T, N}) where {T <: Real, N}
	new_elements = Vector{KSElement{T, N}}()

	# Perform h-refinement by splitting the element into smaller elements
	mid_points = [(element.nodes[1][1:(end - 1)] + element.nodes[1][2:end]) / 2 for i in 1:length(element.nodes)]

	# Further refinement logic based on your specific requirements (e.g., element splitting)
	# Placeholder for splitting elements and creating new ones
	# ...

	return new_elements
end

"""
	estimate_mesh_error(mesh::KSMesh{T, N}, problem::AbstractKSProblem) -> Vector{T}

Estimate the error for each element in the mesh based on the given problem.
"""
function estimate_mesh_error(mesh::KSMesh{T, N}, problem::AbstractKSProblem) where {T <: Real, N}
	error_estimates = Vector{T}(undef, length(mesh.elements))
	for i in eachindex(mesh.elements)
		error_estimates[i] = ErrorEstimation.estimate_error(mesh.elements[i], problem)
	end
	return error_estimates
end

"""
	create_OCFE_discretization(mesh::KSMesh{T, N},
							   problem::AbstractKSProblem,
							   max_derivative_order::Int) where {T <: Real, N}

Create an OCFE (Orthogonal Collocation on Finite Elements) discretization for the given problem.
"""
function create_OCFE_discretization(mesh::KSMesh{T, N},
									problem::AbstractKSProblem,
									max_derivative_order::Int) where {T <: Real, N}
	n = CommonMethods.total_dofs(mesh)
	A = spzeros(T, n, n)

	# Assemble the system matrix
	for element in mesh.elements
		A_local = CommonMethods.assemble_local_system_matrix(element)
		indices = CommonMethods.get_active_indices(mesh, element)
		A[indices, indices] += A_local
	end

	return A
end

"""
	adapt_mesh!(mesh::KSMesh{T, N},
				problem::AbstractKSProblem,
				tolerance::T) where {T <: Real, N}

Adapt the mesh based on error estimates and a given tolerance.
"""
function adapt_mesh!(mesh::KSMesh{T, N},
					 problem::AbstractKSProblem,
					 tolerance::T) where {T <: Real, N}
	error_estimates = estimate_mesh_error(mesh, problem)

	new_elements = Vector{KSElement{T, N}}()

	# Adapt each element based on the error estimate
	for (i, element) in enumerate(mesh.elements)
		if error_estimates[i] > tolerance
			refined_elements = refine_element!(element)
			append!(new_elements, refined_elements)
		else
			push!(new_elements, element)
		end
	end

	# Update mesh elements and connectivity
	mesh.elements = new_elements
	mesh.tensor_product_masks = [trues(tuple(fill(el.polynomial_degree + 1, length(mesh.elements[1].collocation_points))...)) for el in new_elements]
	mesh.location_matrices = [Dict{Int, Int}() for _ in 1:length(new_elements)]

	update_mesh_connectivity!(mesh)
	update_tensor_product_masks!(mesh)
	update_location_matrices!(mesh)
end

"""
	update_tensor_product_masks_with_trunk!(mesh::KSMesh{T, N}) where {T, N}

Update the tensor product masks with trunk for the given mesh.
"""
function update_tensor_product_masks_with_trunk!(mesh::KSMesh{T, N}) where {T, N}
	for (i, element) in enumerate(mesh.elements)
		mask = trues(tuple(fill(element.polynomial_degree + 1, length(element.collocation_points))...))
		mesh.tensor_product_masks[i] = mask
	end

	update_tensor_product_masks!(mesh)
end

"""
	update_mesh_connectivity!(mesh::KSMesh{T, N}) where {T, N}

Update the connectivity information of the mesh elements.
"""
function update_mesh_connectivity!(mesh::KSMesh{T, N}) where {T, N}
	for element in mesh.elements
		element.neighbors = find_neighboring_elements(element, mesh)
	end
end

"""
	find_neighboring_elements(element::KSElement{T, N},
							  mesh::KSMesh{T, N}) where {T <: Real, N}

Find the neighboring elements of a given element in the mesh.
"""
function find_neighboring_elements(element::KSElement{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
	neighbors = Vector{Union{Nothing, KSElement{T, N}}}(undef, 2^N)

	for (i, other_element) in enumerate(mesh.elements)
		if other_element.id != element.id
			for (j, face) in enumerate(element_faces(element))
				if is_face_adjacent(face, other_element)
					neighbors[j] = other_element
					break
				end
			end
		end
	end

	return neighbors
end

"""
	element_faces(element::KSElement{T, N}) where {T <: Real, N}

Compute the faces of a given element.
"""
function element_faces(element::KSElement{T, N}) where {T <: Real, N}
	faces = Vector{Vector{Vector{T}}}()
	dim = length(element.collocation_points[1])

	for i in 1:dim
		for j in 0:1
			if j == 0
				face_points = filter(p -> p[i] ≈ minimum(getindex.(getfield.(element.collocation_points, :data), i)), element.collocation_points)
			else
				face_points = filter(p -> p[i] ≈ maximum(getindex.(getfield.(element.collocation_points, :data), i)), element.collocation_points)
			end
			push!(faces, face_points)
		end
	end

	return faces
end

"""
	is_face_adjacent(face::Vector{Vector{T}}, element::KSElement{T, N}) where {T <: Real, N}

Check if a given face is adjacent to an element.
"""
function is_face_adjacent(face::Vector{Vector{T}}, element::KSElement{T, N}) where {T <: Real, N}
	for element_face in element_faces(element)
		if all(p -> any(q -> p ≈ q, face), element_face)
			return true
		end
	end
	return false
end

end # module Preprocessing
