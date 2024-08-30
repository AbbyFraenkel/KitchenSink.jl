module IntergridOperators

using LinearAlgebra
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods

export prolongate!, restrict!, interpolate_between_meshes!, find_parent_element, find_child_elements
export find_containing_element, interpolate_element, restrict_element, interpolate_point
export compute_interpolation_weights, is_parent, validate_mesh_compatibility

"""
	prolongate!(fine_solution::AbstractVector{T}, coarse_solution::AbstractVector{T},
				coarse_mesh::KSMesh{T,N}, fine_mesh::KSMesh{T,N}) where {T<:Real,N}

Prolongate a solution from a coarse mesh to a fine mesh. This function modifies the `fine_solution` in place.

# Arguments
- `fine_solution::AbstractVector{T}`: The vector to store the prolongated solution on the fine mesh.
- `coarse_solution::AbstractVector{T}`: The solution on the coarse mesh.
- `coarse_mesh::KSMesh{T,N}`: The coarse mesh.
- `fine_mesh::KSMesh{T,N}`: The fine mesh.
"""
function prolongate!(fine_solution::AbstractVector{T}, coarse_solution::AbstractVector{T},
					 coarse_mesh::KSMesh{T, N}, fine_mesh::KSMesh{T, N}) where {T <: Real, N}
	for (i, fine_element) in enumerate(fine_mesh.elements)
		coarse_element = find_parent_element(fine_element, coarse_mesh)
		if coarse_element !== nothing
			std_elem = SpectralMethods.get_or_create_standard_element(coarse_element.polynomial_degree)
			transformed_elem = SpectralMethods.transform_standard_element(std_elem, coarse_element.level, coarse_element.polynomial_degree)

			coarse_indices = get_active_indices(coarse_mesh, coarse_element)
			coarse_local_solution = coarse_solution[coarse_indices]

			fine_local_solution = interpolate_element_with_standard(fine_element, coarse_element, coarse_local_solution, transformed_elem)
			fine_indices = get_active_indices(fine_mesh, fine_element)
			fine_solution[fine_indices] .= fine_local_solution
		end
	end
	return fine_solution
end

"""
	restrict!(coarse_solution::AbstractVector{T}, fine_solution::AbstractVector{T},
			  fine_mesh::KSMesh{T,N}, coarse_mesh::KSMesh{T,N}) where {T<:Real,N}

Restrict a solution from a fine mesh to a coarse mesh. This function modifies the `coarse_solution` in place.

# Arguments
- `coarse_solution::AbstractVector{T}`: The vector to store the restricted solution on the coarse mesh.
- `fine_solution::AbstractVector{T}`: The solution on the fine mesh.
- `fine_mesh::KSMesh{T,N}`: The fine mesh.
- `coarse_mesh::KSMesh{T,N}`: The coarse mesh.
"""
function restrict!(coarse_solution::AbstractVector{T}, fine_solution::AbstractVector{T},
				   fine_mesh::KSMesh{T, N}, coarse_mesh::KSMesh{T, N}) where {T <: Real, N}
	for (i, coarse_element) in enumerate(coarse_mesh.elements)
		fine_elements = find_child_elements(coarse_element, fine_mesh)
		if !isempty(fine_elements)
			coarse_local_solution = zeros(T, length(coarse_element.collocation_points))
			for fine_element in fine_elements
				std_elem = SpectralMethods.get_or_create_standard_element(fine_element.polynomial_degree)
				transformed_elem = SpectralMethods.transform_standard_element(std_elem, fine_element.physical_domain, fine_element.level)

				fine_indices = get_active_indices(fine_mesh, fine_element)
				fine_local_solution = fine_solution[fine_indices]

				coarse_local_solution .+= restrict_element_with_standard(fine_element, coarse_element, fine_local_solution, transformed_elem)
			end
			coarse_local_solution ./= length(fine_elements)  # Average the contributions
			coarse_indices = get_active_indices(coarse_mesh, coarse_element)
			coarse_solution[coarse_indices] .= coarse_local_solution
		end
	end
	return coarse_solution
end

"""
	interpolate_between_meshes!(target_solution::AbstractVector{T}, source_solution::AbstractVector{T},
								source_mesh::KSMesh{T,N}, target_mesh::KSMesh{T,N}) where {T<:Real,N}

Interpolate a solution from one mesh to another. This function modifies the `target_solution` in place.

# Arguments
- `target_solution::AbstractVector{T}`: The vector to store the interpolated solution on the target mesh.
- `source_solution::AbstractVector{T}`: The solution on the source mesh.
- `source_mesh::KSMesh{T,N}`: The source mesh.
- `target_mesh::KSMesh{T,N}`: The target mesh.
"""
function interpolate_between_meshes!(target_solution::AbstractVector{T}, source_solution::AbstractVector{T},
									 source_mesh::KSMesh{T, N}, target_mesh::KSMesh{T, N}) where {T <: Real, N}
	for (i, target_element) in enumerate(target_mesh.elements)
		source_element = find_containing_element(target_element.collocation_points[1], source_mesh)
		if source_element !== nothing
			source_local_solution = source_solution[get_active_indices(source_mesh, source_element)]
			target_local_solution = interpolate_element(source_element, target_element, source_local_solution)
			target_solution[get_active_indices(target_mesh, target_element)] .= target_local_solution
		end
	end
	return target_solution
end

"""
	find_parent_element(fine_element::KSElement{T,N}, coarse_mesh::KSMesh{T,N}) where {T<:Real,N}

Find the parent element of a fine element in the coarse mesh.

# Arguments
- `fine_element::KSElement{T,N}`: The fine element.
- `coarse_mesh::KSMesh{T,N}`: The coarse mesh.

# Returns
- `Union{KSElement{T,N}, Nothing}`: The parent element if found, nothing otherwise.
"""
function find_parent_element(fine_element::KSElement{T, N}, coarse_mesh::KSMesh{T, N}) where {T <: Real, N}
	parent_coords = fine_element.get_parent_coords()
	for coarse_element in coarse_mesh.elements
		if coarse_element.matches_coords(parent_coords)
			return coarse_element
		end
	end
	return nothing
end

"""
	find_child_elements(coarse_element::KSElement{T,N}, fine_mesh::KSMesh{T,N}) where {T<:Real,N}

Find the child elements of a coarse element in the fine mesh.

# Arguments
- `coarse_element::KSElement{T,N}`: The coarse element.
- `fine_mesh::KSMesh{T,N}`: The fine mesh.

# Returns
- `Vector{KSElement{T,N}}`: The child elements.
"""
function find_child_elements(coarse_element::KSElement{T, N}, fine_mesh::KSMesh{T, N}) where {T <: Real, N}
	child_coords = coarse_element.get_child_coords()
	return [fine_element for fine_element in fine_mesh.elements if fine_element.matches_coords(child_coords)]
end

"""
	find_containing_element(point::AbstractVector{T}, mesh::KSMesh{T,N}) where {T<:Real,N}

Find the element in the mesh that contains the given point.

# Arguments
- `point::AbstractVector{T}`: The point to locate.
- `mesh::KSMesh{T,N}`: The mesh to search.

# Returns
- `Union{KSElement{T,N}, Nothing}`: The containing element if found, nothing otherwise.
"""
function find_containing_element(point::AbstractVector{T}, mesh::KSMesh{T, N}) where {T <: Real, N}
	for element in mesh.elements
		if CoordinateSystems.is_point_in_element(point, element)
			return element
		end
	end
	return nothing
end

"""
	interpolate_element(source_element::KSElement{T,N}, target_element::KSElement{T,N},
						source_solution::AbstractVector{T}) where {T<:Real,N}

Interpolate the solution from a source element to a target element.

# Arguments
- `source_element::KSElement{T,N}`: The source element.
- `target_element::KSElement{T,N}`: The target element.
- `source_solution::AbstractVector{T}`: The solution on the source element.

# Returns
- `AbstractVector{T}`: The interpolated solution on the target element.
"""
function interpolate_element(source_element::KSElement{T, N}, target_element::KSElement{T, N},
							 source_solution::AbstractVector{T}) where {T <: Real, N}
	std_elem = SpectralMethods.get_or_create_standard_element(source_element.polynomial_degree)
	transformed_elem = SpectralMethods.transform_standard_element(std_elem, source_element.physical_domain, source_element.level)

	target_solution = similar(source_solution, length(target_element.collocation_points))
	for (i, point) in enumerate(target_element.collocation_points)
		target_solution[i] = interpolate_point(point, source_element, source_solution, transformed_elem)
	end
	return target_solution
end

"""
	interpolate_point(point::AbstractVector{T}, element::KSElement{T,N},
					  solution::AbstractVector{T}, std_elem::StandardElement{T,N}) where {T<:Real,N}

Interpolate the solution at a specific point within an element.

# Arguments
- `point::AbstractVector{T}`: The point coordinates.
- `element::KSElement{T,N}`: The element containing the solution.
- `solution::AbstractVector{T}`: The solution on the element.
- `std_elem::StandardElement{T,N}`: The standard element containing precomputed data.

# Returns
- `T`: The interpolated value at the point.
"""
function interpolate_point(point::AbstractVector{T}, element::KSElement{T, N},
						   solution::AbstractVector{T}, std_elem::StandardElement{T, N}) where {T <: Real, N}
	basis_values = SpectralMethods.evaluate_basis_functions(std_elem, [point])
	return dot(basis_values[1, :], solution)
end

"""
	is_parent(coarse_element::KSElement{T,N}, fine_element::KSElement{T,N}) where {T<:Real,N}

Check if a coarse element is the parent of a fine element.

# Arguments
- `coarse_element::KSElement{T,N}`: The potential parent element.
- `fine_element::KSElement{T,N}`: The potential child element.

# Returns
- `Bool`: True if coarse_element is the parent of fine_element, false otherwise.
"""
function is_parent(coarse_element::KSElement{T, N}, fine_element::KSElement{T, N}) where {T <: Real, N}
	return all(CoordinateSystems.is_point_in_element(point, coarse_element) for point in fine_element.collocation_points)
end

"""
	validate_mesh_compatibility(mesh1::KSMesh{T,N}, mesh2::KSMesh{T,N}) where {T<:Real,N}

Validate that two meshes are compatible for intergrid operations.

# Arguments
- `mesh1::KSMesh{T,N}`: The first mesh.
- `mesh2::KSMesh{T,N}`: The second mesh.

# Throws
- `ArgumentError`: If the meshes are not compatible.
"""
function validate_mesh_compatibility(mesh1::KSMesh{T, N}, mesh2::KSMesh{T, N}) where {T <: Real, N}
	if mesh1.dimensions != mesh2.dimensions
		throw(ArgumentError("Meshes must have the same dimension"))
	end
	if typeof(mesh1.coordinate_system) != typeof(mesh2.coordinate_system)
		throw(ArgumentError("Meshes must use the same coordinate system"))
	end
end

"""
	get_active_indices(mesh::KSMesh{T,N}, element::KSElement{T,N}) where {T<:Real,N}

Helper function to get active indices for an element in a mesh.
"""
function get_active_indices(mesh::KSMesh{T, N}, element::KSElement{T, N}) where {T <: Real, N}
	element_index = findfirst(e -> e.id == element.id, mesh.elements)
	return collect(values(mesh.location_matrices[element_index]))
end

"""
	compute_interpolation_weights(point::AbstractVector{T}, element::KSElement{T,N}) where {T<:Real,N}

Compute the interpolation weights for a point within an element.

# Arguments
- `point::AbstractVector{T}`: The coordinates of the point to compute weights for.
- `element::KSElement{T,N}`: The element containing the point.

# Returns
- `AbstractVector{T}`: The interpolation weights.
"""
function compute_interpolation_weights(point::AbstractVector{T}, element::KSElement{T, N}) where {T <: Real, N}
	cache_key = (point, element.id)  # Create a cache key based on point coordinates and element ID
	if haskey(CommonMethods.weight_cache, cache_key)
		return CommonMethods.weight_cache[cache_key]  # Return cached weights if they exist
	end

	weights = Vector{T}(undef, length(element.polynomial_degree))
	for (i, basis) in enumerate(element.polynomial_degree)
		weights[i] = element.legendre_decay_rate * point[i]
	end

	CommonMethods.weight_cache[cache_key] = weights  # Cache the computed weights
	return weights
end

"""
	restrict_element(fine_element::KSElement{T,N}, coarse_element::KSElement{T,N},
					 fine_solution::AbstractVector{T}) where {T<:Real,N}

Restrict the solution from a fine element to a coarse element.

# Arguments
- `fine_element::KSElement{T,N}`: The fine element.
- `coarse_element::KSElement{T,N}`: The coarse element.
- `fine_solution::AbstractVector{T}`: The solution on the fine element.

# Returns
- `AbstractVector{T}`: The restricted solution on the coarse element.
"""
function restrict_element(fine_element::KSElement{T, N}, coarse_element::KSElement{T, N},
						  fine_solution::AbstractVector{T}) where {T <: Real, N}
	coarse_solution = Vector{T}(undef, length(coarse_element.polynomial_degree))
	std_elem = SpectralMethods.get_or_create_standard_element(fine_element.polynomial_degree)

	for i in 1:length(coarse_element.polynomial_degree)
		point = coarse_element.polynomial_degree[i]
		weights = compute_interpolation_weights(point, fine_element)
		coarse_solution[i] = dot(weights, fine_solution)
	end

	return coarse_solution
end

end # module IntergridOperators
