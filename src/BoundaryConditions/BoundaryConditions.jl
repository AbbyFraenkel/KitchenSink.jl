module BoundaryConditions

using LinearAlgebra, StaticArrays
using ..KSTypes, ..CoordinateSystems, ..CacheManagement, ..NumericUtilities

# Core boundary condition functions
export apply_boundary_condition,
	get_boundary_nodes,
	get_local_boundary_nodes,
	verify_boundary_condition

# Normal vector computation
export compute_normal_vector_cartesian,
	compute_normal_vector

# Direction parsing and validation
export parse_direction,
	validate_boundary_direction,
	check_boundary_region

# Boundary condition specific helpers
export apply_dirichlet_bc,
	apply_neumann_bc,
	apply_robin_bc,
	verify_bc_coefficients

# Node mapping and coordinate helpers
export get_node_coordinates,
	map_local_to_global_index,
	get_boundary_faces

# Validation utilities
export verify_bc_consistency,
	validate_boundary_values,
	check_boundary_coverage,
	verify_normal_vectors
@doc raw"""
	apply_boundary_condition(bc::AbstractKSBoundaryCondition, x::Vector{T}) where T <: Number

Applies the boundary condition to a given position vector.

# Arguments
- `bc::AbstractKSBoundaryCondition`: The boundary condition to apply.
- `x::Vector{T}`: The position vector.

# Returns
- The result of applying the boundary condition.

# Mathematical Description
For Dirichlet boundary conditions: \[ \text{result} = \text{bc.boundary\_value}(x) \]

For Neumann boundary conditions: \[ \text{normal} =
\text{compute\_normal\_vector\_cartesian}(x) \] \[ \text{flux\_vector} =
\text{bc.flux\_value}(x) \] \[ \text{result} = \text{dot}(\text{flux\_vector}, \text{normal})
\]

For Robin boundary conditions: \[ \text{normal\_tuple} =
\text{compute\_normal\_vector}(\text{coord\_sys}, \text{Tuple}(x)) \] \[
\text{normal} = \text{collect}(\text{normal\_tuple}) \] \[ \alpha =
\text{bc.dirichlet\_coefficient}(x) \] \[ \beta = \text{bc.neumann\_coefficient}(x) \] \[ g =
\text{bc.boundary\_value}(x) \] If \(\alpha\), \(\beta\), and \(g\) are real numbers: \[
\text{result}(u) = \alpha \cdot u + \beta \cdot \text{dot}(\text{normal},
\text{ones}(\text{coord\_dim})) - g \] If \(\alpha\), \(\beta\), and \(g\) are vectors: \[
\text{result}(u) = \alpha \cdot u + (\beta \cdot \text{normal}) - g \]
"""
function apply_boundary_condition(
	bc::AbstractKSBoundaryCondition,
	x::Vector{T}) where {T <: Number}
	if isempty(x)
		throw(ArgumentError("Position vector x cannot be empty."))
	end

	# Add validation for invalid coordinate values
	if any(isnan, x) || any(isinf, x)
		throw(ArgumentError("Position vector contains invalid values (NaN or Inf)"))
	end

	coord_sys = bc.coordinate_system
	coord_dim = if isa(coord_sys, KSCartesianCoordinates)
		count(identity, coord_sys.active)
	elseif isa(coord_sys, KSPolarCoordinates)
		1 + (coord_sys.theta !== nothing ? 1 : 0)
	elseif isa(coord_sys, KSCylindricalCoordinates)
		1 + (coord_sys.theta !== nothing ? 1 : 0) + (coord_sys.z !== nothing ? 1 : 0)
	elseif isa(coord_sys, KSSphericalCoordinates)
		1 + (coord_sys.theta !== nothing ? 1 : 0) + (coord_sys.phi !== nothing ? 1 : 0)
	else
		throw(ArgumentError("Unsupported coordinate system type: $(typeof(coord_sys))"))
	end

	if length(x) != coord_dim
		throw(
			ArgumentError(
				"Position vector length ($(length(x))) does not match active coordinate system dimension ($coord_dim)."
			),
		)
	end

	# Add validation for boundary region
	if !bc.boundary_region(x)
		throw(ArgumentError("Position x is not in the specified boundary region"))
	end

	if bc isa KSDirichletBC
		return bc.boundary_value(x)
	elseif bc isa KSNeumannBC
		normal = compute_normal_vector_cartesian(x, length(x))  # Updated to pass dimension
		flux_vector = bc.flux_value(x)
		if length(flux_vector) != coord_dim
			throw(
				ArgumentError(
					"Flux vector length ($(length(flux_vector))) does not match active coordinate system dimension ($coord_dim)."
				),
			)
		end
		return dot(flux_vector, normal)
	elseif bc isa KSRobinBC
		normal_tuple = compute_normal_vector(coord_sys, Tuple(x))
		normal = collect(normal_tuple)
		alpha = bc.dirichlet_coefficient(x)
		beta = bc.neumann_coefficient(x)
		g = bc.boundary_value(x)

		# Validate coefficients for zero case and mixed types
		if (isa(alpha, Real) && iszero(alpha)) && (isa(beta, Real) && iszero(beta))
			throw(ArgumentError("Both Robin coefficients cannot be zero"))
		end

		# Check if point is actually on boundary
		is_on_boundary = any(coord -> abs(coord) < 1e-8 || abs(coord - 1.0) < 1e-8, x)
		if !is_on_boundary
			throw(ArgumentError("Position x is not on the boundary"))
		end

		if isa(alpha, Real) && isa(beta, Real) && isa(g, Real)
			return (u) -> alpha * u + beta * dot(normal, ones(coord_dim)) - g
		elseif isa(alpha, AbstractVector) && isa(beta, AbstractVector) &&
			isa(g, AbstractVector)
			if length(alpha) != length(beta) || length(beta) != length(g)
				throw(
					ArgumentError(
						"Inconsistent vector lengths in Robin boundary condition coefficients"
					),
				)
			end
			return (u) -> alpha .* u .+ (beta .* normal) .- g
		else
			throw(
				ArgumentError(
					"Inconsistent types for Robin boundary condition coefficients"
				),
			)
		end
	else
		throw(ArgumentError("Unsupported boundary condition type: $(typeof(bc))"))
	end
end

@doc raw"""
    compute_normal_vector_cartesian(x::Vector{Float64})

**Deprecated:** Use `compute_normal_vector_cartesian(x, length(x))` instead.

This function is maintained for backward compatibility but will be removed in a future version.
"""
function compute_normal_vector_cartesian(x::Vector{Float64})
    Base.depwarn("compute_normal_vector_cartesian(x::Vector{Float64}) is deprecated, use compute_normal_vector_cartesian(x, length(x)) instead", :compute_normal_vector_cartesian)
    return compute_normal_vector_cartesian(x, length(x))
end

@doc raw"""
    compute_normal_vector_cartesian(x::Vector{T}, N::Int) where T <: Number

Computes the normal vector at a given point in an N-dimensional Cartesian coordinate system.

# Arguments
- `x::Vector{T}`: The point at which to compute the normal vector.
- `N::Int`: The dimensionality of the space.

# Returns
- A vector representing the outward-pointing normal vector.

# Mathematical Description
For a point on the boundary in N dimensions:
- For each dimension i from 1 to N:
  - If \( |x[i]| < \text{tol} \), the normal vector is -eᵢ (negative boundary)
  - If \( |x[i] - 1| < \text{tol} \), the normal vector is eᵢ (positive boundary)
where eᵢ is the i-th standard basis vector.
"""
function compute_normal_vector_cartesian(x::Vector{T}, N::Int) where T <: Number
    N > 0 || throw(ArgumentError("Dimension N must be positive"))
    length(x) == N || throw(DimensionMismatch("Vector length must match dimension N"))

    tol = 1e-8
    for i in 1:N
        if abs(x[i] - 0.0) < tol
            return [-one(T) * (j == i ? 1 : 0) for j in 1:N]  # Negative boundary
        elseif abs(x[i] - 1.0) < tol
            return [one(T) * (j == i ? 1 : 0) for j in 1:N]   # Positive boundary
        end
    end
    throw(ArgumentError("Position x is not on the boundary"))
end

@doc raw"""
	apply_boundary_condition(bc::AbstractKSBoundaryCondition, u::Union{T, AbstractVector{T}}, x::AbstractVector{T}) where T <: Number

Applies the boundary condition to a given position vector and solution vector.

# Arguments
- `bc::AbstractKSBoundaryCondition`: The boundary condition to apply.
- `u::Union{T, AbstractVector{T}}`: The solution vector or scalar.
- `x::AbstractVector{T}`: The position vector.

# Returns
- The result of applying the boundary condition.

# Mathematical Description
For Dirichlet and Neumann boundary conditions: \[ \text{result} =
\text{apply\_boundary\_condition}(bc, x) \]

For Robin boundary conditions: \[ \text{result} = \text{apply\_boundary\_condition}(bc, x)(u)
\]
"""
function apply_boundary_condition(
	u::Union{T, AbstractVector{T}}, x::AbstractVector{T}) where {T <: Number}
	bc::AbstractKSBoundaryCondition,
	result = apply_boundary_condition(bc, x)
	if bc isa KSRobinBC
		return result(u)
	else
		return result
	end
end

@doc raw"""
	get_boundary_nodes(mesh::KSMesh)

Retrieves the global indices of boundary nodes in a mesh.

# Arguments
- `mesh::KSMesh`: The mesh object.

# Returns
- A sorted vector of unique global indices of boundary nodes.

# Mathematical Description
For each cell in the mesh:
- For each neighbor in the cell:
  - If `neighbor_id == -1` (indicating a boundary):
	- Retrieve local boundary nodes using `get_local_boundary_nodes(cell, direction)`.
	- Map local indices to global indices and collect them.

\[ \text{boundary\_nodes} = \text{sort}(\text{unique}(\text{collected\_global\_indices})) \]
"""
function get_boundary_nodes(mesh::KSMesh)
	boundary_nodes = Int[]
	for cell in mesh.cells
		for (direction, neighbor_id) in cell.neighbors
			if neighbor_id == -1  # This indicates a boundary
				local_boundary_nodes = get_local_boundary_nodes(cell, direction)
				for local_idx in local_boundary_nodes
					global_idx = cell.node_map[local_idx]
					push!(boundary_nodes, global_idx)
				end
			end
		end
	end
	return sort(unique(boundary_nodes))
end

@doc raw"""
	get_local_boundary_nodes(cell::KSCell{T, N}, direction::Symbol) where {T, N}

Retrieves the local indices of boundary nodes for a given cell and direction.

# Arguments
- `cell::KSCell{T, N}`: The cell object.
- `direction::Symbol`: The direction symbol (e.g., `:dim1_pos`, `:dim2_neg`).

# Returns
- A sorted vector of local indices of boundary nodes.

# Mathematical Description
For each local index in the cell:
- If `direction` is positive and the index matches the positive boundary:
  - Collect the index.
- If `direction` is negative and the index matches the negative boundary:
  - Collect the index.

\[ \text{local\_boundary\_nodes} = \text{sort}(\text{collected\_local\_indices}) \]
"""
function get_local_boundary_nodes(cell::KSCell{T, N}, direction::Symbol) where {T, N}
	local_boundary_nodes = NTuple{N, Int}[]
	dim, side = parse_direction(direction)

	for idx in keys(cell.node_map)
		if (side == :neg && idx[dim] == 1) || (side == :pos && idx[dim] == cell.p[dim] + 2)
			push!(local_boundary_nodes, idx)
		end
	end

	return sort(local_boundary_nodes)
end

@doc raw"""
	parse_direction(direction::Symbol)

Parses a direction symbol into its dimension and side components.

# Arguments
- `direction::Symbol`: The direction symbol (e.g., `:dim1_pos`, `:dim2_neg`).

# Returns
- A tuple containing the dimension (as an integer) and the side (as a symbol, either `:pos`
  or `:neg`).

# Mathematical Description
Uses regular expression matching to extract the dimension and side from the direction symbol.

\[ \text{dim} = \text{parse}(Int, \text{captures}[1]) \] \[ \text{side} = (\text{captures}[2]
== "pos") ? :pos : :neg \]
"""
function parse_direction(direction::Symbol)
	str = string(direction)
	m = match(r"dim(\d+)_(pos|neg)", str)
	if m === nothing
		throw(
			ArgumentError(
				"Invalid direction: $direction. Expected format: dim<number>_<pos|neg>"
			),
		)
	end
	dim = parse(Int, m.captures[1])
	side = m.captures[2] == "pos" ? :pos : :neg
	return (dim, side)
end

# Update verify_boundary_condition to use mesh information properly
function verify_boundary_condition(
	bc::AbstractKSBoundaryCondition,
	solution::AbstractVector,
	mesh::KSMesh{T, N},
) where {T, N}
	try
		# Get boundary nodes from mesh
		boundary_nodes = get_boundary_nodes(mesh)
		isempty(boundary_nodes) && return true

		# Get coordinates for each boundary node
		for node in boundary_nodes
			coords = get_node_coordinates(node, mesh)

			# Skip if point not in boundary region
			!bc.boundary_region(coords) && continue

			 # Verify boundary position using N-dimensional check
            is_on_boundary = any(coord -> abs(coord) < 1e-8 || abs(coord - 1.0) < 1e-8, coords)
            if !is_on_boundary
                continue
            end

			# Get expected boundary value
			expected = if bc.boundary_value isa Function
				bc.boundary_value(coords)
			else
				bc.boundary_value
			end

			# Compare solution value with expected
			if !isapprox(solution[node], expected; rtol = sqrt(eps(T)))
				return false
			end
		end

		return true
	catch e
		@warn "Boundary condition verification failed" exception = e
		return false
	end
end

"""Get physical coordinates for a node in the mesh"""
function get_node_coordinates(cell::KSCell{T,N}, std_cell::StandardKSCell, local_idx::NTuple{N,Int}) where {T,N}
    return ntuple(i -> std_cell.nodes_with_boundary[i][local_idx[i]], N)
end

"""Verify boundary condition satisfaction"""
function verify_boundary_condition(bc::AbstractKSBoundaryCondition, solution::AbstractVector, mesh::KSMesh)
    # Get boundary nodes that this BC applies to
    boundary_nodes = get_boundary_nodes(mesh)

    # Check each relevant boundary node
    for node in boundary_nodes
        if bc.boundary_region(node)
            if bc isa KSDirichletBC
                expected = bc.boundary_value(node)
                # Allow some numerical tolerance
                if !isapprox(solution[node], expected, rtol=sqrt(eps()))
                    return false
                end
            end
        end
    end
    return true
end

end # module BoundaryConditions
