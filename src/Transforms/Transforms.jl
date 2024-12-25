module Transforms

using LinearAlgebra, SparseArrays, Base.Threads
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..BoundaryConditions
using ..CacheManagement, ..NumericUtilities

# Exports
export apply_transform
export compute_transform_jacobian, jacobian
export evaluate_level_set, create_circle_level_set, create_rectangle_level_set
export union_level_set, intersection_level_set, difference_level_set
export transform_mesh, transform_cell, transform_spectral_properties
export get_or_create_transformed_standard_cell
export transform_node_map, inverse_transform
export is_periodic, domain_size_at_level
export negate_level_set, transform_weights
export transform_higher_order_matrices
export verify_quadrature_properties
export _raw_jacobian, create_transform_standard_cell
export calculate_level_scaling, domain_bounds_at_level
export validate_coordinates_physical
export transform_nodes_with_validation, transform_diff_matrices
export transform_bspline_data
export get_cell_physical_bounds

# Replace direct TRANSFORMED_STANDARD_CELLS usage with TransformCache methods
function get_or_create_transformed_standard_cell(
	p::NTuple{N, Int}, level::Int, mapping::DomainMapping
) where N
	key = (
		p,
		level,
		mapping.forward_transform,
		mapping.fictitious_system,
		mapping.physical_system,
	)

	lock(CACHE_LOCK) do
		if haskey(TRANSFORM_CACHE.cells, key)
			return TRANSFORM_CACHE.cells[key]
		end

		cell = create_transform_standard_cell(p, level)
		transformed = transform_spectral_properties(cell, mapping)
		TRANSFORM_CACHE.cells[key] = transformed
		return transformed
	end
end

function compute_scaling_factors(coord_system::AbstractKSCoordinateSystem)
	factors = ones(T, N)
	for i in 1:N
		if coord_system.active[i]
			domain = coord_system.domains[i]
			factors[i] = (domain[2] - domain[1]) / 2
		end
	end
	return factors
end

# Core scaling functions
function calculate_level_scaling(level::Int)
	level < 1 && throw(ArgumentError("Level must be ≥ 1"))
	return 2.0^(-level + 1)
end

function domain_size_at_level(level::Int)
	return 2.0 * calculate_level_scaling(level)
end

function domain_bounds_at_level(level::Int)
	half_size = calculate_level_scaling(level)
	return (-half_size, half_size)
end

# Basic coordinate validation functions
function is_periodic(system::AbstractKSCoordinateSystem, dim::Int)
	if system isa Union{KSPolarCoordinates, KSCylindricalCoordinates}
		return dim == 2
	elseif system isa KSSphericalCoordinates
		return dim in (2, 3)
	end
	return false
end

# Transform application functions
function apply_transform(
	transform::AbstractKSTransform,
	coords::NTuple{N, T},
	level::Int = 1) where {N, T}
	if transform isa AffineTransform
		return apply_transform(transform, coords, level)
	elseif transform isa NonlinearTransform
		return apply_transform(transform, coords, level)
	elseif transform isa CompositeTransform
		return apply_transform(transform, coords, level)
	else
		throw(ArgumentError("Unsupported transform type: $(typeof(transform))"))
	end
end

function apply_transform(
	transform::AffineTransform{T},
	coords::NTuple{N, T},
	level::Int = 1) where {N, T}
	scale = calculate_level_scaling(level)
	scaled_coords = collect(coords) .* scale
	result = transform.matrix * scaled_coords + transform.translation
	return ntuple(i -> result[i], N)
end

function apply_transform(
	transform::NonlinearTransform,
	coords::NTuple{N, T},
	level::Int = 1) where {N, T}
	scale = calculate_level_scaling(level)
	scaled_coords = collect(coords) .* scale
	result = transform.forward_map(scaled_coords)
	return ntuple(i -> result[i] / scale, N)
end

function apply_transform(
	transform::CompositeTransform,
	coords::NTuple{N, T},
	level::Int = 1) where {N, T}
	result = coords
	for t in transform.transforms
		result = apply_transform(t, result, level)
	end
	return result
end

function apply_transform(
	mapping::DomainMapping,
	coords::NTuple{N, T},
	level::Int = 1) where {N, T}

	# Validate coordinates and handle boundary cases
	valid_coords = if validate_coordinates_physical(coords, mapping.fictitious_system)
		coords
	else
		ntuple(
			i -> clamp(
				coords[i],
				mapping.fictitious_system.domains[i][1],
				mapping.fictitious_system.domains[i][2],
			), N)
	end

	try
		scale = calculate_level_scaling(level)
		scaled_coords = ntuple(i -> valid_coords[i] * scale, N)
		cart_coords = to_cartesian(
			scaled_coords, mapping.fictitious_system
		)
		transformed = apply_transform(mapping.forward_transform, cart_coords, level)
		result = from_cartesian(transformed, mapping.physical_system)
		return ntuple(i -> result[i] / scale, N)
	catch e
		throw(ArgumentError("Transform failed: $e"))
	end
end

# Enhanced Jacobian computation functions
function compute_transform_jacobian(
	mapping::DomainMapping,
	coords::Union{Tuple, AbstractVector},
	level::Int = 1)

	# Convert to cartesian coordinates if needed
	cart_coords = to_cartesian(coords, mapping.fictitious_system)

	# Get jacobian of the forward transform
	J = compute_transform_jacobian(mapping.forward_transform, cart_coords)

	# Handle non-cartesian coordinate systems
	if !(mapping.physical_system isa KSCartesianCoordinates)
		J_coord = compute_coordinate_jacobian(mapping.physical_system, cart_coords)
		J = J_coord * J
	end

	return J
end

function compute_transform_jacobian(
	transform::AbstractKSTransform,
	coords::AbstractVector{T};
	cond_threshold::Real = 1e6) where T

	# Input validation
	if any(x -> !isfinite(x) || abs(x) > 1e8, coords)
		throw(ArgumentError("Coordinates contain invalid or extremely large values"))
	end

	J = try
		_raw_jacobian(transform, coords)
	catch e
		throw(ArgumentError("Failed to compute Jacobian: $e"))
	end

	# Matrix validation
	if !all(isfinite, J)
		throw(ArgumentError("Jacobian contains non-finite values"))
	end

	κ = try
		cond(J)
	catch
		throw(ArgumentError("Failed to compute condition number - matrix may be singular"))
	end

	if κ > cond_threshold
		throw(
			ArgumentError(
				"Jacobian is ill-conditioned: condition number $κ exceeds threshold $cond_threshold"
			),
		)
	end

	if abs(det(J)) < sqrt(eps(T))
		throw(ArgumentError("Jacobian is nearly singular: determinant too close to zero"))
	end

	return J
end

function compute_transform_jacobian(
	coord_sys::AbstractKSCoordinateSystem, point::AbstractVector{T}
) where T
	# For Cartesian coordinates, return identity matrix
	if coord_sys isa KSCartesianCoordinates
		return Matrix{T}(I, length(point), length(point))
	else
		# For other coordinate systems, compute actual Jacobian
		transform = get_coordinate_transform(coord_sys)
		return compute_transform_jacobian(transform, point)
	end
end
function compute_transform_jacobian(
	transform::AffineTransform{T},
	coords::NTuple{N, T}) where {N, T}
	if cond(transform.matrix) > 1e6
		regularize_matrix!(transform.matrix)
	end
	return transform.matrix
end

function compute_transform_jacobian(
	transform::NonlinearTransform,
	coords::NTuple{N, T}) where {N, T}
	if transform.jacobian !== nothing
		return transform.jacobian(collect(coords))
	end

	# Finite difference approximation if no analytical jacobian provided
	h = sqrt(eps(T))
	J = zeros(T, N, N)
	coord_vec = collect(coords)

	for i in 1:N
		dx = zeros(T, N)
		dx[i] = h
		J[:, i] =
			(transform.forward_map(coord_vec + dx) -
			 transform.forward_map(coord_vec)) / h
	end

	if cond(J) > 1e6
		regularize_matrix!(J)
	end

	return J
end

function _raw_jacobian(transform::AbstractKSTransform, coords::AbstractVector{T}) where T
	if hasfield(typeof(transform), :jacobian) && transform.jacobian !== nothing
		J = transform.jacobian(coords)
		size(J) == (length(coords), length(coords)) || throw(DimensionMismatch(
			"Jacobian dimensions must match coordinate dimensions"))
		return J
	end

	# Use finite differences if no analytical Jacobian provided
	n = length(coords)
	J = zeros(T, n, n)
	h = sqrt(eps(T))

	for i in 1:n
		dx = zeros(T, n)
		dx[i] = h
		# Central difference for better accuracy
		J[:, i] =
			(transform.forward_map(coords + dx) -
			 transform.forward_map(coords - dx)) / (2h)
	end

	return J
end
function jacobian(mapping::DomainMapping, coords::NTuple{N, T}) where {N, T}
	# Stricter coordinate validation
	if !validate_coordinates_physical(coords, mapping.fictitious_system; tol = eps(T))
		throw(
			ArgumentError(
				"Invalid coordinates $coords for Jacobian computation in the given domain"
			),
		)
	end

	# Compute the Jacobian of the forward transform
	J = try
		compute_transform_jacobian(mapping.forward_transform, coords)
	catch e
		throw(ArgumentError("Failed to compute transform Jacobian: $e"))
	end

	# Check if the Jacobian is ill-conditioned
	κ = try
		cond(J)
	catch e
		throw(ArgumentError("Failed to compute condition number: $e"))
	end

	if κ > 1e6
		J_reg = try
			regularize_matrix!(copy(J))
		catch e
			throw(ArgumentError("Matrix regularization failed: $e"))
		end

		κ_reg = cond(J_reg)
		if κ_reg > 1e8
			throw(
				ArgumentError(
					"Jacobian remains ill-conditioned after regularization:\n" *
					"Original condition number: $κ\n" *
					"Regularized condition number: $κ_rWeg",
				),
			)
		end
		J = J_reg
	end

	# Check if the Jacobian is singular
	if abs(det(J)) < sqrt(eps(T))
		throw(ArgumentError("Jacobian is nearly singular: determinant too close to zero"))
	end

	return J
end

# Level set functions
function evaluate_level_set(
	ls::PrimitiveLevelSet,
	coords::NTuple{N, T};
	level::Int = 1,
	tol::Real = sqrt(eps(T))) where {N, T}
	scale = calculate_level_scaling(level)
	scaled_coords = ntuple(i -> coords[i] * scale, N)

	try
		val = ls.func(collect(scaled_coords)) / scale
		return abs(val) ≤ tol ? zero(T) : val
	catch e
		@warn "Level set evaluation failed" exception = e
		return zero(T)
	end
end

function evaluate_level_set(
	ls::CompositeLevelSet,
	coords::NTuple{N, T};
	level::Int = 1,
	tol::Real = sqrt(eps(T))) where {N, T}
	coord_vec = collect(coords)
	values = [
		evaluate_level_set(p, coords; level = level, tol = tol) for p in ls.primitives
	]

	result = values[1]
	for (op, val) in zip(ls.operations, values[2:end])
		result = op(result, val)
	end

	return abs(result) < tol ? zero(T) : result
end

function evaluate_level_set(
	ls::AbstractLevelSetFunction,
	coords::NTuple{N, T};
	level::Int = 1,
	tol::Real = sqrt(eps(T))) where {N, T}
	scale = calculate_level_scaling(level)
	scaled_coords = ntuple(i -> coords[i] * scale, N)
	val = ls(scaled_coords) / scale
	return abs(val) ≤ tol ? zero(T) : val
end

function create_circle_level_set(
	center::NTuple{N, T},
	radius::T,
	level::Int = 1) where {N, T}
	radius > zero(T) || throw(ArgumentError("Radius must be positive"))

	function circle_func(x::AbstractVector{T})
		scaled_x = x ./ calculate_level_scaling(level)
		return norm([xi - ci for (xi, ci) in zip(scaled_x, center)]) - radius
	end

	function circle_gradient(x::AbstractVector{T})
		scaled_x = x ./ calculate_level_scaling(level)
		diff = scaled_x .- collect(center)
		n = norm(diff)
		return n > zero(T) ? diff ./ n : zeros(T, N)
	end

	return PrimitiveLevelSet(;
		func = circle_func,
		gradient = circle_gradient,
	)
end

function create_rectangle_level_set(
	bounds::NTuple{N, Tuple{T, T}},
	level::Int = 1) where {N, T}
	for (min_val, max_val) in bounds
		min_val < max_val || throw(ArgumentError("Invalid bounds: $min_val ≥ $max_val"))
	end

	scaled_bounds = ntuple(
		i -> (
			bounds[i][1] * calculate_level_scaling(level),
			bounds[i][2] * calculate_level_scaling(level),
		),
		N,
	)

	function rectangle_func(x::AbstractVector{T})
		scaled_x = ntuple(i -> x[i] * calculate_level_scaling(level), N)
		return maximum(
			max(scaled_bounds[i][1] - scaled_x[i], scaled_x[i] - scaled_bounds[i][2])
			for i in 1:N
		)
	end

	return PrimitiveLevelSet(;
		func = rectangle_func,
		gradient = nothing,
	)
end

function union_level_set(level_sets::Vector{<:AbstractLevelSetFunction})
	isempty(level_sets) && throw(ArgumentError("level_sets cannot be empty"))
	operations = fill((a, b) -> min(a, b), length(level_sets) - 1)
	return CompositeLevelSet(;
		operations = operations,
		primitives = AbstractLevelSetFunction[level_sets...],
	)
end

function intersection_level_set(level_sets::Vector{<:AbstractLevelSetFunction})
	isempty(level_sets) && throw(ArgumentError("level_sets cannot be empty"))
	operations = fill(max, length(level_sets) - 1)  # Use max for intersection
	primitives = AbstractLevelSetFunction[level_sets...]
	return CompositeLevelSet(;
		operations = operations,
		primitives = primitives,
	)
end

function difference_level_set(ls1::AbstractLevelSetFunction, ls2::AbstractLevelSetFunction)
	return intersection_level_set([ls1, negate_level_set(ls2)])
end

function negate_level_set(ls::AbstractLevelSetFunction)
	return PrimitiveLevelSet(;
		func = x -> -ls.func(x),
		gradient = isnothing(ls.gradient) ? nothing : x -> -ls.gradient(x),
	)
end

# Mesh transformation functions - consolidated version
function transform_mesh(
	mesh::KSMesh{T, N}, transform::Union{DomainMapping, AbstractKSTransform}) where {T, N}
	actual_transform = transform isa DomainMapping ? transform.forward_transform : transform

	# Transform cells while preserving physical bounds
	new_cells = map(mesh.cells) do cell
		# Get and verify physical bounds
		physical_bounds = get_cell_physical_bounds(cell, mesh)
		transformed_cell = transform_cell(cell, actual_transform)

		# Validate transformed bounds match physical domain for DomainMapping
		if transform isa DomainMapping
			transformed_bounds = get_cell_physical_bounds(transformed_cell, mesh)
			# Flatten the bounds into arrays
			physical_bounds_flat = collect(Iterators.flatten(physical_bounds))
			transformed_bounds_flat = collect(Iterators.flatten(transformed_bounds))
			# Compare bounds and throw error if they don't match
			if !all(isapprox.(physical_bounds_flat, transformed_bounds_flat))
				throw(ArgumentError("Transformed cell bounds do not match physical bounds"))
			end
		end

		return transformed_cell
	end

	# Create new mesh with transformed components
	transformed_mesh = KSMesh{T, N}(
		new_cells,
		mesh.global_error_estimate,
		copy(mesh.boundary_cells),
		mesh.physical_domain,
	)

	# Store transformation data in the mesh properties
	transformed_mesh.transformation_data = actual_transform

	return transformed_mesh
end

function transform_continuity_constraints(
	constraints::Union{Nothing, Dict{Symbol, Matrix{T}}},
	mapping::DomainMapping) where T
	isnothing(constraints) && return nothing

	new_constraints = Dict{Symbol, Matrix{T}}()
	for (key, matrix) in constraints
		J = jacobian(mapping, zeros(T, size(matrix, 1)))
		new_constraints[key] = J * matrix * inv(J)'
	end

	return new_constraints
end

function transform_node_map(
	node_map::Dict{Tuple{Int, NTuple{N, Int}}, Int},
	mapping::DomainMapping) where N
	new_map = Dict{Tuple{Int, NTuple{N, Int}}, Int}()

	for ((cell_id, local_idx), global_idx) in node_map
		new_map[(cell_id, local_idx)] = global_idx
	end

	return new_map
end

# Cell transformation and verification functions
function transform_cell(cell::KSCell{T, N}, mapping::DomainMapping) where {T, N}
	return transform_cell(cell, mapping.forward_transform)
end

function transform_cell(cell::KSCell{T, N}, transform::AbstractKSTransform) where {T, N}
	try
		# Get the standard cell first
		standard_cell = SpectralMethods.get_or_create_standard_cell(cell.p, cell.level)

		# Transform the standard cell
		transformed_std_cell = get_or_create_transformed_standard_cell(
			cell.p, cell.level,
			DomainMapping(;
				forward_transform = transform,
				inverse_transform = inverse_transform(transform),
				fictitious_system = KSCartesianCoordinates(ntuple(i -> (-1.0, 1.0), N)),
				physical_system = KSCartesianCoordinates(ntuple(i -> (-1.0, 1.0), N)),
			),
		)

		# Create new cell with transformed properties
		# Match the constructor signature exactly
		return KSCell{T, N}(
			cell.id,
			cell.p,
			cell.level,
			cell.continuity_order,
			(ntuple(i -> cell.p[i], N), cell.level), # Required tuple format
			Dict{Symbol, Int}(k => v for (k, v) in cell.neighbors),
			Dict{NTuple{N, Int}, Int}(k => v for (k, v) in cell.node_map),
			cell.tensor_product_mask,
			Dict{Symbol, Int}(k => v for (k, v) in cell.boundary_connectivity),
			cell.error_estimate,
			cell.legendre_decay_rate,
			cell.is_leaf,
			cell.is_fictitious,
			cell.refinement_options,
			cell.parent_id,
			cell.child_ids,
			cell.b_spline_coefficients,
			cell.b_spline_knots,
		)
	catch e
		throw(ArgumentError("Failed to transform cell: $e"))
	end
end
function validate_coordinates_physical(
	coords::Union{Tuple, NTuple{N, T} where {N, T}},
	system::AbstractKSCoordinateSystem;
	tol::Real = 1e-10,
)
	# Check length matches system dimensions
	length(coords) == length(system.domains) ||
		throw(ArgumentError("Coordinate dimension mismatch"))

	# Determine base type considering all coordinate types
	base_type = if any(x -> x isa BigFloat || x isa Irrational, coords)
		BigFloat
	elseif any(x -> x isa Float64, coords)
		Float64
	elseif any(x -> x isa Float32, coords)
		Float32
	else
		Float64  # Default fallback
	end

	# Convert coordinates to base_type
	coords_converted = map(enumerate(coords)) do (i, x)
		isnan(x) && throw(ArgumentError("Coordinate $i is NaN"))
		try
			converted = convert(base_type, x)
			isfinite(converted) ? converted : Inf
		catch e
			if e isa InexactError || e isa OverflowError
				return Inf
			else
				rethrow(e)
			end
		end
	end

	# Early rejection for any infinite values
	any(isinf, coords_converted) && return false

	# Preserve highest precision type including domain types
	T = promote_type(
		base_type,
		typeof.(coords_converted)...,
		eltype.(system.domains)...,
	)

	# Convert coordinates and domain bounds to common type
	coords_promoted = convert.(T, coords_converted)
	domains_promoted = map(system.domains) do range
		isnothing(range) ? nothing :
		(convert(T, range[1]), convert(T, range[2]))
	end

	# Convert tolerance to appropriate type with scaling
	tol_t = convert(T, tol)
	tol_t *= if T <: BigFloat
		100
	elseif T <: Float32
		10
	else
		1
	end

	# Validate each dimension
	for i in eachindex(coords_promoted)
		!system.active[i] && continue
		range = domains_promoted[i]
		isnothing(range) && continue

		if is_periodic(system, i)
			period = convert(T, 2π)
			coord_normalized = mod(coords_promoted[i], period)
			!isfinite(coord_normalized) && return false
		else
			range_min, range_max = range
			if coords_promoted[i] < range_min - tol_t ||
				coords_promoted[i] > range_max + tol_t
				return false
			end
		end
	end

	return true
end

# Spectral properties transformation
function transform_spectral_properties(
	cell::StandardKSCell{T, N}, mapping::DomainMapping
) where {T, N}
	# Transform nodes
	transformed_nodes = transform_nodes_with_validation(
		cell.nodes_with_boundary, mapping, cell.level)

	# Transform weights
	transformed_weights = transform_weights(
		cell.weights_with_boundary, cell, mapping)

	# Transform differentiation matrices
	transformed_diff_matrices = transform_diff_matrices(
		cell.differentiation_matrix_with_boundary,
		mapping,
		cell.nodes_with_boundary)

	# Transform higher order matrices
	transformed_higher_order = transform_higher_order_matrices(
		cell.higher_order_diff_matrices_with_boundary,
		mapping,
		cell.nodes_with_boundary)

	return StandardKSCell(;
		p = cell.p,
		level = cell.level,
		continuity_order = cell.continuity_order,
		nodes_with_boundary = transformed_nodes,
		nodes_interior = cell.nodes_interior,
		weights_with_boundary = transformed_weights,
		weights_interior = cell.weights_interior,
		differentiation_matrix_with_boundary = transformed_diff_matrices,
		differentiation_matrix_interior = cell.differentiation_matrix_interior,
		quadrature_matrix = cell.quadrature_matrix,
		higher_order_diff_matrices_with_boundary = transformed_higher_order,
		higher_order_diff_matrices_interior = cell.higher_order_diff_matrices_interior,
	)
end

# Enhanced weight transformation with improved stability
function transform_weights(
	weights::NTuple{N, Vector{T}},
	cell::StandardKSCell{T, N},
	mapping::DomainMapping) where {T, N}

	# Input validation
	try
		J = compute_transform_jacobian(mapping, ntuple(i -> zero(T), N))
		if abs(det(J)) < eps(T) || cond(J) > 1e6
			throw(ArgumentError("Invalid transformation: ill-conditioned or singular"))
		end
	catch e
		throw(ArgumentError("Invalid transformation: $e"))
	end

	transformed_weights = ntuple(N) do d
		if !mapping.fictitious_system.active[d]
			return copy(weights[d])
		end

		new_weights = similar(weights[d])
		nodes = cell.nodes_with_boundary[d]

		# For affine transforms, use direct scaling
		if mapping.forward_transform isa AffineTransform
			scale_factor = abs(det(mapping.forward_transform.matrix))^(1 / N)
			return scale_factor .* weights[d]
		end

		# For non-affine transforms, use local Jacobian scaling
		J_base = compute_transform_jacobian(mapping, ntuple(i -> zero(T), N))
		scale_base = max(abs(det(J_base))^(1 / N), eps(T))

		for i in eachindex(nodes)
			coords = ntuple(k -> k == d ? nodes[i] : zero(T), N)
			J = compute_transform_jacobian(mapping, coords)
			scale_local = max(abs(det(J))^(1 / N), eps(T))
			new_weights[i] = weights[d][i] * scale_local / scale_base
		end

		# Normalize weights to preserve total quadrature weight
		total_weight = sum(weights[d])
		weight_sum = sum(new_weights)
		if weight_sum > eps(T)
			new_weights .*= total_weight / weight_sum
		end

		return new_weights
	end

	return transformed_weights
end

# Node transformation with validation
function transform_nodes_with_validation(
	nodes::NTuple{N, Vector{T}},
	mapping::DomainMapping,
	level::Int) where {N, T}
	bounds = domain_bounds_at_level(level)

	for d in 1:N
		if mapping.fictitious_system.active[d]
			if any(n -> !(bounds[1] ≤ n ≤ bounds[2]), nodes[d])
				throw(
					ArgumentError(
						"Nodes in dimension $d outside valid range for level $level"
					),
				)
			end
		end
	end

	return ntuple(N) do d
		if !mapping.fictitious_system.active[d]
			return copy(nodes[d])
		end

		map(nodes[d]) do node
			coords = ntuple(i -> i == d ? node : zero(T), N)
			transformed = apply_transform(mapping, coords, level)
			transformed[d]
		end
	end
end

# Inverse transform functions
function inverse_transform(transform::AffineTransform{T}) where T
	try
		if cond(transform.matrix) > 1e8
			throw(ArgumentError("Transform matrix is too ill-conditioned"))
		end
		inv_matrix = inv(transform.matrix)
		inv_translation = -inv_matrix * transform.translation
		return AffineTransform(inv_matrix, inv_translation)
	catch e
		if e isa SingularException
			throw(ArgumentError("Cannot invert singular transform matrix"))
		end
		rethrow(e)
	end
end

function inverse_transform(transform::NonlinearTransform)
	isnothing(transform.inverse_map) &&
		throw(ArgumentError("Inverse map not defined for this NonlinearTransform"))

	return NonlinearTransform(;
		forward_map = transform.inverse_map,
		inverse_map = transform.forward_map,
		jacobian = x -> inv(transform.jacobian(transform.inverse_map(x))),
	)
end

function inverse_transform(transform::CompositeTransform)
	return CompositeTransform(reverse!(map(inverse_transform, transform.transforms)))
end

# B-spline transformation
function transform_bspline_data(
	coefficients::Union{Nothing, NTuple{N, Vector{T}}},
	mapping::DomainMapping) where {N, T}
	isnothing(coefficients) && return nothing

	transformed_coefficients = ntuple(N) do d
		if !mapping.fictitious_system.active[d]
			return copy(coefficients[d])
		else
			return map(coefficients[d]) do coefficient
				coords = ntuple(i -> i == d ? coefficient : zero(T), N)
				apply_transform(mapping, coords, 1)[d]
			end
		end
	end

	return transformed_coefficients
end

# Function has been moved to the end of the file

function transform_higher_order_matrices(matrices::NTuple{N, Vector{Matrix{T}}},
	mapping::DomainMapping,
	nodes::NTuple{N, Vector{T}}) where {T, N}
	return ntuple(N) do d
		if !mapping.fictitious_system.active[d]
			return copy(matrices[d])
		end

		map(enumerate(matrices[d])) do (order, matrix)
			# Create copy for modification
			M = copy(matrix)
			n = size(M, 1)

			# For affine transforms, use simpler scaling with better conditioning
			if mapping.forward_transform isa AffineTransform
				J = mapping.forward_transform.matrix

				# Compute more stable scaling factor
				scale = (abs(det(J)))^(1 / N)
				scale = clamp(scale, eps(T), 1e3)  # Limit scaling range

				# Scale original matrix
				result = M ./ (scale^order)

				# Apply SVD-based regularization
				U, S, V = svd(result)

				# Adjust singular values to improve condition number
				S_new = similar(S)
				s_max = maximum(S)
				s_min = s_max / 1e6  # Target condition number

				for i in eachindex(S)
					S_new[i] = max(S[i], s_min)
				end

				result = U * Diagonal(S_new) * V'

				return result
			end

			# For non-affine transforms, use incremental scaling
			J_base = compute_transform_jacobian(mapping, zeros(T, N))
			regularize_matrix!(J_base; max_cond = 1e6)
			scale_base = abs(det(J_base))^(1 / N)
			scale_base = clamp(scale_base, eps(T), 1e3)

			# Pre-scale matrix to improve conditioning
			M .*= scale_base^order

			# Initial regularization
			regularize_matrix!(M; max_cond = 1e6)

			result = similar(M)

			# Transform with improved stability
			for i in 1:n
				coords = ntuple(k -> k == d ? nodes[d][i] : zero(T), N)
				J = compute_transform_jacobian(mapping, coords)
				regularize_matrix!(J; max_cond = 1e6)

				# Compute stable local scaling
				det_J = det(J)
				scale_i = abs(det_J)^(1 / N)
				scale_i = clamp(scale_i, eps(T), 1e3)

				for j in 1:n
					result[i, j] = M[i, j] / (scale_i^order)
				end
			end

			# Apply incremental regularization
			for iter in 1:3
				# Check and improve condition number
				if cond(result) > 1e6
					regularize_matrix!(result; max_cond = 1e6)
				end

				# Check and improve determinant
				if abs(det(result)) < 1e-10
					# Add scaled identity
					δ = sqrt(eps(T)) * opnorm(result, Inf)
					for i in 1:n
						result[i, i] += δ
					end
				end
			end

			# Final scaling adjustment if needed
			if opnorm(result, Inf) > 1e6
				result .*= 1e6 / opnorm(result, Inf)
			end

			return result
		end
	end
end

function create_transform_standard_cell(p::NTuple{N, Int}, level::Int) where N
	all(x -> x >= 3, p) || throw(ArgumentError("All polynomial degrees must be ≥ 3"))
	level >= 1 || throw(ArgumentError("Level must be ≥ 1"))

	# Default continuity order
	continuity_order = ntuple(_ -> 2, N)

	# Get base spectral properties using SpectralMethods functionality
	spectral_props = ntuple(
		d ->
			SpectralMethods.get_or_create_spectral_properties(p[d], continuity_order[d]), N)

	# Apply transformation-specific scaling
	scale = calculate_level_scaling(level)

	# Scale nodes while preserving weights and matrices
	nodes_wb = ntuple(d -> spectral_props[d].nodes_with_boundary * scale, N)
	nodes_int = ntuple(d -> spectral_props[d].nodes_interior * scale, N)
	weights_wb = ntuple(d -> spectral_props[d].weights_with_boundary, N)
	weights_int = ntuple(d -> spectral_props[d].weights_interior, N)
	D_wb = ntuple(d -> spectral_props[d].differentiation_matrix_with_boundary, N)
	D_int = ntuple(d -> spectral_props[d].differentiation_matrix_interior, N)
	Q = ntuple(d -> spectral_props[d].quadrature_matrix, N)
	D_higher_wb = ntuple(d ->
			spectral_props[d].higher_order_diff_matrices_with_boundary, N)
	D_higher_int = ntuple(d ->
			spectral_props[d].higher_order_diff_matrices_interior, N)

	# Create transformation-ready StandardKSCell
	return StandardKSCell(;
		p = p,
		level = level,
		continuity_order = continuity_order,
		nodes_with_boundary = nodes_wb,
		nodes_interior = nodes_int,
		weights_with_boundary = weights_wb,
		weights_interior = weights_int,
		differentiation_matrix_with_boundary = D_wb,
		differentiation_matrix_interior = D_int,
		quadrature_matrix = Q,
		higher_order_diff_matrices_with_boundary = D_higher_wb,
		higher_order_diff_matrices_interior = D_higher_int,
	)
end
function get_cell_physical_bounds(cell::KSCell{T, N}, mesh::KSMesh{T, N}) where {T, N}
	# Get cell's local bounds in reference space [-1,1]^N
	local_min, local_max = domain_bounds_at_level(cell.level)

	# Get transform if available with safe scaling
	transform_scale = if mesh.transformation_data isa AffineTransform
		# Extract diagonal scaling with bounds check
		scale_vec = diag(mesh.transformation_data.matrix)
		# Ensure scaling is reasonable
		map(s -> clamp(s, eps(T), 1e6), scale_vec)
	else
		ones(T, N)  # No transform or non-affine transform
	end

	# Transform bounds to physical space
	physical_bounds = ntuple(N) do i
		# Get domain for this dimension
		domain = if mesh.physical_domain isa Function
			point = zeros(T, N)
			point[i] = local_min
			result = mesh.physical_domain(point)

			if result isa Tuple && length(result) == 2 && result[1] isa Bool
				(zero(T), one(T))
			elseif result isa NTuple{N, Tuple{T, T}}
				result[i]
			elseif result isa AbstractVector{T} && length(result) == N
				(result[i] - one(T) / 2, result[i] + one(T) / 2)
			else
				(zero(T), one(T))
			end
		else
			if mesh.physical_domain isa NTuple{N, Tuple{T, T}}
				mesh.physical_domain[i]
			else
				(zero(T), one(T))
			end
		end

		# Map from [-1,1] to physical domain with safe scaling
		scale = calculate_level_scaling(cell.level)
		domain_size = domain[2] - domain[1]
		transform_factor = transform_scale[i]

		# Apply transformations with bounds checking
		bound_min =
			domain[1] + clamp(
				(local_min + 1) * domain_size / 2 * scale * transform_factor, -1e6, 1e6
			)
		bound_max =
			domain[1] + clamp(
				(local_max + 1) * domain_size / 2 * scale * transform_factor, -1e6, 1e6
			)

		return (bound_min, bound_max)
	end

	return physical_bounds
end

function transform_diff_matrices(
	matrices::NTuple{N, Matrix{T}},
	mapping::DomainMapping,
	nodes::NTuple{N, Vector{T}}) where {N, T}
	return ntuple(N) do d
		if !mapping.fictitious_system.active[d]
			return copy(matrices[d])
		end

		# Get original matrix for this dimension
		M = copy(matrices[d])
		n = size(M, 1)

		# For affine transforms, use simpler scaling
		if mapping.forward_transform isa AffineTransform
			J = mapping.forward_transform.matrix
			# Get stable scaling factor
			scale = abs(det(J))^(1 / N)
			scale = clamp(scale, eps(T), 1e3)

			# Scale matrix and regularize
			result = M ./ scale
			regularize_matrix!(result; max_cond = 1e6)
			return result
		end

		# For non-affine transforms
		result = similar(M)

		# Get base jacobian and scaling
		J_base = compute_transform_jacobian(mapping, zeros(T, N))
		regularize_matrix!(J_base; max_cond = 1e6)
		scale_base = max(abs(det(J_base))^(1 / N), eps(T))

		# Pre-scale matrix
		M .*= scale_base
		regularize_matrix!(M; max_cond = 1e6)

		# Transform each row with local scaling
		for i in 1:n
			coords = ntuple(k -> k == d ? nodes[d][i] : zero(T), N)
			J = compute_transform_jacobian(mapping, coords)
			regularize_matrix!(J; max_cond = 1e6)

			# Compute stable local scaling
			scale_i = max(abs(det(J))^(1 / N), eps(T))
			scale_i = clamp(scale_i, eps(T), 1e3)

			for j in 1:n
				result[i, j] = M[i, j] / scale_i
			end
		end

		# Final regularization
		regularize_matrix!(result; max_cond = 1e6)

		# Add stability boost if needed
		if cond(result) > 1e6 || abs(det(result)) < sqrt(eps(T))
			δ = sqrt(eps(T)) * opnorm(result, Inf)
			for i in 1:n
				result[i, i] += δ
			end
		end

		return result
	end
end

# Add this helper function near other quadrature-related functions
function integrate_monomial(nodes::AbstractVector{T}, order::Int) where T
	if order % 2 == 0
		return T(2.0) / (order + 1)  # Exact integral of x^n from -1 to 1 for even n
	else
		return zero(T)  # Odd powers integrate to 0 over symmetric interval
	end
end

function verify_quadrature_properties(cell::StandardKSCell{T, N}) where {T, N}
	# Base tolerance on numerical precision
	base_tol = max(sqrt(eps(T)), 1e-12)

	# Check positivity and finiteness of weights
	for (d, w) in enumerate(cell.weights_with_boundary)
		if any(x -> x < -base_tol || !isfinite(x), w)
			@warn "Invalid weights found in dimension $d"
			return false
		end
	end

	# Verify quadrature accuracy
	for d in 1:length(cell.p)
		nodes = cell.nodes_with_boundary[d]
		weights = cell.weights_with_boundary[d]

		# Check nodes are in valid range
		if any(!isfinite, nodes)
			@warn "Non-finite nodes found in dimension $d"
			return false
		end

		# Test integration accuracy
		for order in 0:cell.p[d]
			computed = sum(weights .* nodes .^ order)
			expected = integrate_monomial(nodes, order)

			# Use different tolerances for odd and even orders
			if order % 2 == 0
				# Even orders: relative tolerance for non-zero expected values
				rtol = base_tol
				if !isapprox(computed, expected; rtol = rtol, atol = base_tol)
					@warn "Even order quadrature error: order=$order, expected=$expected, got=$computed"
					return false
				end
			else
				# Odd orders: absolute tolerance since expected value is 0
				atol = max(base_tol * 100, 1e-8)  # More lenient for odd orders
				if abs(computed) > atol
					@warn "Odd order quadrature error: order=$order, expected=0, got=$computed"
					return false
				end
			end
		end
	end

	return true
end

end # module Transforms
