#  Adaptive

function estimate_error(problem::AbstractKSProblem, mesh::KSMesh{T, N},
	solution::Vector{T}) where {T <: Real, N}
	error_estimates = zeros(T, length(mesh.cells))

	for (i, cell) in enumerate(mesh.cells)
		error_estimates[i] = estimate_cell_error(problem, cell, mesh, solution)
	end

	return error_estimates
end

function estimate_cell_error(problem::AbstractKSProblem, cell::KSCell{T, N},
	mesh::KSMesh{T, N}, solution::Vector{T}) where {T <: Real, N}
	residual = compute_residual(problem, cell, solution, mesh)
	return norm(residual)
end

function compute_residual(problem::AbstractKSProblem, cell::KSCell{T, N},
	solution::Vector{T}, mesh::KSMesh{T, N}) where {T <: Real, N}
	standard_cell = get_or_create_standard_cell(
		cell.p, cell.level; continuity_order = cell.continuity_order)
	num_vars = problem.num_vars
	residual = zeros(T, prod(cell.p .+ 2) * num_vars)

	for quad_point in CartesianIndices(tuple(cell.p .+ 2...))
		phys_point = get_node_coordinates(
			problem.coordinate_system, cell, Tuple(quad_point), mesh.num_cells_per_dim)
		ref_point = map_to_reference_cell(collect(phys_point), problem.coordinate_system)

		local_solution = extract_local_solution(solution, cell, quad_point, num_vars)
		derivatives = compute_high_order_derivatives(
			standard_cell, get_problem_order(problem))
		pde_value = problem.equation(
			zero(T), phys_point, ref_point, local_solution, derivatives...)
		idx_range = ((quad_point.I[1] - 1) * num_vars + 1):(quad_point.I[1] * num_vars)
		residual[idx_range] = pde_value
	end

	return residual
end

function adapt_mesh!(problem::AbstractKSProblem, mesh::KSMesh{T, N},
	error_estimates::Vector{T}, tolerance::T) where {T <: Real, N}
	for (i, cell) in enumerate(mesh.cells)
		if error_estimates[i] > tolerance
			if should_h_refine(cell, error_estimates[i], tolerance)
				h_refine!(mesh, cell)
			else
				p_refine!(mesh, cell)
			end
		elseif error_estimates[i] < tolerance / 10  # Coarsening threshold
			if can_coarsen(mesh, cell)
				coarsen!(mesh, cell)
			end
		end
	end

	return update_mesh_connectivity!(mesh)
end

function should_h_refine(cell::KSCell{T, N}, error::T, tolerance::T) where {T <: Real, N}
	max_p = 10  # Maximum polynomial order
	decay_rate_threshold = 0.5  # Threshold for Legendre decay rate

	high_error = error > 10 * tolerance
	max_order_reached = all(cell.p .>= max_p)
	slow_decay = cell.legendre_decay_rate > decay_rate_threshold

	return high_error && (max_order_reached || slow_decay)
end

function h_refine!(mesh::KSMesh{T, N}, cell::KSCell{T, N}) where {T <: Real, N}
	new_cells = split_cell(cell)
	deleteat!(mesh.cells, findfirst(c -> c.id == cell.id, mesh.cells))
	append!(mesh.cells, new_cells)
	return update_node_maps!(mesh)
end

function p_refine!(mesh::KSMesh{T, N}, cell::KSCell{T, N}) where {T <: Real, N}
	cell.p = cell.p .+ 1
	return update_cell_spectral_properties!(cell)
end

function can_coarsen(mesh::KSMesh{T, N}, cell::KSCell{T, N}) where {T <: Real, N}
	return cell.level > 1 &&
		   all(sibling_ids(mesh, cell) .∈ Ref(getfield.(mesh.cells, :id)))
end

function coarsen!(mesh::KSMesh{T, N}, cell::KSCell{T, N}) where {T <: Real, N}
	siblings = filter(c -> c.id in sibling_ids(mesh, cell), mesh.cells)
	parent_id = cell.id ÷ 2^N

	new_cell = create_parent_cell(siblings)

	deleteat!(mesh.cells, findall(c -> c.id in getfield.(siblings, :id), mesh.cells))
	push!(mesh.cells, new_cell)
	return update_node_maps!(mesh)
end

function update_cell_neighbors!(cell::KSCell{T, N}, mesh::KSMesh{T, N}) where {T <: Real, N}
	cell.neighbors = Dict{Symbol, Int}()
	for d in 1:N
		for direction in (:neg, :pos)
			neighbor_id = find_neighbor(cell, d, direction, mesh)
			cell.neighbors[Symbol("dim$(d)_$direction")] = neighbor_id
		end
	end
end

function create_global_node_map(cells::Vector{KSCell{T, N}}) where {T <: Real, N}
	node_map = Dict{Tuple{Int, NTuple{N, Int}}, Int}()
	global_index = 1

	for cell in cells
		for (local_index, _) in cell.node_map
			if !haskey(node_map, (cell.id, local_index))
				node_map[(cell.id, local_index)] = global_index
				global_index += 1
			end
		end
	end

	return node_map
end

# function update_boundary_cells!(mesh::KSMesh{T, N}) where {T <: Real, N}
# 	mesh.boundary_cells = Dict{Symbol, Vector{Int}}()

# 	for (i, cell) in enumerate(mesh.cells)
# 		for (direction, neighbor_id) in cell.neighbors
# 			if neighbor_id == -1  # Boundary cell
# 				direction_sym = Symbol(direction)
# 				if !haskey(mesh.boundary_cells, direction_sym)
# 					mesh.boundary_cells[direction_sym] = Int[]
# 				end
# 				push!(mesh.boundary_cells[direction_sym], i)
# 			end
# 		end
# 	end
# end

function split_cell(cell::KSCell{T, N}) where {T <: Real, N}
	new_cells = Vector{KSCell{T, N}}()
	for i in 1:(2^N)
		new_cell = KSCell{T, N}(;
			id = cell.id * 2^N + i - 1,
			p = cell.p,
			level = cell.level + 1,
			continuity_order = cell.continuity_order,
			standard_cell_key = (cell.p, cell.level + 1),
			neighbors = Dict{Symbol, Int}(),
			node_map = Dict{NTuple{N, Int}, Int}(),
			tensor_product_mask = cell.tensor_product_mask,
			boundary_connectivity = Dict{Symbol, Int}(),
			error_estimate = zero(T),
			legendre_decay_rate = zero(T),
			is_leaf = true,
			is_fictitious = cell.is_fictitious,
			refinement_options = nothing,
			parent_id = cell.id,
			child_ids = nothing,
		)
		push!(new_cells, new_cell)
	end
	return new_cells
end

function update_node_maps!(mesh::KSMesh{T, N}) where {T <: Real, N}
	global_node_counter = 1
	for cell in mesh.cells
		update_cell_node_map!(cell, mesh, global_node_counter)
		global_node_counter += length(cell.node_map)
	end
end

function update_cell_node_map!(
	cell::KSCell{T, N}, mesh::KSMesh{T, N}, start_index::Int) where {T <: Real, N}
	standard_cell = get_or_create_standard_cell(
		cell.p, cell.level; continuity_order = cell.continuity_order)
	cell.node_map = Dict{NTuple{N, Int}, Int}()

	local_indices = CartesianIndices(ntuple(d -> 1:(cell.p[d] + 2), N))
	for (idx, local_index) in enumerate(local_indices)
		global_index = start_index + idx - 1
		cell.node_map[Tuple(local_index)] = global_index
	end
end

function update_cell_spectral_properties!(cell::KSCell{T, N}) where {T <: Real, N}
	standard_cell = get_or_create_standard_cell(
		cell.p, cell.level; continuity_order = cell.continuity_order)

	# Update differentiation matrices
	cell.differentiation_matrix_with_boundary =
		standard_cell.differentiation_matrix_with_boundary
	cell.differentiation_matrix_interior = standard_cell.differentiation_matrix_interior

	# Update quadrature weights
	cell.weights_with_boundary = standard_cell.weights_with_boundary
	cell.weights_interior = standard_cell.weights_interior

	# Update nodes
	cell.nodes_with_boundary = standard_cell.nodes_with_boundary
	cell.nodes_interior = standard_cell.nodes_interior

	# Update higher-order differentiation matrices if they exist
	if hasfield(typeof(standard_cell), :higher_order_diff_matrices_with_boundary)
		cell.higher_order_diff_matrices_with_boundary =
			standard_cell.higher_order_diff_matrices_with_boundary
		cell.higher_order_diff_matrices_interior =
			standard_cell.higher_order_diff_matrices_interior
	end

	# Update Legendre decay rate
	return cell.legendre_decay_rate = compute_legendre_decay_rate(cell, solution)
end

function sibling_ids(mesh::KSMesh{T, N}, cell::KSCell{T, N}) where {T <: Real, N}
	if cell.level == 0
		return Int[]
	end

	parent_id = cell.parent_id
	if parent_id === nothing
		return Int[]
	end

	sibling_start = parent_id * 2^N
	return [sibling_start + i for i in 0:(2^N - 1)]
end

function create_parent_cell(siblings::Vector{KSCell{T, N}}) where {T <: Real, N}
	if isempty(siblings)
		error("Cannot create parent cell from empty siblings list")
	end

	first_sibling = siblings[1]
	parent_id = first_sibling.id ÷ 2^N

	parent_cell = KSCell{T, N}(;
		id = parent_id,
		p = first_sibling.p,
		level = first_sibling.level - 1,
		continuity_order = first_sibling.continuity_order,
		standard_cell_key = (first_sibling.p, first_sibling.level - 1),
		neighbors = Dict{Symbol, Int}(),
		node_map = Dict{NTuple{N, Int}, Int}(),
		tensor_product_mask = first_sibling.tensor_product_mask,
		boundary_connectivity = Dict{Symbol, Int}(),
		error_estimate = maximum(sibling.error_estimate for sibling in siblings),
		legendre_decay_rate = mean(sibling.legendre_decay_rate for sibling in siblings),
		is_leaf = true,
		is_fictitious = all(sibling.is_fictitious for sibling in siblings),
		refinement_options = nothing,
		parent_id = nothing,
		child_ids = [sibling.id for sibling in siblings],
	)

	return parent_cell
end

function compute_legendre_decay_rate(
	cell::KSCell{T, N}, solution::Vector{T}) where {T <: Real, N}
	# Get the standard cell for this cell's polynomial order and level
	standard_cell = get_or_create_standard_cell(
		cell.p, cell.level; continuity_order = cell.continuity_order)

	# Extract the solution values for this cell
	local_solution = extract_local_solution(solution, cell)

	# Compute Legendre coefficients
	legendre_coeffs = compute_legendre_coefficients(local_solution, standard_cell)

	# Compute decay rate
	decay_rate = estimate_decay_rate(legendre_coeffs)

	return decay_rate
end

function compute_legendre_coefficients(
	local_solution::Vector{T}, standard_cell::StandardKSCell{T, N}) where {T <: Real, N}
	# Use the quadrature matrix to project the solution onto Legendre polynomials
	legendre_coeffs = standard_cell.quadrature_matrix * local_solution
	return legendre_coeffs
end

function estimate_decay_rate(coeffs::Vector{T}) where {T <: Real}
	# Compute the logarithm of the absolute values of coefficients
	log_coeffs = log.(abs.(coeffs) .+ eps(T))  # Add eps to avoid log(0)

	# Perform linear regression to estimate the decay rate
	n = length(coeffs)
	x = collect(1:n)
	slope =
		(n * sum(x .* log_coeffs) - sum(x) * sum(log_coeffs)) /
		(n * sum(x .^ 2) - sum(x)^2)

	# The decay rate is the negative of the slope
	return -slope
end

function update_mesh_connectivity!(mesh::KSMesh{T, N}) where {T <: Real, N}
	# Update neighbor information for all cells
	for cell in mesh.cells
		update_cell_neighbors!(cell, mesh)
	end

	# Update global node map
	mesh.node_map = create_global_node_map(mesh.cells)

	# Update boundary cells
	return update_boundary_cells!(mesh)
end

function adaptive_solve(problem::AbstractKSProblem, mesh::KSMesh{T, N},
	tolerance::T, max_iterations::Int) where {T <: Real, N}
	for iter in 1:max_iterations
		solution = solve_problem(problem, mesh)
		error_estimates = estimate_error(problem, mesh, solution)

		max_error = maximum(error_estimates)
		if max_error < tolerance
			return solution, mesh
		end

		adapt_mesh!(problem, mesh, error_estimates, tolerance)
	end

	return error("Maximum iterations reached without converging to specified tolerance")
end
