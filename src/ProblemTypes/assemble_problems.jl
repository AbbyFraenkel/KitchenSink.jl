"""
	create_system_matrix_and_vector(problem::AbstractKSProblem, mesh::KSMesh{T,N},
								  coord_sys::AbstractKSCoordinateSystem) where {T,N}

Create system matrices and vectors for problem solution.
"""
function create_system_matrix_and_vector(
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N},
	coord_sys::AbstractKSCoordinateSystem) where {T, N}

	# Initial validation
	if !validate_problem_setup(problem, mesh, coord_sys)
		throw(ArgumentError("Invalid problem setup"))
	end

	# Calculate system size
	mesh_dofs = get_mesh_dof(mesh)
	total_dofs = get_total_problem_dof(problem, mesh)

	if has_temporal_dof(problem)
		temporal_dofs = get_temporal_dof(problem)
		total_dofs *= temporal_dofs
	end

	# Initialize system with proper size
	A = spzeros(T, total_dofs, total_dofs)
	b = zeros(T, total_dofs)

	# Get cache key for this system
	cache_key = generate_cache_key(problem, mesh, total_dofs)

	return CacheManagement.get_or_create_cached_item(SOLVER_CACHE, cache_key) do
		# Assemble cell contributions
		for cell in mesh.cells
			if !cell.is_fictitious
				try
					assemble_cell_contributions!(A, b, problem, cell, mesh)
				catch e
					@warn "Cell assembly failed" cell = cell exception = e
					rethrow(e)
				end
			end
		end

		# Add coupling terms if needed
		if problem isa KSCoupledProblem
			add_coupling_terms!(A, b, problem, mesh)
		end

		finalize_system(A, b, problem, mesh, coord_sys)
	end
end

"""Generate cache key for system assembly."""
function generate_cache_key(problem::AbstractKSProblem, mesh::KSMesh, total_dofs::Int)
	key_components = Any[
		typeof(problem),
		total_dofs,
		problem.num_vars,
		get_mesh_dof(mesh),
		has_temporal_dof(problem) ? get_temporal_dof(problem) : nothing,
	]

	if problem isa KSCoupledProblem
		push!(key_components, [typeof(p) for p in problem.problems])
	end

	return (key_components...,)
end

"""
	assemble_cell_contributions!(A::AbstractMatrix{T}, b::AbstractVector{T},
							   problem::AbstractKSProblem, cell::KSCell{T,N},
							   mesh::KSMesh{T,N}) where {T,N}

Assemble contributions from a single cell into the global system.
"""
function assemble_cell_contributions!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	problem::AbstractKSProblem,
	cell::KSCell{T, N},
	mesh::KSMesh{T, N}) where {T, N}

	# Get cell data
	std_cell = SpectralMethods.get_or_create_standard_cell(
		cell.standard_cell_key[1],
		cell.standard_cell_key[2],
	)

	# Initialize local matrices
	local_A, local_b = initialize_local_matrices(problem, cell)

	# Get operators and check for null/empty
	diff_ops = get_cell_operators(std_cell)
	if isempty(diff_ops)
		@warn "No differential operators found for cell" cell_id=cell.id
		return false
	end

	# Add contributions and check return value
	success = try
		add_problem_contributions!(local_A, local_b, problem, cell, mesh, std_cell, diff_ops)
	catch e
		@warn "Cell assembly failed" cell=cell exception=e
		false
	end

	if success === true # Only add if explicitly successful
		# Map to global with bounds checking
		add_to_global_system!(A, b, local_A, local_b, cell, problem, mesh)
		return true
	end

	return false
end

"""Initialize local matrices for cell contributions."""
function initialize_local_matrices(
    problem::AbstractKSProblem,
    cell::KSCell{T,N}) where {T,N}

    # Calculate basic dimensions
    n_nodes = length(cell.node_map)
    n_vars = problem.num_vars
    local_size = n_nodes * n_vars

    # Just return matrices of correct size for cell
    return zeros(T, local_size, local_size), zeros(T, local_size)
end

"""
	add_coupling_terms!(A::AbstractMatrix{T}, b::AbstractVector{T},
					   problem::KSCoupledProblem, mesh::KSMesh{T,N}) where {T,N}

Add coupling terms between subproblems for coupled problems.
"""
function add_coupling_terms!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	problem::KSCoupledProblem,
	mesh::KSMesh{T, N}) where {T, N}
	n_probs = length(problem.problems)
	mesh_dofs = get_mesh_dof(mesh)

	# Calculate offsets for each subproblem
	sizes = [get_total_problem_dof(p, mesh) for p in problem.problems]
	offsets = [0; cumsum(sizes)[1:(end - 1)]]

	# Add coupling terms
	for i in 1:n_probs
		for j in 1:n_probs
			if i != j && !isnothing(problem.coupling_terms[i, j])
				# Get ranges for subproblems
				i_range = (offsets[i] + 1):(offsets[i] + sizes[i])
				j_range = (offsets[j] + 1):(offsets[j] + sizes[j])

				add_coupling_block!(
					view(A, i_range, j_range),
					view(b, i_range),
					view(A, i_range, :),
					view(A, j_range, :),
					problem.coupling_terms[i, j],
				)
			end
		end
	end
end

"""Add coupling block between subproblems."""
function add_coupling_block!(
	block::SubArray,
	b_block::SubArray,
	u_i::SubArray,
	u_j::SubArray,
	coupling_term::Function)
	try
		coupling_val = coupling_term(u_i, u_j)
		if length(b_block) == length(coupling_val)
			b_block .+= coupling_val
		end
	catch e
		@warn "Coupling term evaluation failed" exception = e
	end
end

"""
	add_to_global_system!(A::AbstractMatrix{T}, b::AbstractVector{T},
						 local_A::AbstractMatrix{T}, local_b::AbstractVector{T},
						 cell::KSCell{T,N}, problem::AbstractKSProblem, mesh::KSMesh{T,N}) where {T,N}

Add local cell contributions to global system.
"""
function add_to_global_system!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	cell::KSCell{T, N},
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}  # Added mesh parameter

	# Calculate temporal offset using passed mesh
	temporal_offset = if has_temporal_dof(problem)
		(get_temporal_dof(problem) - 1) * get_mesh_dof(mesh)
	else
		0
	end

	# Map indices
	for (local_idx, global_idx) in pairs(cell.node_map)
		# Convert multi-indices if needed
		local_i = local_idx isa Tuple ?
				  linear_index(local_idx, size(local_A)) : local_idx

		# Add contributions with temporal offset
		global_i = global_idx + temporal_offset
		if checkbounds(Bool, b, global_i) && checkbounds(Bool, local_b, local_i)
			b[global_i] += local_b[local_i]

			for (other_local_idx, other_global_idx) in pairs(cell.node_map)
				local_j = if other_local_idx isa Tuple
					linear_index(other_local_idx, size(local_A))
				else
					other_local_idx
				end
				global_j = other_global_idx + temporal_offset

				if checkbounds(Bool, A, global_i, global_j) &&
					checkbounds(Bool, local_A, local_i, local_j)
					A[global_i, global_j] += local_A[local_i, local_j]
				end
			end
		end
	end
end

"""
	finalize_system(A::AbstractMatrix{T}, b::AbstractVector{T},
				   problem::AbstractKSProblem, mesh::KSMesh{T,N},
				   coord_sys::AbstractKSCoordinateSystem) where {T,N}

Finalize system by applying boundary conditions and ensuring stability.
"""
function finalize_system(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N},
	coord_sys::AbstractKSCoordinateSystem) where {T, N}

	# Apply boundary conditions
	if hasfield(typeof(problem), :boundary_conditions) &&
		!isempty(problem.boundary_conditions)
		apply_boundary_conditions!(A, b, problem, mesh)
	end

	# Ensure diagonal dominance for stability
	ensure_diagonal_dominance!(A)

	return A, b
end

"""Ensure matrix diagonal dominance."""
function ensure_diagonal_dominance!(A::AbstractMatrix{T}) where T
	n = size(A, 1)
	for i in 1:n
		row_sum = sum(abs.(view(A, i, :))) - abs(A[i, i])
		if abs(A[i, i]) ≤ row_sum
			A[i, i] = T(2.0) * (row_sum + one(T))
		end
	end
end

"""Apply boundary conditions to system."""
function apply_boundary_conditions!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}
	boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)
	n = size(A, 1)

	# Reset boundary rows
	for node in boundary_nodes
		A[node, :] .= zero(T)
		A[node, node] = one(T)

		# Apply boundary condition
		for bc in problem.boundary_conditions
			if bc.boundary_region(node)
				b[node] = if bc.boundary_value isa Function
					convert(T, bc.boundary_value(node))
				else
					convert(T, bc.boundary_value)
				end
				break
			end
		end
	end
end
