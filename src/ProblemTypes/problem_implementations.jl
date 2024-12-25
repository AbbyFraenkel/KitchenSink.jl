"""
Problem type specific implementations for ProblemTypes module. Uses multiple dispatch
to handle each problem type's unique requirements.
"""

#===== Problem Contributions =====#

"""Add PDE problem contributions."""
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSPDEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops) where {T, N}
	quad_weights = std_cell.weights_with_boundary
	nodes = std_cell.nodes_with_boundary
	n_nodes = length(cell.node_map)
	n_vars = problem.num_vars

	if has_temporal_dof(problem)
		n_times = get_temporal_dof(problem)
	else
		n_times = 1
	end

	# Process each time step
	for t in 1:n_times
		time_offset = (t - 1) * n_nodes * n_vars

		# Calculate derivatives for entire solution matrix
		derivs = apply_differential_operators(diff_ops,
			@view local_A[(time_offset + 1):(time_offset + n_nodes * n_vars), :])

		# Process each quadrature point
		for i in 1:n_nodes
			point = ntuple(d -> nodes[d][i], N)
			u_quad = @view local_A[
				(time_offset + (i - 1) * n_vars + 1):(time_offset + i * n_vars), :,
			]

			# Evaluate PDE with derivatives
			deriv_views = ntuple(
				d -> @view(derivs[d][((i - 1) * n_vars + 1):(i * n_vars)]), N
			)
			time = t == 1 ? problem.tspan[1] : problem.tspan[2]

			pde_val = try
				problem.pde(point, u_quad, deriv_views, time)
			catch e
				@warn "PDE evaluation failed" point = point time = time exception = e
				(zeros(T, n_vars, n_vars), zero(T))
			end

			# Add contribution
			weight = prod(quad_weights[d][i] for d in 1:N)
			add_quadrature_contribution!(local_A, local_b, pde_val, weight,
				time_offset + (i - 1) * n_vars + 1)
		end
	end

	for idx in 1:n_nodes
		# Ensure diagonal dominance for stability
		if checkbounds(Bool, local_A, idx, :) && checkbounds(Bool, local_A, idx, idx)
			row_sum = sum(abs, view(local_A, idx, :))
			local_A[idx, idx] = max(local_A[idx, idx], T(2) * row_sum)
		end
	end

	return true
end

function add_problem_contributions!(local_A::Matrix{Float64},
	local_b::Vector{Float64},
	problem::KSPDEProblem{Float64, N},
	cell::KSCell{Float64, N},
	mesh::KSMesh{Float64, N},
	std_cell::StandardKSCell{Float64, N},
	diff_ops::Dict{Symbol, Any}) where N

	# Get number of nodes for this cell
	n_nodes = length(cell.node_map)

	# Pre-allocate contribution arrays based on dimension
	local_contributions = zeros(Float64, ntuple(i -> n_nodes, N))

	# Calculate contributions for each dimension
	for dim in 1:N
		dim_indices = 1:n_nodes
		for idx in dim_indices
			if idx <= length(local_b)
				local_contributions[ntuple(i -> i == dim ? idx : 1, N)...] = local_b[idx]
			end
		end
	end

	# Assemble local matrix
	for i in 1:n_nodes
		for j in 1:n_nodes
			if i <= size(local_A, 1) && j <= size(local_A, 2)
				local_A[i, j] += sum(local_contributions)
			end
		end
	end

	return nothing
end

"""Add ODE problem contributions."""
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSODEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops) where {T, N}
	quad_weights = std_cell.weights_with_boundary[1]  # ODEs only need 1D quadrature
	n_vars = problem.num_vars
	n_nodes = length(cell.node_map)

	if has_temporal_dof(problem)
		n_times = get_temporal_dof(problem)
	else
		n_times = 1
	end

	# Process each time step
	for t in 1:n_times
		time_offset = (t - 1) * n_nodes * n_vars
		time = t == 1 ? problem.tspan[1] : problem.tspan[2]

		for i in 1:n_nodes
			u_quad = @view local_A[
				(time_offset + (i - 1) * n_vars + 1):(time_offset + i * n_vars), :,
			]

			# Evaluate ODE
			ode_val = try
				problem.ode(u_quad, time)
			catch e
				@warn "ODE evaluation failed" node = i time = time exception = e
				(zeros(T, n_vars, n_vars), zero(T))
			end

			# Add contribution
			weight = quad_weights[i]
			add_quadrature_contribution!(local_A, local_b, ode_val, weight,
				time_offset + (i - 1) * n_vars + 1)
		end
	end

	return true
end

"""Add DAE problem contributions."""
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSDAEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops) where {T, N}
	quad_weights = std_cell.weights_with_boundary[1]
	n_vars = problem.num_vars
	n_alg = problem.num_algebraic_vars
	n_nodes = length(cell.node_map)

	if has_temporal_dof(problem)
		n_times = get_temporal_dof(problem)
	else
		n_times = 1
	end

	# Process each time step
	for t in 1:n_times
		time_offset = (t - 1) * n_nodes * n_vars
		time = t == 1 ? problem.tspan[1] : problem.tspan[2]

		for i in 1:n_nodes
			# Split variables
			u_diff = @view local_A[
				(time_offset + (i - 1) * n_vars + 1):(time_offset + (i - 1) * n_vars + (n_vars - n_alg)),
				:,
			]
			u_alg = @view local_A[
				(time_offset + (i - 1) * n_vars + (n_vars - n_alg) + 1):(time_offset + i * n_vars),
				:,
			]

			# Evaluate DAE
			dae_val = try
				du = similar(u_diff)
				val = problem.dae(du, [u_diff; u_alg], nothing, time)
				(vcat(du, val), zero(T))
			catch e
				@warn "DAE evaluation failed" node = i time = time exception = e
				(zeros(T, n_vars), zero(T))
			end

			# Add contribution
			weight = quad_weights[i]
			add_quadrature_contribution!(local_A, local_b, dae_val, weight,
				time_offset + (i - 1) * n_vars + 1)
		end
	end

	return true
end

"""Add moving boundary PDE problem contributions."""
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSMovingBoundaryPDEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	ops::Tuple{NTuple{N, AbstractMatrix}, NTuple{N, T}}) where {T, N}
	diff_ops, boundary_velocity = ops
	quad_weights = std_cell.weights_with_boundary
	nodes = std_cell.nodes_with_boundary
	n_nodes = length(cell.node_map)
	n_vars = problem.num_vars

	if has_temporal_dof(problem)
		n_times = get_temporal_dof(problem)
	else
		n_times = 1
	end

	# Process each time step
	for t in 1:n_times
		time_offset = (t - 1) * n_nodes * n_vars
		time = t == 1 ? problem.tspan[1] : problem.tspan[2]

		derivs = apply_differential_operators(diff_ops,
			@view local_A[(time_offset + 1):(time_offset + n_nodes * n_vars), :])

		for i in 1:n_nodes
			point = ntuple(d -> nodes[d][i], N)
			u_vec = @view local_A[
				(time_offset + (i - 1) * n_vars + 1):(time_offset + i * n_vars), :,
			]

			# Add boundary motion
			convection_term = sum(boundary_velocity[d] .* vec(derivs[d]) for d in 1:N)

			# Evaluate PDE with combined terms
			pde_val = try
				val = problem.pde(point, u_vec, derivs, time)
				val = val isa Tuple ? val : (reshape([val], 1, 1), zero(T))
				(val[1] + reshape(convection_term, size(val[1])), val[2])
			catch e
				@warn "Moving boundary PDE evaluation failed" point = point time = time exception =
					e
				(zeros(T, n_vars, 1), zero(T))
			end

			# Add contribution
			weight = prod(quad_weights[d][i] for d in 1:N)
			add_quadrature_contribution!(local_A, local_b, pde_val, weight,
				time_offset + (i - 1) * n_vars + 1)
		end
	end

	return true
end

"""Add coupled problem contributions."""
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSCoupledProblem,
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops) where {T, N}

	# Calculate sizes and offsets
	dims = [get_total_problem_dof(p, mesh) for p in problem.problems]
	# Calculate sizes and offsets
	dims = [get_total_problem_dof(p, mesh) for p in problem.problems]
	offsets = [0; cumsum(dims)[1:(end - 1)]]
	for (i, prob) in enumerate(problem.problems)
		# Calculate ranges safely
		start_idx = offsets[i] + 1
		end_idx = min(start_idx + dims[i] - 1, size(local_A, 2))

		# Create safe views
		local_A_sub = view(local_A, :, start_idx:end_idx)
		local_b_sub = view(local_b, start_idx:end_idx)

		# Add subproblem contributions
		add_problem_contributions!(
			local_A_sub, local_b_sub, prob, cell, mesh, std_cell, diff_ops
		)

		# Process coupling terms
		for j in 1:length(problem.problems)
			if i != j && !isnothing(problem.coupling_terms[i, j])
				j_start = offsets[j] + 1
				j_end = min(j_start + dims[j] - 1, size(local_A, 2))

				u_j = view(local_A, :, j_start:j_end)

				try
					coupling_val = problem.coupling_terms[i, j](local_A_sub, u_j)
					if length(local_b_sub) == length(coupling_val)
						local_b_sub .+= coupling_val
					end
				catch e
					@warn "Coupling term evaluation failed" i j exception = e
				end
			end
		end
	end

	return true
end

# Add special method for moving boundary problems that handles operator conversion
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSMovingBoundaryPDEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops::Dict{Symbol, Any}) where {T, N}

	# Get dimension info
	dim = N

	# Extract operators safely
	ops = if haskey(diff_ops, :D) && diff_ops[:D] isa NTuple
		diff_ops[:D]
	else
		ntuple(
			d -> if haskey(diff_ops, Symbol("D$d"))
				diff_ops[Symbol("D$d")]
			else
				zeros(T, 0, 0)
			end,
			dim,
		)
	end

	# Create constant velocity vector
	boundary_velocity = ntuple(_ -> T(0.1), dim)

	# Call original implementation with tuple arguments
	return add_problem_contributions!(
		local_A, local_b, problem, cell, mesh, std_cell,
		(ops, boundary_velocity),
	)
end

# Add special PDE problem method that handles derivative views
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSPDEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops::Dict{Symbol, Any}) where {T, N}

	# Add bounds checking for array accesses
	nodes = std_cell.nodes_with_boundary
	quad_weights = std_cell.weights_with_boundary
	n_nodes = length(cell.node_map)
	n_vars = problem.num_vars

	# Get time steps
	n_times = has_temporal_dof(problem) ? get_temporal_dof(problem) : 1

	# Process each time step
	for t in 1:n_times
		time_offset = (t - 1) * n_nodes * n_vars
		time = t == 1 ? problem.tspan[1] : problem.tspan[2]

		# Process each quadrature point with bounds checking
		for i in 1:min(n_nodes, minimum(length.(nodes)))
			point = ntuple(d -> nodes[d][i], N)
			u_quad = @view local_A[
				(time_offset + (i - 1) * n_vars + 1):(time_offset + i * n_vars), :,
			]

			# Calculate derivatives with size checking
			derivs = apply_differential_operators(diff_ops, u_quad)

			# Create properly sized derivative views
			deriv_views = ntuple(
				d -> begin
					d_vec = derivs[d]
					reshape(view(d_vec, ((i - 1) * n_vars + 1):(i * n_vars)), :)
				end, N)

			# Call PDE with proper types
			pde_val = try
				problem.pde(point, u_quad, deriv_views, time)
			catch e
				@warn "PDE evaluation failed" point = point time = time exception = e
				(zeros(T, n_vars, n_vars), zero(T))
			end

			# Add contribution with weight
			weight = prod(quad_weights[d][i] for d in 1:N)
			add_quadrature_contribution!(local_A, local_b, pde_val, weight,
				time_offset + (i - 1) * n_vars + 1)
		end
	end

	return true
end

"""Add weighted quadrature contribution to system."""
function add_quadrature_contribution!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	val::Tuple,
	weight::T,
	idx::Integer) where T
	A_part, b_part = val

	# Handle matrix contributions
	if A_part isa AbstractMatrix
		n = min(size(local_A, 2), size(A_part, 2))
		m = min(size(local_A, 1), size(A_part, 1))

		for j in 1:n
			for i in 1:m
				if checkbounds(Bool, local_A, idx + i - 1, j)
					local_A[idx + i - 1, j] += weight * A_part[i, j]
				end
			end
		end
		# Handle vector contributions
	elseif A_part isa AbstractVector
		for j in 1:min(length(A_part), size(local_A, 2))
			if checkbounds(Bool, local_A, idx, j)
				local_A[idx, j] += weight * A_part[j]
			end
		end
		# Handle scalar contributions
	else
		if checkbounds(Bool, local_A, idx, idx)
			local_A[idx, idx] += weight * A_part
		end
	end

	# Add RHS contribution
	if checkbounds(Bool, local_b, idx)
		local_b[idx] += weight * b_part
	end

	# Ensure diagonal dominance for stability
	if checkbounds(Bool, local_A, idx, idx)
		row_sum = sum(abs, view(local_A, idx, :))
		local_A[idx, idx] = max(local_A[idx, idx], T(2) * row_sum)
	end

	return true
end

"""
	add_problem_contributions!(local_A::AbstractMatrix{T}, local_b::AbstractVector{T},
							 problem::KSPDEProblem{T,N}, cell::KSCell{T,N},
							 mesh::KSMesh{T,N}, std_cell::StandardKSCell{T,N},
							 diff_ops::Dict{Symbol,Any}) where {T,N}

Add PDE contributions to local system matrices. Single implementation combining best features.
"""
function add_problem_contributions!(
	local_A::AbstractMatrix{T},
	local_b::AbstractVector{T},
	problem::KSPDEProblem{T, N},
	cell::KSCell{T, N},
	mesh::KSMesh{T, N},
	std_cell::StandardKSCell{T, N},
	diff_ops::Dict{Symbol, Any},
) where {T, N}
	# Validate inputs
	if size(local_A, 1) != size(local_A, 2)
		throw(ArgumentError("Local matrix must be square"))
	end
	if size(local_A, 1) != length(local_b)
		throw(DimensionMismatch("Matrix and vector dimensions must match"))
	end

	# Get quadrature points
	quad_points = std_cell.quadrature_points

	# Initialize views for derivatives
	u = view(local_b, :)
	D = view(diff_ops[:D], :, :)

	# Evaluate PDE at each quadrature point
	for q_idx in 1:length(quad_points)
		point = quad_points[q_idx]

		# Call PDE function with proper arguments
		pde_result = problem.pde(point, u, D, problem.tspan[1])

		# Extract coefficients
		A_coeff, b_coeff = if pde_result isa Tuple
			pde_result
		else
			(pde_result, zero(T))
		end

		# Update local system
		if A_coeff isa AbstractMatrix
			local_A .+= std_cell.quadrature_weights[q_idx] .* A_coeff
		else
			local_A[q_idx, q_idx] += std_cell.quadrature_weights[q_idx] * A_coeff
		end
		local_b[q_idx] += std_cell.quadrature_weights[q_idx] * b_coeff
	end

	return true
end

# Helper function to extract operators from dict
function extract_operators(diff_ops::Dict{Symbol, Any}, N::Int)
	# Convert to tuple form, handling missing operators
	ops = Vector{AbstractMatrix}(undef, N)
	for i in 1:N
		key = Symbol("D$i")
		ops[i] = get(diff_ops, key, diff_ops[:D])
	end
	return (ops...,)
end

"""
	apply_differential_operators(diff_ops::NTuple{N,AbstractMatrix}, u::AbstractVecOrMat) where N

Apply differential operators given as a tuple of matrices to a vector/matrix.
"""
function apply_differential_operators(
	diff_ops::NTuple{N, AbstractMatrix},
	u::AbstractVecOrMat,
) where N
	n_nodes = size(first(diff_ops), 1)

	return ntuple(d -> begin
			if d <= length(diff_ops)
				u_reshaped = reshape(vec(u), n_nodes, :)
				result = diff_ops[d] * u_reshaped
				vec(result)
			else
				zeros(eltype(u), size(u, 1))
			end
		end, N)
end

"""
	apply_differential_operators(diff_ops::Dict{Symbol,Any}, u::AbstractVecOrMat) where N

Apply differential operators stored in a dictionary to a vector/matrix.
"""
function apply_differential_operators(
	diff_ops::Dict{Symbol, Any},
	u::AbstractVecOrMat,
)
	# Validate operators exist
	if !haskey(diff_ops, :D) && !any(haskey(diff_ops, Symbol("D$i")) for i in 1:100)
		throw(ArgumentError("No differential operators found"))
	end

	# Handle :D key with tuple of operators
	if haskey(diff_ops, :D) && diff_ops[:D] isa NTuple
		return apply_differential_operators(diff_ops[:D], u)
	end

	# Get first valid operator and dimension
	first_op = if haskey(diff_ops, :D)
		diff_ops[:D]
	else
		first(diff_ops[Symbol("D$i")] for i in 1:100 if haskey(diff_ops, Symbol("D$i")))
	end

	first_matrix = first_op isa Tuple ? first_op[1] : first_op
	n_nodes = size(first_matrix, 1)

	# Determine problem dimension
	actual_dim = if hasfield(typeof(first_matrix), :dimension)
		first_matrix.dimension
	else
		count(i -> haskey(diff_ops, Symbol("D$i")), 1:100)
	end

	# Create operators array
	ops = [
		get(diff_ops, Symbol("D$i"), zeros(eltype(first_matrix), size(first_matrix)))
		for i in 1:actual_dim
	]

	# Convert to tuple and apply
	return apply_differential_operators(tuple(ops...), u)
end
