#===== Core DOF Functions =====#

"""Check if problem has valid boundary conditions"""
function has_boundary_condition(problem::AbstractKSProblem)::Bool
	return hasproperty(problem, :boundary_conditions) &&
		   !isnothing(problem.boundary_conditions) &&
		   !isempty(problem.boundary_conditions)
end

"""Base system DOF calculation"""
function get_system_dof(problem::AbstractKSProblem)::Int
	if !hasproperty(problem, :num_vars)
		throw(ArgumentError("Problem must have num_vars field"))
	end

	base_dofs = problem.num_vars

	# Add boundary condition DOFs if present
	if has_boundary_condition(problem)
		base_dofs += compute_boundary_dofs(problem.boundary_conditions)
	end

	return base_dofs
end

"""Compute DOFs contributed by boundary conditions"""
const BOUNDARY_DOF_CACHE = Dict{UInt, Int}()

function compute_boundary_dofs(
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
)::Int
	cache_key = hash(boundary_conditions)
	return get!(BOUNDARY_DOF_CACHE, cache_key) do
		count(bc -> bc isa KSRobinBC, boundary_conditions)
	end
end

# Mesh-aware version
function compute_boundary_dofs(
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}, mesh::KSMesh
)
	# Get boundary nodes from mesh
	boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)

	# Count additional DOFs needed for boundary conditions
	additional_dofs = 0
	for bc in boundary_conditions
		if bc isa KSRobinBC
			# Robin BCs need extra DOF for flux
			additional_dofs += count(n -> bc.boundary_region(n), boundary_nodes)
		end
	end
	return additional_dofs
end

"""Mesh DOF calculations"""
const MESH_DOF_CACHE = Dict{UInt, Int}()

function get_mesh_dof(mesh::KSMesh{T, N})::Int where {T, N}
	# Use mesh hash as cache key
	key = hash(mesh)

	return get!(MESH_DOF_CACHE, key) do
		node_count = Set{Int}()

		@inbounds for cell in mesh.cells
			if !cell.is_fictitious
				union!(node_count, values(cell.node_map))
			end
		end

		# Remove fictitious nodes
		for cell in mesh.cells
			if cell.is_fictitious
				setdiff!(node_count, values(cell.node_map))
			end
		end

		length(node_count)
	end
end

function get_physical_mesh_dof(mesh::KSMesh{T, N})::Int where {T, N}
	# Use SpectralMethods and CoordinateSystems for node handling
	physical_nodes = Set{Int}()

	for cell_idx in mesh.physical_cells
		cell = mesh.cells[cell_idx]
		try
			std_cell = SpectralMethods.get_or_create_standard_cell(
				cell.standard_cell_key[1],
				cell.standard_cell_key[2],
			)

			for (local_idx, global_idx) in cell.node_map
				# Validate node indices
				if all(
					i -> 1 <= local_idx[i] <= length(std_cell.nodes_with_boundary[i]), 1:N
				)
					coords = ntuple(i -> std_cell.nodes_with_boundary[i][local_idx[i]], N)

					# Use Transforms to validate physical coordinates
					if validate_coordinates_physical(coords, mesh.coordinate_system)
						if mesh.physical_domain(coords)
							push!(physical_nodes, global_idx)
						end
					end
				end
			end
		catch e
			@warn "Failed to get standard cell" cell_key = cell.standard_cell_key exception =
				e
			continue
		end
	end

	return length(physical_nodes)
end

function get_boundary_mesh_dof(mesh::KSMesh{T, N})::Int where {T, N}
	# Use BoundaryConditions for node identification
	boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)
	return length(boundary_nodes)
end

"""Get boundary mesh DOFs for moving boundary problems"""
function get_boundary_mesh_dof(
	mesh::KSMesh{T, N},
	problem::KSMovingBoundaryPDEProblem{T, N})::Int where {T, N}

	# Get base boundary DOFs
	base_dofs = get_boundary_mesh_dof(mesh)

	# Add extra DOFs for boundary motion (position and velocity)
	motion_dofs = 2N * base_dofs  # N for position, N for velocity

	return base_dofs + motion_dofs
end

#===== Problem-Specific Implementations =====#

function get_system_dof(problem::KSDAEProblem{T, N})::Int where {T, N}
	return problem.num_vars - problem.num_algebraic_vars
end

function get_system_dof(problem::KSPIDEProblem{T, N})::Int where {T, N}
	return problem.num_vars * compute_kernel_complexity(problem.kernel)
end

function get_system_dof(problem::KSIDEProblem{T, N})::Int where {T, N}
	num_quadrature_points = if hasproperty(problem, :num_quadrature_points)
		problem.num_quadrature_points
	else
		compute_kernel_complexity(problem.kernel)
	end
	dof = problem.num_vars * num_quadrature_points
	return Int(dof)
end

function get_system_dof(problem::KSMovingBoundaryPDEProblem{T, N})::Int where {T, N}
	position_velocity_dofs = 2N # N each for position and velocity
	return problem.num_vars + position_velocity_dofs
end

function get_system_dof(problem::KSOptimalControlProblem{T})::Int where T
	state_dofs = length(problem.initial_state)
	control_dofs = length(problem.control_bounds)
	adjoint_dofs = state_dofs
	constraint_dofs = if hasproperty(problem, :constraints)
		compute_constraint_dofs(problem.constraints)
	else
		0
	end
	return state_dofs + control_dofs + adjoint_dofs + constraint_dofs
end

function get_system_dof(problem::KSCoupledProblem)::Int
	n = length(problem.problems)
	# Preallocate for type stability
	base_dofs = zero(Int)
	coupling_dofs = zero(Int)
	dofs = Vector{Int}(undef, n)

	# Single pass accumulation
	@inbounds for i in 1:n
		dofs[i] = get_system_dof(problem.problems[i])
		base_dofs += dofs[i]

		# Process coupling terms in same loop
		for j in 1:i
			if !isnothing(problem.coupling_terms[i, j])
				coupling_dofs += compute_coupling_points(
					problem.problems[i],
					problem.problems[j],
					problem.coupling_terms[i, j],
				)
			end
		end
	end

	return base_dofs + coupling_dofs
end

function get_system_dof(problem::KSBVPProblem{T, N})::Int where {T, N}
    # Start with base DOFs from num_vars
    base_dofs = problem.num_vars

    # Count Robin BCs and add DOFs
    if has_boundary_condition(problem)
        robin_count = count(bc -> bc isa KSRobinBC, problem.boundary_conditions)
        base_dofs += robin_count * (N + 1)  # N flux variables plus 1 boundary value per Robin BC
    end

    return base_dofs
end

# Add this helper function to validate BC DOFs
function validate_boundary_conditions_dof(problem::KSBVPProblem{T,N}, mesh::KSMesh) where {T,N}
    if !has_boundary_condition(problem)
        return true
    end

    robin_count = count(bc -> bc isa KSRobinBC, problem.boundary_conditions)
    return robin_count * (N + 1) <= get_mesh_dof(mesh)
end

#===== Temporal DOF Functions =====#

"""Get temporal DOFs with improved stability"""
function get_temporal_dof(problem::AbstractKSProblem)::Int
	if !hasproperty(problem, :tspan) || isnothing(problem.tspan)
		return 1
	end

	dt = if hasproperty(problem, :dt)
		problem.dt
	else
		T = eltype(problem.tspan)
		(problem.tspan[2] - problem.tspan[1]) / T(100)
	end

	return ceil(Int, (problem.tspan[2] - problem.tspan[1]) / dt)
end

"""Get temporal DOFs for optimal control"""
function get_temporal_dof(problem::KSOptimalControlProblem{T})::Int where T
	return problem.num_time_steps
end

"""
	has_temporal_dof(problem)

Determine if a problem has temporal degrees of freedom.
"""
function has_temporal_dof(problem::AbstractKSProblem)::Bool
	if !hasfield(typeof(problem), :tspan)
		return false
	end

	# Early return if tspan is nothing or empty
	if isnothing(problem.tspan) || (problem.tspan isa Tuple && isempty(problem.tspan))
		return false
	end

	return true
end

function has_temporal_dof(problem::KSBVPProblem)::Bool
    if !hasfield(typeof(problem), :tspan)
        return false
    end
    return !isnothing(problem.tspan)
end

function has_temporal_dof(problem::KSOptimalControlProblem)::Bool
	return true
end

function has_temporal_dof(problem::KSCoupledProblem)::Bool
	return any(p -> has_temporal_dof(p), problem.problems)
end

"""Get temporal DOFs for coupled problems"""
function get_temporal_dof(problem::KSCoupledProblem)::Int
	# Get maximum temporal DOFs from subproblems
	return maximum(get_temporal_dof.(problem.problems))
end

#===== Combined DOF Calculations =====#

"""Calculate total DOFs for a problem"""
function get_total_problem_dof(
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T <: Number, N}
	sys_dof = get_system_dof(problem)
	mesh_dof = get_mesh_dof(mesh)

	# Include boundary DOFs if applicable
	if hasfield(typeof(problem), :boundary_conditions)
		boundary_dof = compute_boundary_dofs(problem.boundary_conditions, mesh)
		sys_dof += boundary_dof
	end

	# Include temporal DOFs if applicable
	if has_temporal_dof(problem)
		return sys_dof * mesh_dof * get_temporal_dof(problem)
	end

	return sys_dof * mesh_dof
end

"""Calculate total DOFs for optimal control problems"""
function get_total_problem_dof(
	problem::KSOptimalControlProblem{T},
	mesh::KSMesh{S, N}) where {T <: Number, S <: Number, N}
	system_dofs::Int = get_system_dof(problem)
	mesh_dofs::Int = get_mesh_dof(mesh)
	temporal_dofs::Int = get_temporal_dof(problem)

	# Optimal control problems always have temporal DOFs
	return system_dofs * mesh_dofs * temporal_dofs
end

"""Calculate total DOFs for coupled problems"""
function get_total_problem_dof(
	problem::KSCoupledProblem,
	mesh::KSMesh{T, N}) where {T <: Number, N}

	# Compute mesh DOFs once
	mesh_dofs::Int = get_mesh_dof(mesh)

	# Get system DOFs with optimized calculation
	system_dofs::Int = get_system_dof(problem)

	# Calculate spatial DOFs
	spatial_dofs::Int = system_dofs * mesh_dofs

	# Include temporal DOFs only if any subproblem has them
	if has_temporal_dof(problem)
		temporal_dofs::Int = get_temporal_dof(problem)
		return spatial_dofs * temporal_dofs
	end

	return spatial_dofs
end

"""
Compute additional DOFs needed for problem coupling.
"""
function compute_coupling_points(
	prob1::AbstractKSProblem,
	prob2::AbstractKSProblem,
	coupling::Union{Function, Vector{<:Function}})::Int
	dof1::Int = get_system_dof(prob1)
	dof2::Int = get_system_dof(prob2)

	n_eqns::Int = if coupling isa Vector
		length(coupling)
	elseif coupling isa Function
		1
	else
		0
	end

	return max(dof1, dof2) ÷ 4 * n_eqns
end

function compute_kernel_complexity(kernel::Function)::Int
	# Test points for sample evaluation
	test_points = [
		(0.0, 0.0), (0.5, 0.5), (1.0, 1.0),
		(0.25, 0.75), (0.75, 0.25),
	]

	try
		# Sample kernel values
		samples = [kernel(x, y) for (x, y) in test_points]

		# Analyze kernel behavior
		variations = diff(samples)
		max_variation = maximum(abs.(variations))

		# Estimate required quadrature points
		if max_variation < sqrt(eps())
			return 3  # Constant or near-constant kernel
		elseif max_variation < 1e-6
			return 5  # Slowly varying kernel
		elseif max_variation < 1e-3
			return 7  # Moderately varying kernel
		elseif !any(isnan.(samples)) && !any(isinf.(samples))
			return 11  # Rapidly varying but well-behaved kernel
		else
			return 15  # Potentially singular or ill-behaved kernel
		end
	catch
		# If kernel evaluation fails, assume complex behavior
		return 15
	end
end

# Method for scaled kernels
function compute_kernel_complexity(kernel::Function, scale::Real)::Int
	base_points = compute_kernel_complexity(kernel)
	return ceil(Int, base_points * abs(scale))
end

# Method for composite kernels
function compute_kernel_complexity(kernels::Vector{<:Function})::Int
	individual_points = compute_kernel_complexity.(kernels)
	return maximum(individual_points)
end

# Method for IDE kernels using NTuple input
function compute_kernel_complexity(kernel::Function,
	domain::NTuple{N, Tuple{T, T}}) where {N, T <: Number}
	# Test with midpoint of domain
	midpoints = ntuple(i -> sum(domain[i]) / 2, N)

	# Use partial kernel evaluation at midpoints
	function partial_kernel(x, y)
		return kernel(ntuple(i -> i == 1 ? x : midpoints[i], N),
			ntuple(i -> i == 1 ? y : midpoints[i], N))
	end

	return compute_kernel_complexity(partial_kernel)
end

# Fallback method
function compute_kernel_complexity(kernel::Any)::Int
	return 7  # Default to moderate complexity
end
