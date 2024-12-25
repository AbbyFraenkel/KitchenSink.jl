# TimeStepping.jl

module TimeStepping

using LinearAlgebra, SparseArrays
using OrdinaryDiffEq
using ..KSTypes, ..ProblemTypes, ..SpectralMethods, ..CacheManagement, ..NumericUtilities

export solve_time_dependent_problem, to_ode_problem
export get_time_stepper, create_time_discretization
export extract_solution_at_time

# OrdinaryDiffEq.jl Integration

function solve_time_dependent_problem(
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N};
	alg = AutoVern7(Rodas5()),
	dt = 0.01,
	kwargs...) where {T, N}

	# Convert to ODEProblem
	ode_prob = to_ode_problem(problem, mesh)

	# Add problem-specific options
	opts = process_problem_options(problem, dt)

	# Solve with OrdinaryDiffEq
	sol = solve(ode_prob, alg; opts..., kwargs...)

	return sol
end

function to_ode_problem(
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}

	# Create initial condition
	u0 = get_initial_condition(problem, mesh)

	# Create ODE function
	f = create_ode_function(problem, mesh)

	# Get timespan
	tspan = problem.tspan

	# Create ODEProblem
	return ODEProblem{true}(f, u0, tspan)
end

# ODE Function Creation

function create_ode_function(
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}

	# Pre-allocate system matrices
	A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)

	# Create in-place update function
	function f!(du, u, p, t)
		# Update time-dependent terms if needed
		update_time_dependent_terms!(A, b, t, problem, mesh)

		# Compute derivative
		mul!(du, A, u)
		du .+= b
		return nothing
	end

	return f!
end

# Problem-Specific Handlers

function process_problem_options(
	problem::AbstractKSProblem,
	dt::Real)
	opts = Dict{Symbol, Any}()

	# Problem-specific options
	if problem isa KSPDEProblem
		opts[:dtmax] = dt  # Maximum timestep for PDEs
		opts[:adaptive] = true
	elseif problem isa KSDAEProblem
		opts[:diffstates] = collect(.!problem.algebraic_vars)
		opts[:dtmax] = dt
	elseif problem isa KSMovingBoundaryPDEProblem
		opts[:adaptive] = true
		opts[:dtmax] = dt
		opts[:force_dtmin] = true  # For mesh motion
	end

	# Common options
	opts[:saveat] = collect(problem.tspan[1]:dt:problem.tspan[2])

	return opts
end

# Time-dependent Updates

function update_time_dependent_terms!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	t::Real,
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}
	if problem isa KSMovingBoundaryPDEProblem
		update_moving_boundary!(A, b, t, problem, mesh)
	elseif problem isa KSIDEProblem
		update_integral_terms!(A, b, t, problem, mesh)
	elseif problem isa KSPIDEProblem
		update_pide_terms!(A, b, t, problem, mesh)
	end
end

function update_moving_boundary!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	t::Real,
	problem::KSMovingBoundaryPDEProblem,
	mesh::KSMesh{T, N}) where {T, N}

	# Compute new domain position
	new_position = problem.boundary_motion.(mesh.nodes, t)

	# Update mesh velocities
	mesh_velocities = compute_mesh_velocities(new_position, mesh)

	# Update ALE terms in system
	return update_ale_terms!(A, b, mesh_velocities, problem, mesh)
end

function update_integral_terms!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	t::Real,
	problem::KSIDEProblem,
	mesh::KSMesh{T, N}) where {T, N}

	# Update kernel evaluation if time-dependent
	for cell in mesh.cells
		if !cell.is_fictitious
			update_cell_integral!(A, b, t, cell, problem, mesh)
		end
	end
end

function update_pide_terms!(
	A::AbstractMatrix{T},
	b::AbstractVector{T},
	t::Real,
	problem::KSPIDEProblem,
	mesh::KSMesh{T, N}) where {T, N}

	# Update both PDE and integral terms
	update_pide_pde_terms!(A, b, t, problem, mesh)
	return update_pide_integral_terms!(A, b, t, problem, mesh)
end

# Solution Processing

function extract_solution_at_time(
	sol::ODESolution,
	t::Real,
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}

	# Interpolate solution at time t
	u_t = sol(t)

	# Convert to physical space if needed
	if needs_physical_transform(problem)
		u_t = transform_to_physical_space(u_t, problem, mesh)
	end

	return u_t
end

# Utility Functions

function get_initial_condition(
	problem::AbstractKSProblem,
	mesh::KSMesh{T, N}) where {T, N}
	if isa(problem.initial_conditions, Function)
		# Evaluate function at mesh nodes
		return evaluate_at_nodes(
			problem.initial_conditions, mesh)
	else
		# Use provided vector
		return copy(problem.initial_conditions)
	end
end

function evaluate_at_nodes(
	f::Function,
	mesh::KSMesh{T, N}) where {T, N}
	u0 = zeros(T, get_total_dof(mesh))

	for (i, node) in enumerate(mesh.nodes)
		u0[i] = f(node)
	end

	return u0
end

function needs_physical_transform(problem::AbstractKSProblem)
	return !isa(problem.coordinate_system, KSCartesianCoordinates)
end

end # module TimeStepping
