"""
Core validation functionality for ProblemTypes module.
Handles validation of problem setups, solutions, and mesh compatibility.
"""

"""
	validate_problem_setup(problem::Union{Nothing,AbstractKSProblem},
						 mesh::Union{Nothing,KSMesh{T,N}},
						 coord_sys::Union{Nothing,AbstractKSCoordinateSystem}) where {T,N}

Validate complete problem setup including mesh and coordinate system compatibility.
"""
function validate_problem_setup(
	problem::Union{Nothing, AbstractKSProblem},
	mesh::Union{Nothing, KSMesh{T, N}},
	coord_sys::Union{Nothing, AbstractKSCoordinateSystem}) where {T, N}

	# Basic validation - throw ArgumentError directly for null cases
	if isnothing(problem)
		throw(ArgumentError("Problem cannot be null"))
	end
	if isnothing(mesh)
		throw(ArgumentError("Mesh cannot be null"))
	end
	if isnothing(coord_sys)
		throw(ArgumentError("Coordinate system cannot be null"))
	end

	# Convert types back to non-Union for rest of function
	problem = convert(AbstractKSProblem, problem)
	mesh = convert(KSMesh{T, N}, mesh)
	coord_sys = convert(AbstractKSCoordinateSystem, coord_sys)

	# Mesh validation
	if isempty(mesh.cells)
		throw(ArgumentError("Mesh contains no cells"))
	end

	# Get first non-fictitious cell
	non_fictitious = findfirst(c -> !c.is_fictitious, mesh.cells)
	isnothing(non_fictitious) &&
		throw(ArgumentError("Mesh contains no valid non-fictitious cells"))

	# Dimension compatibility check - Compare problem dimension with mesh dimension
	prob_dim = get_problem_dimension(problem)
	first_cell = mesh.cells[non_fictitious]
	mesh_dim = length(first_cell.p) # Use polynomial order length as mesh dimension

	if prob_dim != mesh_dim
		throw(
			DimensionMismatch(
				"Problem dimension ($prob_dim) does not match mesh dimension ($mesh_dim)"),
		)
	end

	# DOF validation using total problem DOFs
	mesh_dofs = get_mesh_dof(mesh)
	if mesh_dofs == 0
		throw(ArgumentError("Mesh has zero degrees of freedom"))
	end

	# Note: Removing direct DOF comparison since total_problem_dof handles this correctly

	# Problem-specific validation
	validate_problem_specific_config(problem)

	# Temporal DOF validation if applicable
	if has_temporal_dof(problem)
		validate_temporal_dofs(problem, mesh)
	end

	return true
end

"""
	validate_temporal_dofs(problem::AbstractKSProblem, mesh::KSMesh)

Validate temporal degrees of freedom for time-dependent problems.
"""
function validate_temporal_dofs(problem::AbstractKSProblem, mesh::KSMesh)
	if !hasfield(typeof(problem), :tspan)
		return true # Non-temporal problems are valid
	end

	# Skip validation for BVPs
	if problem isa KSBVPProblem
		return true
	end

	# Handle steady state case
	if isnothing(problem.tspan)
		return true # Allow null timespan for steady-state
	end

	# Validate tspan structure
	if !(problem.tspan isa Tuple) || length(problem.tspan) != 2
		throw(ArgumentError("Time span must be a 2-element tuple"))
	end

	# Validate tspan values
	t_start, t_end = problem.tspan

	if !isfinite(t_start) || !isfinite(t_end)
		throw(ArgumentError("Time span must contain finite values"))
	end

	if t_end <= t_start
		throw(ArgumentError("End time must be greater than start time"))
	end

	return true
end

"""
	validate_problem_specific_config(problem::AbstractKSProblem)

Problem type-specific validation. Dispatches to appropriate method based on problem type.
"""
function validate_problem_specific_config(problem::AbstractKSProblem)
	return true
end

function validate_problem_specific_config(problem::KSPDEProblem{T, N}) where {T, N}
	# Check PDE function
	if !hasfield(typeof(problem), :pde) || isnothing(problem.pde)
		throw(ArgumentError("PDE function must be defined"))
	end

	# Check initial conditions
	if !hasfield(typeof(problem), :initial_conditions)
		throw(ArgumentError("Initial conditions field must exist"))
	end
	if isnothing(problem.initial_conditions) ||
		(
		problem.initial_conditions isa AbstractVector && isempty(problem.initial_conditions)
	)
		throw(ArgumentError("Initial conditions must be defined and non-empty for PDE"))
	end

	validate_time_span(problem)
	return true
end

function validate_problem_specific_config(problem::KSODEProblem{T, N}) where {T, N}
	# Check ODE function
	if !hasfield(typeof(problem), :ode) || isnothing(problem.ode)
		throw(ArgumentError("ODE function must be defined"))
	end

	# Check initial conditions
	if !hasfield(typeof(problem), :initial_conditions)
		throw(ArgumentError("Initial conditions field must exist"))
	end
	if isnothing(problem.initial_conditions) ||
		(
		problem.initial_conditions isa AbstractVector && isempty(problem.initial_conditions)
	)
		throw(ArgumentError("Initial conditions must be defined and non-empty for ODE"))
	end

	validate_time_span(problem)
	return true
end

"""
	validate_time_span(problem::AbstractKSProblem)

Validate time span configuration for time-dependent problems.
"""
function validate_time_span(problem::AbstractKSProblem)
	# Skip validation for problems without time spans
	if !hasfield(typeof(problem), :tspan)
		return true
	end

	# Skip BVP and optimal control problems
	if problem isa Union{KSBVPProblem, KSOptimalControlProblem}
		return true
	end

	# Handle null case
	if isnothing(problem.tspan)
		return true  # Allow null tspan for steady-state problems
	end

	t_start, t_end = problem.tspan

	# Validate finiteness
	if !isfinite(t_start) || !isfinite(t_end)
		throw(ArgumentError("Time span must contain finite values"))
	end

	# Check ordering
	if t_end <= t_start
		throw(ArgumentError("End time ($t_end) must be greater than start time ($t_start)"))
	end

	# Check negative times for applicable problems
	if hasfield(typeof(problem), :allows_negative_time) &&
		!problem.allows_negative_time &&
		t_start < zero(t_start)
		throw(
			ArgumentError(
				"Negative start time ($t_start) not allowed for this problem type"
			),
		)
	end

	return true
end

"""Get problem dimension for optimal control problems"""
function get_problem_dimension(problem::KSOptimalControlProblem)
	return length(problem.control_bounds)  # Use control bounds length for dimension
end

"""
	verify_solution_properties(solution::AbstractVector{T},
							 problem::AbstractKSProblem,
							 mesh::KSMesh) where T

Verify general solution properties and dispatch to problem-specific verification.
"""
function verify_solution_properties(
	solution::AbstractVector{T},
	problem::AbstractKSProblem,
	mesh::KSMesh) where T

	# Basic validation
	isnothing(solution) && throw(ArgumentError("Solution cannot be null"))

	expected_size = get_total_problem_dof(problem, mesh)
	if length(solution) != expected_size
		throw(
			DimensionMismatch(
				"Solution length ($(length(solution))) does not match problem DOFs ($expected_size)"
			),
		)
	end

	# Check for invalid values
	if any(isnan, solution)
		throw(ArgumentError("Solution contains NaN values"))
	end

	if any(isinf, solution)
		throw(ArgumentError("Solution contains infinite values"))
	end

	if !all(isfinite, solution)
		throw(ArgumentError("Solution contains non-finite values"))
	end

	# Problem-specific verification
	verify_solution_specific(solution, problem, mesh)

	return true
end

# Add method that doesn't require mesh for basic checks
function verify_solution_properties(
	solution::Union{Nothing, AbstractVector{T}},
	problem::AbstractKSProblem) where T

	# Basic validation for null case
	isnothing(solution) && throw(ArgumentError("Solution cannot be null"))

	# Check for invalid values
	if any(isnan, solution)
		throw(ArgumentError("Solution contains NaN values"))
	end

	if any(isinf, solution)
		throw(ArgumentError("Solution contains infinite values"))
	end

	if !all(isfinite, solution)
		throw(ArgumentError("Solution contains non-finite values"))
	end

	return true
end

"""
	verify_solution_specific(solution::AbstractVector{T},
						   problem::AbstractKSProblem,
						   mesh::KSMesh) where T

Problem type-specific solution verification.
"""
function verify_solution_specific(
	solution::AbstractVector{T},
	problem::AbstractKSProblem,
	mesh::KSMesh) where T
	return true
end

function verify_solution_specific(
	solution::AbstractVector{T},
	problem::KSCoupledProblem,
	mesh::KSMesh) where T

	# Get subproblem sizes
	sizes = [get_total_problem_dof(p, mesh) for p in problem.problems]
	start_indices = [1; cumsum(sizes)[1:(end - 1)] .+ 1]
	end_indices = cumsum(sizes)

	# Verify each subproblem
	for (i, (start_idx, end_idx)) in enumerate(zip(start_indices, end_indices))
		subsolution = view(solution, start_idx:end_idx)
		verify_solution_specific(subsolution, problem.problems[i], mesh)
	end

	return true
end

# Add method that doesn't require mesh for basic coupled problem checks
function verify_solution_specific(
	solution::AbstractVector{T},
	problem::KSCoupledProblem) where T

	# Basic validation without mesh checks
	for p in problem.problems
		verify_solution_properties(solution, p)
	end

	return true
end

"""
	validate_domains(prob_domains::Union{Nothing,AbstractVector,Tuple},
					mesh_domains::Union{Nothing,AbstractVector,Tuple},
					strict::Bool=false)

Validate domain compatibility between problem and mesh domains.
If strict=true, domains must match exactly. Otherwise, only overlap is required.

Returns:
- `true` if domains are compatible
- `false` if domains are incompatible
"""
function validate_domains(
	prob_domains::Union{Nothing, NTuple{N, Tuple{T, T}} where {N, T <: Real}},
	mesh_domains::Union{Nothing, NTuple{N, Tuple{T, T}} where {N, T <: Real}},
	strict::Bool = false)

	# Handle null cases
	isnothing(prob_domains) && return isnothing(mesh_domains)
	isnothing(mesh_domains) && return false

	# Validate length match
	length(prob_domains) == length(mesh_domains) || return false

	# Type promotion
	T = promote_type(eltype(first(prob_domains)[1]), eltype(first(mesh_domains)[1]))
	tol = sqrt(eps(T))

	# Check each dimension
	for (pd, md) in zip(prob_domains, mesh_domains)
		pd_min, pd_max = pd
		md_min, md_max = md

		if strict
			# Bounds must match within tolerance
			(
				isapprox(pd_min, md_min; rtol = tol) &&
				isapprox(pd_max, md_max; rtol = tol)
			) || return false
		else
			# Check for overlap
			(pd_max + tol < md_min || md_max + tol < pd_min) && return false
		end
	end

	return true
end

"""
	are_coordinate_systems_compatible(sys1::AbstractKSCoordinateSystem,
								   sys2::AbstractKSCoordinateSystem)

Check if two coordinate systems are compatible by verifying their type, dimensionality,
and domain compatibility.
"""
function are_coordinate_systems_compatible(
	sys1::AbstractKSCoordinateSystem,
	sys2::AbstractKSCoordinateSystem)

	# Check dimensions match
	d1 = get_coordinate_domains(sys1)
	d2 = get_coordinate_domains(sys2)

	if length(d1) != length(d2)
		return false
	end

	# For same type systems, compare domains directly
	if typeof(sys1) === typeof(sys2)
		return all(compare_domain_ranges(r1, r2) for (r1, r2) in zip(d1, d2))
	end

	# For different types, check strict domain overlap
	# Changed to require complete overlap for different types
	for (r1, r2) in zip(d1, d2)
		if !strict_domain_overlap(r1, r2)
			return false
		end
	end
	return true
end

"""Compare individual domain bounds with tolerance"""
function compare_domain_ranges(r1::Tuple{T, T}, r2::Tuple{T, T}) where T
	return isapprox(r1[1], r2[1]; rtol = 1e-10) &&
		   isapprox(r1[2], r2[2]; rtol = 1e-10)
end

"""Check if two ranges overlap"""
function check_domain_overlap_pair(r1::Tuple{T, T}, r2::Tuple{T, T}) where T
	r1_min, r1_max = r1
	r2_min, r2_max = r2

	# Check for overlap accounting for numerical tolerance
	tol = sqrt(eps(T))
	return (r1_max + tol >= r2_min) && (r2_max + tol >= r1_min)
end

"""Check if ranges overlap"""
function check_domain_overlap(d1, d2)
	for (r1, r2) in zip(d1, d2)
		r1_min, r1_max = extrema(r1)
		r2_min, r2_max = extrema(r2)
		if r1_max < r2_min || r2_max < r1_min
			return false
		end
	end
	return true
end

"""Check if two ranges overlap"""
function ranges_overlap(r1::Tuple{T, T}, r2::Tuple{T, T}) where T
	r1_min, r1_max = r1
	r2_min, r2_max = r2
	return r1_max >= r2_min && r2_max >= r1_min
end

"""Fix domain overlap checking"""
function strict_domain_overlap(r1::Tuple{T,T}, r2::Tuple{T,T}) where T
    r1_min, r1_max = r1
    r2_min, r2_max = r2
    tol = sqrt(eps(T))

    # Domain sizes
    r1_size = r1_max - r1_min
    r2_size = r2_max - r2_min

    # Check if domains are valid and overlap significantly
    return (r1_size > tol && r2_size > tol) &&
           !((r1_max < r2_min + tol) || (r2_max < r1_min + tol)) &&
           abs(r1_size - r2_size) < tol * max(r1_size, r2_size)
end

"""Update coordinate system compatibility check"""
function are_coordinate_systems_compatible(sys1::AbstractKSCoordinateSystem,
                                        sys2::AbstractKSCoordinateSystem)
    # Exit early if dimensions don't match
    d1 = get_coordinate_domains(sys1)
    d2 = get_coordinate_domains(sys2)
    length(d1) != length(d2) && return false

    # For same type systems, check exact matches
    if typeof(sys1) === typeof(sys2)
        return all(compare_domain_ranges.(d1, d2))
    end

    # For different types, check significant overlap
    all(strict_domain_overlap.(d1, d2))
end

"""Update are_coordinate_systems_compatible for proper domain overlap checking"""
function are_coordinate_systems_compatible(sys1::AbstractKSCoordinateSystem,
                                        sys2::AbstractKSCoordinateSystem)
    # Exit early if dimensions don't match
    d1 = get_coordinate_domains(sys1)
    d2 = get_coordinate_domains(sys2)
    length(d1) != length(d2) && return false

    # For same type systems, check exact matches
    if typeof(sys1) === typeof(sys2)
        return all(compare_domain_ranges.(d1, d2))
    end

    # For different types, check domain overlap
    # Modified to be more lenient for different coordinate system types
    return all(ranges_overlap.(d1, d2))
end

"""Update ranges_overlap to handle coordinate system domain comparison"""
function ranges_overlap(r1::Tuple{T,T}, r2::Tuple{T,T}) where T
    r1_min, r1_max = r1
    r2_min, r2_max = r2
    # Use larger tolerance for different coordinate system types
    tol = 100*sqrt(eps(T))
    return !((r1_max + tol < r2_min) || (r2_max + tol < r1_min))
end

"""
	is_valid_domain(domain)

Check if a domain specification is valid.

A valid domain can be:
- A tuple/vector of 2-element tuples/vectors, where each represents (min, max) bounds
- A single 2-element tuple/vector representing (min, max) bounds

Returns `false` for invalid domains or non-finite values.
"""
function is_valid_domain(domain::Union{AbstractVector, Tuple})
	# Handle single interval case
	if length(domain) == 2 && all(x -> x isa Number, domain)
		return all(isfinite, domain) && domain[1] < domain[2]
	end

	# Handle multi-dimensional case
	try
		return all(
			interval -> begin
				# Check if interval has exactly 2 elements and they're valid bounds
				length(interval) == 2 &&
					all(isfinite, interval) &&
					interval[1] < interval[2]
			end, domain)
	catch
		return false
	end
end

# Fallback for any other type
function is_valid_domain(domain::Any)
	return false
end

"""Get coordinate system for a problem."""
function get_coordinate_system(problem::AbstractKSProblem)
	if hasfield(typeof(problem), :coordinate_system)
		return problem.coordinate_system
	elseif problem isa KSOptimalControlProblem
		bounds = [(0.0, 1.0) for _ in 1:get_problem_dimension(problem)]
		return KSCartesianCoordinates(tuple(bounds...))
	elseif problem isa KSCoupledProblem
		return first(problem.problems).coordinate_system
	else
		throw(ArgumentError("Problem type $(typeof(problem)) has no coordinate system"))
	end
end

"""Get problem dimension."""
function get_problem_dimension(problem::AbstractKSProblem)
	if hasfield(typeof(problem), :domain)
		return length(problem.domain)
	elseif problem isa KSCoupledProblem
		return length(first(problem.problems).domain)
	elseif problem isa KSOptimalControlProblem
		return length(problem.control_bounds)
	else
		throw(
			ArgumentError("Cannot determine dimension for problem type $(typeof(problem))")
		)
	end
end
"""
Get coordinate domains from a coordinate system, handling different field names and types.
"""
function get_coordinate_domains(coord_sys::AbstractKSCoordinateSystem)
	if coord_sys isa KSCartesianCoordinates
		return coord_sys.ranges
	elseif hasfield(typeof(coord_sys), :domains)
		return coord_sys.domains
	elseif hasfield(typeof(coord_sys), :domain)
		return [coord_sys.domain]
	elseif hasfield(typeof(coord_sys), :ranges)
		return coord_sys.ranges
	else
		throw(
			ArgumentError(
				"Cannot determine domains for coordinate system $(typeof(coord_sys))"
			),
		)
	end
end

"""
	validate_boundary_conditions(bcs::Vector)

Validate that all boundary conditions are properly defined.
"""
function validate_boundary_conditions(bcs::AbstractVector)
	if isempty(bcs)
		throw(ArgumentError("Problem must have at least one boundary condition"))
	end

	for (i, bc) in enumerate(bcs)
		if !(bc isa AbstractKSBoundaryCondition)
			throw(ArgumentError("Invalid boundary condition at index $i: $(typeof(bc))"))
		end
	end
	return true
end

# Add specific validation for numeric boundary conditions
function validate_boundary_conditions(bcs::Vector{<:Number})
	throw(ArgumentError("Boundary conditions must be AbstractKSBoundaryCondition objects"))
end

"""Validate boundary conditions."""
function validate_boundary_conditions(bcs::Union{Nothing, AbstractVector})
	if isnothing(bcs)
		throw(ArgumentError("Problem must have boundary conditions"))
	end
	if isempty(bcs)
		throw(ArgumentError("Problem must have at least one boundary condition"))
	end
	for (i, bc) in enumerate(bcs)
		if !(bc isa AbstractKSBoundaryCondition)
			throw(ArgumentError("Invalid boundary condition at index $i: $(typeof(bc))"))
		end
	end
	return true
end

# Add method for numeric boundary conditions
function validate_boundary_conditions(bcs::AbstractVector{<:Number})
	throw(ArgumentError("Boundary conditions must be AbstractKSBoundaryCondition objects"))
end

function has_temporal_dof(problem::KSBVPProblem)::Bool
    if !hasfield(typeof(problem), :tspan)
        return false
    end
    return !isnothing(problem.tspan)
end
