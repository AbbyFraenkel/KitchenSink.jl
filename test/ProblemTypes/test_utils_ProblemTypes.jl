using LinearAlgebra, StaticArrays, FastGaussQuadrature, SparseArrays
using KitchenSink.SpectralMethods, KitchenSink.KSTypes, KitchenSink.CoordinateSystems
using KitchenSink.BoundaryConditions, KitchenSink.Transforms
using KitchenSink.ProblemTypes

# Constants for testing
TEST_DIMENSIONS = (1, 2, 3)
TEST_PROBLEM_TYPES = [
    :pde,                # Partial Differential Equations
    :ode,                # Ordinary Differential Equations
    :dae,                # Differential Algebraic Equations
    :bvp,                # Boundary Value Problems
    :ide,                # Integral Differential Equations
    :pide,               # Partial Integral Differential Equations
    :bvdae,              # Boundary Value DAEs
    :optimal_control,    # Optimal Control Problems
    :moving_boundary,    # Moving Boundary Problems
    :coupled_problem,     # Coupled Systems
]

TEST_POLYNOMIAL_ORDERS = (3, 5, 8, 13, 21, 34)

"""Create a test problem of specified type and dimension."""
function create_test_problem(prob_type::Symbol, dim::Int; kwargs...)
    domain = ntuple(i -> (-1.0, 1.0), dim)
    coord_sys = KSCartesianCoordinates(domain)

    # Add default temporal settings
    default_kwargs = Dict{Symbol, Any}(
        :tspan => (0.0, 1.0)
    )

    # Merge with user provided kwargs
    merged_kwargs = merge(default_kwargs, Dict(kwargs))
    delete!(merged_kwargs, :dt)  # Remove dt since not used by constructors

    # Create problem based on type
    return if prob_type == :pde
        create_test_pde(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :ode
        create_test_ode(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :dae
        create_test_dae(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :bvp
        create_test_bvp(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :ide
        create_test_ide(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :pide
        create_test_pide(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :bvdae
        create_test_bvdae(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :optimal_control
        create_test_optimal_control(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :moving_boundary
        create_test_moving_boundary(coord_sys, domain; merged_kwargs...)
    elseif prob_type == :coupled_problem
        create_test_coupled(coord_sys, domain; merged_kwargs...)
    else
        throw(ArgumentError("Unsupported problem type: $prob_type"))
    end

    @eval begin
        # Precompile BVP creation
        precompile(create_test_bvp, (AbstractKSCoordinateSystem, Any))
        precompile(KSBVPProblem, (Function, Vector, Any, Function, AbstractKSCoordinateSystem, Int, Any))
    end
end

# Individual problem creation functions with complete specifications
function create_test_pde(coord_sys, domain; kwargs...)
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 1)
	tspan = get(constructor_kwargs, :tspan, (0.0, 1.0))

	return KSPDEProblem(;
		pde = (x, u, D, t) -> begin  # Add time parameter and fix signature
			# Simple diffusion operator - explicitly specify type
			T = eltype(u)
			# Simple unit diffusion with diagonal identity term
			A = Matrix{T}(I, num_vars, num_vars)
			b = zero(T)
			return (A, b)
		end,
		boundary_conditions = create_test_boundary_conditions(coord_sys),
		domain = domain,
		initial_conditions = zeros(num_vars),
		coordinate_system = coord_sys,
		num_vars = num_vars,
		tspan = tspan,
		constructor_kwargs...,
	)
end

function create_test_ode(coord_sys, domain; kwargs...)
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 1)
	tspan = get(constructor_kwargs, :tspan, (0.0, 1.0))

	return KSODEProblem(;
		# Fix: Return scalar value instead of tuple
		ode = (u, t) -> -1.0,  # Simple decay, return scalar
		boundary_conditions = create_test_boundary_conditions(coord_sys),
		domain = domain,
		initial_conditions = ones(num_vars),
		coordinate_system = coord_sys,
		num_vars = num_vars,
		tspan = tspan,
		constructor_kwargs...,
	)
end

# Update create_test_dae function to properly handle constructor arguments
function create_test_dae(coord_sys::AbstractKSCoordinateSystem,
                        domain::NTuple{N,Tuple{T,T}};
                        kwargs...) where {N,T<:Real}
    # Extract constructor parameters with defaults
    num_vars = get(kwargs, :num_vars, 2)
    num_alg = get(kwargs, :num_algebraic_vars, 1)
    tspan = get(kwargs, :tspan, (zero(T), one(T)))

    # Create DAE function with correct signature
    function dae_fn(t, y, dy, v)
        du = similar(y)
        du[1:end-num_alg] .= dy[1:end-num_alg]
        du[end-num_alg+1:end] .= zero(T)
        return du
    end

    return KSDAEProblem(;
        dae = dae_fn,
        boundary_conditions = [KSDirichletBC(;
            boundary_value = x -> zero(T),
            boundary_region = x -> true,
            coordinate_system = coord_sys
        )],
        domain = domain,
        initial_conditions = zeros(T, num_vars),
        tspan = tspan,
        coordinate_system = coord_sys,
        num_vars = num_vars,
        num_algebraic_vars = num_alg
    )
end

# Update BVP test problem creation to handle arrays correctly
function create_test_bvp(coord_sys::Symbol, domain; kwargs...)
    # Convert symbol to actual coordinate system
    actual_coord_sys = if coord_sys == :cartesian
        # Always wrap domain in a tuple for proper typing
        wrapped_domain = if domain isa Tuple{Real,Real}
            (domain,)  # Convert (a,b) to ((a,b),) for 1D
        else
            domain    # Keep higher dimensions as is
        end
        # Create proper type with dimension
        dim = domain isa Tuple{Real,Real} ? 1 : length(domain)
        KSCartesianCoordinates{Float64,dim}(wrapped_domain)
    else
        throw(ArgumentError("Unsupported coordinate system: $coord_sys"))
    end

    return create_test_bvp(actual_coord_sys, domain; kwargs...)
end

"""Update create_test_bvp to handle function signatures correctly"""
function create_test_bvp(coord_sys::AbstractKSCoordinateSystem,
                        domain; kwargs...)
    T = eltype(first(first(domain)))

    # Get parameters with defaults
    num_vars = get(kwargs, :num_vars, 1)
    tspan = get(kwargs, :tspan, nothing)

    # Create proper BVP function that returns tuples
    function bvp_function(u, du, t)
        return (Matrix{T}(I, num_vars, num_vars), zeros(T, num_vars))
    end

    # Create boundary conditions if not provided
    bcs = if haskey(kwargs, :boundary_conditions)
        kwargs[:boundary_conditions]
    else
        [KSDirichletBC(;
            boundary_value = x -> one(T),
            boundary_region = x -> true,
            coordinate_system = coord_sys
        )]
    end

    # Create initial guess function
    initial_guess_fn = x -> zeros(T, num_vars)

    return KSBVPProblem(;
        bvp = bvp_function,
        boundary_conditions = bcs,
        domain = domain,
        initial_guess = initial_guess_fn,
        coordinate_system = coord_sys,
        num_vars = num_vars,
        tspan = tspan
    )
end

# Fix IDE problem creation
function create_test_ide(coord_sys::AbstractKSCoordinateSystem,
	domain::NTuple{N, Tuple{T, T}};
	kwargs...) where {N, T <: Number}
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 1)
	tspan = get(constructor_kwargs, :tspan, (zero(T), one(T)))

	return KSIDEProblem(;
		ide = (x, t) -> -one(T) * x,  # Type-stable linear IDE
		kernel = (x, y) -> exp(-abs(sum(x .- y))),  # Stable multidimensional kernel
		boundary_conditions = create_test_boundary_conditions(coord_sys),
		domain = domain,
		initial_conditions = zeros(T, num_vars),
		tspan = tspan,
		coordinate_system = coord_sys,
		num_vars = num_vars,
		constructor_kwargs...,
	)
end

"""
	get_polynomial_order(mesh::KSMesh)

Get minimum polynomial order from mesh cells.
"""
function get_polynomial_order(mesh::KSMesh)
	# Get minimum polynomial order across all cells for stability
	return minimum(minimum(cell.p) for cell in mesh.cells)
end

# Fix PIDE problem creation by removing polynomial_order argument
function create_test_pide(coord_sys::AbstractKSCoordinateSystem,
	domain::NTuple{N, Tuple{T, T}};
	kwargs...) where {N, T <: Number}
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 1)
	tspan = get(constructor_kwargs, :tspan, (zero(T), one(T)))

	return KSPIDEProblem(;
		pide = (u, x, t) -> -one(T) .* u,  # Type-stable operator
		kernel = (x, y) -> exp(-abs(sum(x .- y))),  # Consistent kernel
		boundary_conditions = create_test_boundary_conditions(coord_sys),
		domain = domain,
		initial_conditions = zeros(T, num_vars),
		tspan = tspan,
		coordinate_system = coord_sys,
		num_vars = num_vars,
		# Remove polynomial_order as it's not part of the constructor
		constructor_kwargs...,
	)
end

function create_test_bvdae(coord_sys, domain; kwargs...)
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 2)
	algebraic_vars = get(constructor_kwargs, :algebraic_vars, [false, true])

	return KSBVDAEProblem(;
		f = (u, du, t) -> du[1] - u[2],
		g = (u, t) -> u[1] + u[2] - 1,
		boundary_conditions = create_test_boundary_conditions(coord_sys),
		domain = domain,
		initial_conditions = [1.0; 0.0],
		algebraic_vars = algebraic_vars,
		tspan = get(constructor_kwargs, :tspan, (0.0, 1.0)),
		coordinate_system = coord_sys,
		num_vars = num_vars,
	)
end
# Merge the two implementations into one function with all necessary keyword arguments:
"""Create optimal control test problem with proper dimension"""
function create_test_optimal_control(coord_sys, domain; kwargs...)
    # Extract all possible kwargs with defaults
    dim = length(domain)  # Get dimension from domain
    num_vars = get(kwargs, :num_vars, dim)  # Use dimension as default num_vars
    dt = get(kwargs, :dt, 0.01)
    t_span = get(kwargs, :tspan, (0.0, 1.0))

    # Handle infinite time spans
    if any(isinf, t_span)
        t_span = (0.0, 1.0)
    end

    # Create consistent cost functions
    cost_functions = [(t, u) -> u^2 + t^2 for _ in 1:num_vars]

    return KSOptimalControlProblem(;
        state_equations = [u -> -u for _ in 1:num_vars],
        cost_functions = cost_functions,
        terminal_cost = x -> sum(x .^ 2),
        initial_state = zeros(num_vars),
        t_span = t_span,
        control_bounds = [(-1.0, 1.0) for _ in 1:dim], # Use dimension for control bounds
        dt = dt
    )
end

# Update moving boundary problem creation
function create_test_moving_boundary(coord_sys, domain; kwargs...)
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 1)
	tspan = get(constructor_kwargs, :tspan, (0.0, 1.0))
	n_nodes = 5  # Fixed size for test

	return KSMovingBoundaryPDEProblem(;
		pde = (u, x, D) -> begin
			# Handle coefficients vector properly
			val = sum(u)  # Simple aggregation
			return (val, zero(eltype(u)))
		end,
		boundary_motion = (x...) -> 0.1 * sin(2π * first(x)),
		boundary_conditions = create_test_boundary_conditions(coord_sys),
		domain = domain,
		initial_conditions = zeros(n_nodes * num_vars),
		coordinate_system = coord_sys,
		num_vars = num_vars,
		tspan = tspan,
		constructor_kwargs...,
	)
end

# Fix coupled problem creation
function create_test_coupled(coord_sys::AbstractKSCoordinateSystem,
	domain::NTuple{N, Tuple{T, T}};
	kwargs...) where {N, T <: Number}
	# Remove dt from kwargs for constructor
	constructor_kwargs = filter(kv -> kv.first != :dt, kwargs)
	num_vars = get(constructor_kwargs, :num_vars, 1)
	n_nodes = get(constructor_kwargs, :n_nodes, 2)

	# Create consistent kwargs
	pde_kwargs = Dict{Symbol, Any}(
		:num_vars => num_vars,
		:initial_conditions => zeros(T, n_nodes * num_vars),
	)
	ode_kwargs = Dict{Symbol, Any}(
		:num_vars => num_vars,
		:initial_conditions => ones(T, n_nodes * num_vars),
	)

	# Create subproblems with proper type parameters
	pde_prob = create_test_pde(coord_sys, domain; pde_kwargs...)
	ode_prob = create_test_ode(coord_sys, domain; ode_kwargs...)

	# Type-stable coupling terms
	coupling_terms = Matrix{Union{Nothing, Function}}(nothing, 2, 2)
	coupling_terms[1, 2] = coupling_terms[2, 1] = (u1, u2) -> convert(T, sum(u1 .* u2))

	return KSCoupledProblem(;
		problems = [pde_prob, ode_prob],
		coupling_terms = coupling_terms,
	)
end

# Core test mesh creation
# Merge the two create_test_mesh functions into one with default arguments
function create_test_mesh(dim::Int;
	num_cells::NTuple{N, Int} = ntuple(_ -> 2, dim),
	p_orders::NTuple{N, Int} = ntuple(_ -> 5, dim)) where N
	domain = ntuple(i -> (-1.0, 1.0), dim)
	coord_sys = KSCartesianCoordinates(domain)

	# Ensure minimum polynomial order is 3
	validated_p_orders = map(p -> max(p, 3), p_orders)

	# Create mesh with uniform continuity order of 1
	continuity = ntuple(_ -> 1, dim)

	mesh = SpectralMethods.create_ocfc_mesh(
		coord_sys,
		num_cells,
		validated_p_orders,
		continuity,
		dim,
		1,  # Reduced max_level to 1
	)

	return (mesh, coord_sys)
end

# Update test solution creation to match boundary conditions
# Update create_test_solution to respect problem dimensions
"""Update create_test_solution to use get_total_problem_dof"""
function create_test_solution(problem::AbstractKSProblem, n::Int)
    if problem isa Union{KSIDEProblem, KSPIDEProblem}
        # Always create n-sized solution for IDE/PIDE
        return [cos(π * i / n) for i in 1:n]
    else
        # For other problems, use standard sizes
        return ones(n)  # Just use input size directly
    end
end

"""Update create_validation_solution to use get_total_problem_dof"""
function create_validation_solution(problem::AbstractKSProblem, mesh::KSMesh)
    n = get_total_problem_dof(problem, mesh)
    return create_test_solution(problem, n)
end

# Helper functions
# Update the boundary condition creation to ensure consistent behavior
function create_test_boundary_conditions(coord_sys::AbstractKSCoordinateSystem)
	return [
		KSDirichletBC(;
			boundary_value = x -> 1.0,  # Function returning constant
			boundary_region = x -> x == 1 || x == length(x),  # Only boundary points
			coordinate_system = coord_sys,
		),
	]
end

# Fix laplacian function to handle 1D case correctly
function laplacian(u::AbstractVector)
	n = length(u)
	Δu = similar(u)
	h = 1.0 / (n - 1)

	if n == 1
		Δu[1] = 0.0  # Single point case
		return Δu
	end

	if n == 2
		Δu[1] = Δu[2] = (u[2] - u[1]) / h^2
		return Δu
	end

	# Standard case for n >= 3
	Δu[1] = (u[2] - 2u[1]) / h^2
	for i in 2:(n - 1)
		Δu[i] = (u[i - 1] - 2u[i] + u[i + 1]) / h^2
	end
	Δu[n] = (-2u[n] + u[n - 1]) / h^2

	return Δu
end

function verify_boundary_conditions_test(
	solution::AbstractVector, problem::AbstractKSProblem
)
	# Get indices for boundary points
	n_per_var = length(solution) ÷ problem.num_vars
	boundary_indices = 1:n_per_var  # Assuming boundaries are at the first points per variable

	# Check boundary conditions for each variable
	for var_idx in 1:(problem.num_vars)
		var_offset = (var_idx - 1) * n_per_var
		var_points = var_offset .+ boundary_indices

		# Check boundary conditions for this variable's points
		for bc in problem.boundary_conditions
			try
				if hasfield(typeof(bc), :boundary_value)
					for x in var_points
						expected = if bc.boundary_value isa Function
							bc.boundary_value(x)
						else
							bc.boundary_value
						end
						if !isapprox(solution[x], expected; rtol = sqrt(eps()))
							return false
						end
					end
				end
			catch
				return false
			end
		end
	end
	return true
end


"""
	create_test_matrix(n::Int, type::Union{Symbol,Bool} = false)

Create a test matrix of size n×n with specified characteristics.
- If type is Bool, creates a sparse (true) or dense (false) tridiagonal matrix
- If type is Symbol, creates a matrix with specific properties:
  :spd, :diag_dominant, :ill_conditioned, :sparse, :indefinite, :non_diag_dominant,
  :moderately_diag_dominant, :nearly_singular, :wide_spectrum, :clustered, :mixed_sign
"""
function create_test_matrix(n::Int, type::Union{Symbol,Bool} = false)
	if isa(type, Bool)
		if type # sparse case
			return spdiagm(-1 => ones(n - 1), 0 => 2.0 .* ones(n), 1 => ones(n - 1))
		else # dense case
			return Matrix(Tridiagonal(fill(-1.0, n-1), fill(2.0, n), fill(-1.0, n-1)))
		end
	end

	# Handle symbolic types
	try
		Q = qr(randn(n, n)).Q
		return case_matrix(Val(type), n, Q)
	catch e
		throw(ArgumentError("Failed to create matrix of type $type: $(e.msg)"))
	end
end

# Type-dispatch for different matrix types
case_matrix(::Val{:spd}, n, Q) = create_spd_matrix(n)
case_matrix(::Val{:diag_dominant}, n, Q) = make_diag_dominant_matrix(n)
case_matrix(::Val{:ill_conditioned}, n, Q) = gen_ill_conditioned(n)
case_matrix(::Val{:sparse}, n, Q) = create_sparse_matrix(n)
case_matrix(::Val{:indefinite}, n, Q) = create_symmetric_indefinite_matrix(n)
case_matrix(::Val{:non_diag_dominant}, n, Q) = make_non_diag_dominant_matrix(n)
case_matrix(::Val{:moderately_diag_dominant}, n, Q) = create_moderately_diag_dominant(n)
case_matrix(::Val{:nearly_singular}, n, Q) = Q * Diagonal(vcat([1e-10], range(1.0, 10.0; length=n-1))) * Q'
case_matrix(::Val{:wide_spectrum}, n, Q) = Q * Diagonal(exp.(range(0, 20; length=n))) * Q'
case_matrix(::Val{:clustered}, n, Q) = Q * Diagonal(vcat(ones(n ÷ 2), 1e4 * ones(n - n ÷ 2))) * Q'
case_matrix(::Val{:mixed_sign}, n, Q) = Q * Diagonal(vcat(-ones(n ÷ 2), ones(n - n ÷ 2))) * Q'
case_matrix(::Val{type}, n, Q) where {type} = throw(ArgumentError("Unknown matrix type: $type"))

function make_spd(A::AbstractMatrix{T}) where T
	return A + size(A, 1) * I
end

function gen_diag_dominant(n::Int, sparse::Bool = false)
	if sparse
		return spdiagm(-1 => -ones(n - 1), 0 => 4.0 .* ones(n), 1 => -ones(n - 1))
	else
		return Tridiagonal(fill(-1.0, n - 1), fill(4.0, n), fill(-1.0, n - 1))
	end
end

function gen_spd_matrix(n::Int, cond_num::Float64 = 1e3)
	Q = qr(randn(n, n)).Q
	Λ = Diagonal(range(1.0; stop = cond_num, length = n))
	A = Q * Λ * Q'
	# Force symmetry explicitly
	return (A + A') / 2
end
function gen_ill_conditioned(n::Int; cond_num::Float64 = 1e12)
	A = gen_spd_matrix(n, cond_num)
	return (A + A') / 2 + 1e-10 .* I
end

function estimate_condition_number_from_action(op::Function, n::Int; num_vectors::Int = 5)
	# Estimate highest singular value
	σ_max = 0.0
	for _ in 1:num_vectors
		v = normalize(randn(n))
		for _ in 1:10
			v = normalize(op(v))
			σ_max = max(σ_max, norm(op(v)))
		end
	end

	# Estimate lowest singular value using inverse iterations
	σ_min = Inf
	for _ in 1:num_vectors
		v = normalize(randn(n))
		for _ in 1:10
			w = op(v)
			σ = norm(w)
			σ_min = min(σ_min, σ)
			v = normalize(w)
		end
	end
	return σ_max / max(σ_min, eps(Float64))
end

function create_moderately_diag_dominant(n::Int)
	A = randn(n, n)
	# Calculate diagonal scaling that makes matrix moderately diagonally dominant
	row_sums = vec(sum(abs.(A); dims = 2))
	# Create diagonal matrix with 0.6 times the row sums
	D = Diagonal(0.6 .* row_sums)
	return D + A
end

function create_symmetric_indefinite_matrix(n::Int)
	A = randn(n, n)
	A_sym = (A + A') / 2
	# Ensure it's indefinite
	A_sym += LinearAlgebra.Diagonal((-1) .^ (1:n))
	return A_sym
end

function make_diag_dominant_matrix(n::Int, dominance_factor::Real = 2.0)
	A = randn(n, n)
	for i in 1:n
		row_sum = sum(abs, view(A, i, :)) - abs(A[i, i])
		A[i, i] = sign(A[i, i] + eps()) * (dominance_factor * row_sum + 1.0)  # Ensure strong diagonal dominance
	end
	return A
end

function make_ill_conditioned_sparse(n::Int)
	S = spzeros(n, n)

	for i in 1:n
		S[i, i] = 1.0 / (10.0^(i - 1))
	end

	for i in 1:(n - 1)
		S[i, i + 1] = 2.0 * S[i, i]
		S[i + 1, i] = S[i, i + 1]
	end

	for _ in 1:(n ÷ 10)
		i, j = rand(1:n), rand(1:n)
		if i != j
			S[i, j] = rand() * 1e-6
		end
	end

	return S
end

function make_non_diag_dominant_matrix(n::Int)
	A = zeros(n, n)
	for i in 1:n
		A[i, i] = 1.0  # Diagonal elements
		for j in 1:n
			if i != j
				A[i, j] = 1.0  # Off-diagonal elements sum to n-1 > 1
			end
		end
	end
	return A
end

function create_spd_matrix(n::Int; condition_number::Real = 10.0)
	if n == 1
		return fill(condition_number, 1, 1)  # Special case for 1x1 matrices
	end
	Q = qr(randn(n, n)).Q
	λ = range(1.0, condition_number; length = max(2, n))  # Ensure at least 2 points for range
	return Q * Diagonal(λ) * Q'
end

function create_sparse_matrix(n::Int; sparsity::Real = 0.1)
	nnz = round(Int, sparsity * n * n)
	rows = rand(1:n, nnz)
	cols = rand(1:n, nnz)
	vals = randn(nnz)

	# Ensure diagonal entries are nonzero and dominant
	A = sparse(rows, cols, vals, n, n)

	# Add diagonal dominance
	D = spdiagm(0 => vec(sum(abs, A; dims = 2)) .+ 1.0)

	return A + D
end

function create_sparse_matrix(n::Int, actual_sparsity::Float64)
	sparsity = min(max(actual_sparsity, 0.01), 0.99)
	nnz = round(Int, sparsity * n * n)
	rows = rand(1:n, nnz)
	cols = rand(1:n, nnz)
	vals = randn(nnz)

	# Ensure diagonal dominance
	A = sparse(rows, cols, vals, n, n)
	for i in 1:n
		A[i, i] = 2.0 * sum(abs.(A[i, :]))
	end

	return A
end

function get_coupling_block(A::AbstractMatrix, i::Int, j::Int,
	::KSCoupledProblem, mesh::KSMesh)
	n = get_total_dof(mesh)
	range_i = ((i - 1) * n + 1):(i * n)
	range_j = ((j - 1) * n + 1):(j * n)
	return view(A, range_i, range_j)
end

function create_compatible_test_mesh(prob_type::Symbol, dim::Int)
	# Always create mesh in Cartesian coordinates
	cart_domain = ntuple(i -> (-1.0, 1.0), dim)
	cart_coord_sys = KSCartesianCoordinates(cart_domain)

	# Create basic Cartesian mesh - adjust cell counts for 3D
	p_orders = ntuple(_ -> 3, dim)
	num_cells = ntuple(_ -> dim == 3 ? 1 : 2, dim)  # Fewer cells in 3D
	mesh = SpectralMethods.create_ocfc_mesh(
		cart_coord_sys,
		num_cells,
		p_orders,
		ntuple(_ -> 1, dim),  # Reduced continuity order for stability
		dim,
		1,  # Reduced max level for 3D
	)

	# For problems that need different coordinate systems
	if prob_type == :moving_boundary
		# Target coordinate system based on dimension
		target_coord_sys = if dim == 2
			KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
		elseif dim == 3
			KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))
		else
			cart_coord_sys  # Fallback for 1D
		end

		domain_map = DomainMapping(;
			forward_transform = NonlinearTransform(;
				forward_map = x -> from_cartesian(x, target_coord_sys),
				inverse_map = x -> to_cartesian(x, target_coord_sys),
			),
			inverse_transform = NonlinearTransform(;
				forward_map = x -> to_cartesian(x, target_coord_sys),
				inverse_map = x -> from_cartesian(x, target_coord_sys),
			),
			physical_system = target_coord_sys,
			fictitious_system = cart_coord_sys,
		)

		# Store forward transform only
		mesh.transformation_data = domain_map.forward_transform

		return mesh, target_coord_sys
	end

	return mesh, cart_coord_sys
end

# Create copy of all field values
"""
	modify_problem(problem::T, field::Symbol, value::Any) where T <: AbstractKSProblem

Create a modified copy of a problem, preserving type information and field values.
"""
function modify_problem(problem::T, field::Symbol, value::Any) where T <: AbstractKSProblem
    # Validate input for specific fields
    if field == :num_vars && value <= 0
        throw(ArgumentError("num_vars must be positive"))
    end

    if field == :boundary_conditions
        if isnothing(value) || (value isa Vector && isempty(value))
            throw(ArgumentError("boundary_conditions cannot be null or empty"))
        end
        if value isa Vector && !all(x -> x isa AbstractKSBoundaryCondition, value)
            throw(ArgumentError("All boundary conditions must be of type AbstractKSBoundaryCondition"))
        end
    end

    if field == :tspan && !isnothing(value)
        if !(value isa Tuple) || length(value) != 2
            throw(ArgumentError("tspan must be nothing or a 2-tuple"))
        end
        if value[2] <= value[1]
            throw(ArgumentError("End time must be greater than start time"))
        end
    end

    # Get all field values, updating the specified field
    kwargs = Dict{Symbol,Any}()
    for name in fieldnames(T)
        kwargs[name] = name == field ? value : getfield(problem, name)
    end

    # Try construction with error handling
    try
        return T(; kwargs...)
    catch e
        @warn "Problem modification failed" problem_type=T field=field value=value exception=e
        rethrow(e)
    end
end

function get_physical_mesh_dof(mesh::KSMesh{T, N})::Int where {T, N}
	physical_nodes = Set{Int}()
	for cell_idx in mesh.physical_cells
		cell = mesh.cells[cell_idx]
		try
			std_cell = SpectralMethods.get_or_create_standard_cell(
				cell.standard_cell_key[1],
				cell.standard_cell_key[2],
			)
			for (local_idx, global_idx) in cell.node_map
				if all(
					i -> 1 <= local_idx[i] <= length(std_cell.nodes_with_boundary[i]), 1:N
				)
					coords = ntuple(i -> std_cell.nodes_with_boundary[i][local_idx[i]], N)
					if mesh.physical_domain(coords)
						push!(physical_nodes, global_idx)
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

# Update validate_temporal_dofs to throw error for invalid cases
function validate_temporal_dofs(problem::AbstractKSProblem, mesh::KSMesh)
    if !hasfield(typeof(problem), :tspan)
        return true # Non-temporal problems are valid
    end

    # Skip validation for BVPs
    if problem isa KSBVPProblem
        return true
    end

    # Always validate tspan for time-dependent problems
    if !isnothing(problem.tspan)
        if problem.tspan[2] <= problem.tspan[1]
            throw(ArgumentError("End time must be greater than start time"))
        end

        if !all(isfinite, problem.tspan)
            throw(ArgumentError("Time span must contain finite values"))
        end
    end

    return true
end

# Update are_coordinate_systems_compatible to handle different coordinate systems
"""Compare domains with proper element-wise comparison"""
function are_coordinate_systems_compatible(sys1::AbstractKSCoordinateSystem,
    sys2::AbstractKSCoordinateSystem)

    # Check dimensions first
    if length(get_coordinate_domains(sys1)) != length(get_coordinate_domains(sys2))
        return false
    end

    # Get domains
    d1 = get_coordinate_domains(sys1)
    d2 = get_coordinate_domains(sys2)

    # For different types, check domain coverage
    if typeof(sys1) != typeof(sys2)
        return all(check_domain_overlap_pair.(d1, d2))
    end

    # Same type - check exact domain match
    return all(compare_domain_ranges.(d1, d2))
end

"""Compare individual domain bounds with tolerance"""
function compare_domain_ranges(r1::Tuple{<:Real,<:Real}, r2::Tuple{<:Real,<:Real})
    return abs(r1[1] - r2[1]) < sqrt(eps(promote_type(eltype(r1),eltype(r2)))) &&
           abs(r1[2] - r2[2]) < sqrt(eps(promote_type(eltype(r1),eltype(r2))))
end

"""Check if two ranges overlap"""
function check_domain_overlap_pair(r1::Tuple{<:Real,<:Real}, r2::Tuple{<:Real,<:Real})
    r1_min, r1_max = r1
    r2_min, r2_max = r2

    # Check for overlap accounting for numerical tolerance
    tol = sqrt(eps(promote_type(eltype(r1),eltype(r2))))
    return (r1_max + tol >= r2_min) && (r2_max + tol >= r1_min)
end

"""Get the coordinate domains from a given coordinate system."""
function get_coordinate_domains(coord_sys::AbstractKSCoordinateSystem)
    if hasfield(typeof(coord_sys), :ranges)
        return coord_sys.ranges
    elseif hasfield(typeof(coord_sys), :domain)
        # For 1D systems
        return [coord_sys.domain]
    elseif hasfield(typeof(coord_sys), :domains)
        return coord_sys.domains
    else
        throw(ArgumentError("Cannot extract domains from $(typeof(coord_sys))"))
    end
end
