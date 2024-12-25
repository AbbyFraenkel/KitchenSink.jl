using LinearAlgebra, StaticArrays, FastGaussQuadrature
using KitchenSink.SpectralMethods, KitchenSink.KSTypes, KitchenSink.CoordinateSystems
using KitchenSink.BoundaryConditions, KitchenSink.Transforms
using KitchenSink.CacheManagement, KitchenSink.NumericUtilities
using BSplineKit: BSplineBasis, BSplineOrder, KnotVector, evaluate
using BSplineKit: BSplineKit

# Constants for testing
TEST_TOLERANCES = (1e-8, 1e-6, 1e-4)
TEST_POLYNOMIAL_ORDERS = (3, 5, 8, 13, 21, 34)
TEST_DIMENSIONS = (1, 2, 3, 5)
TEST_CONTINUITY_ORDERS = (1, 2, 3)
TEST_REFINEMENT_LEVELS = (1, 2, 3, 4)
test_problem_types = [:ode, :dae]
# Univariate and multivariate test functions with known derivatives

# Univariate
test_f(x) = sin(π * x)
test_df(x) = π * cos(π * x)
test_d2f(x) = -π^2 * sin(π * x)
test_d3f(x) = -π^3 * cos(π * x)

# Multivariate
test_h(x, y) = sin(π * x) * cos(π * y)
test_dhx(x, y) = π * cos(π * x) * cos(π * y)
test_dhy(x, y) = -π * sin(π * x) * sin(π * y)
test_d2hx(x, y) = -π^2 * sin(π * x) * cos(π * y)
test_d2hy(x, y) = -π^2 * sin(π * x) * cos(π * y)
test_d3hx(x, y) = -π^3 * cos(π * x) * cos(π * y)

function analytical_integral(dim::Int)
	return (2.0 / π)^dim
end

function analytical_integral_h(a::Real, b::Real, c::Real, d::Real)
	return (cos(π * a) - cos(π * b)) * (sin(π * d) - sin(π * c)) / π^2
end

function analytical_integral_k(a::Real, b::Real, c::Real, d::Real, e::Real, f::Real)
	return (cos(π * a) - cos(π * b)) * (sin(π * d) - sin(π * c)) * (exp(f) - exp(e)) / π^2
end

function analytical_integral_multidim(limits::NTuple{N, Tuple{Float64, Float64}}) where {N}
	return prod(map(l -> (cos(π * l[1]) - cos(π * l[2])) / π, limits))
end

function generate_test_coordinate_system(dim::Int)
	return KSCartesianCoordinates(ntuple(_ -> (0.0, 1.0), dim))
end

function create_test_coordinate_system_type(dim::Int)
	if dim == 1
		return KSCartesianCoordinates(((0.0, 1.0),))
	elseif dim == 2
		return KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
	elseif dim == 3
		return KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))
	else
		return KSCartesianCoordinates(ntuple(i -> (0.0, 1.0), dim))
	end
end
function generate_nd_grid(n::Int, points_per_dim::Int)
	ranges = [range(-1, 1; length = points_per_dim) for _ in 1:n]
	return collect(Iterators.product(ranges...))
end

function test_multidim(x...)
	return prod(sin(π * xi) for xi in x)
end
function expected_values(p::Int)
	nodes, weights = gausslegendre(p)
	nodes_with_boundary = vcat(-1.0, nodes, 1.0)
	weights_with_boundary = vcat(0.0, weights, 0.0)
	return nodes_with_boundary, weights_with_boundary, nodes, weights
end
# Mesh creation with validation
function create_test_mesh(
	dim::Int,
	num_cells::NTuple{N, Int} = ntuple(_ -> 2, dim),
	p_vec::NTuple{N, Int} = ntuple(_ -> 5, dim),
	continuity_order::NTuple{N, Int} = ntuple(_ -> 1, dim),
	max_level::Int = 1) where {N}
	# Validation for dimensionality and input consistency
	@assert dim > 0 "Dimension must be greater than 0"
	@assert length(num_cells) == dim "num_cells must match the dimension"
	@assert length(p_vec) == dim "p_vec must match the dimension"
	@assert length(continuity_order) == dim "continuity_order must match the dimension"
	@assert max_level >= 1 "max_level must be 1 or greater"

	coord_sys = generate_test_coordinate_system(dim)
	mesh = SpectralMethods.create_ocfc_mesh(
		coord_sys,
		num_cells,
		p_vec,
		continuity_order,
		dim,
		max_level,
	)
	return mesh, coord_sys
end

# Boundary condition generator to reuse across various problem types
function create_boundary_condition(bc_type::Type{<:AbstractKSBoundaryCondition}, coord_sys)
	if bc_type == KSRobinBC
		return KSRobinBC(;
			neumann_coefficient = x -> 1.0,
			dirichlet_coefficient = x -> 2.0,
			boundary_value = x -> sum(x),
			boundary_region = x -> true,
			coordinate_system = coord_sys,
		)
	elseif bc_type == KSNeumannBC
		return KSNeumannBC(;
			flux_value = x -> sum(x),
			boundary_region = x -> true,
			coordinate_system = coord_sys,
		)
	else
		return KSDirichletBC(;
			boundary_value = x -> sum(x),
			boundary_region = x -> true,
			coordinate_system = coord_sys,
		)
	end
end

# Create general test problems with optional custom boundary conditions
function create_test_problem(dim::Int, bc_type::Type{<:AbstractKSBoundaryCondition})
	mesh, coord_sys = create_test_mesh(dim)
	bc = create_boundary_condition(bc_type, coord_sys)

	return KSProblem(;
		equation = (x, t, u, ∇u, Δu) -> -u,
		boundary_conditions = [bc],
		domain = coord_sys.domains,
		coordinate_system = coord_sys,
		initial_conditions = [1.0],
		tspan = (0.0, 1.0),
		continuity_order = 1,
	)
end
# Generalized problem creation for various types: ODE, DAE, IDE, PDE, BVP, etc.
function create_problem_of_type(problem_type::Symbol, dim::Int)
	mesh, coord_sys = create_test_mesh(dim)

	if problem_type == :ode
		return KSODEProblem(;
			ode = (t, y, var) -> (-y, 0.0),
			boundary_conditions = [
				KSDirichletBC(;
					boundary_value = x -> 1.0,
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				),
			],
			domain = coord_sys.domains,
			initial_conditions = [1.0],
			tspan = (0.0, 1.0),
			coordinate_system = coord_sys,
			num_vars = 1,
		)
	elseif problem_type == :dae
		return KSDAEProblem(;
			dae = (t, y, dy, var) -> (dy[1] - y[2], y[1]^2 + y[2]^2 - 1),
			boundary_conditions = [
				KSDirichletBC(;
					boundary_value = x -> 0.0,
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				),
			],
			domain = coord_sys.domains,
			initial_conditions = [1.0, 0.0],
			tspan = (0.0, 1.0),
			coordinate_system = coord_sys,
			num_vars = 2,
			num_algebraic_vars = 1,
		)
		# Add additional problem types as needed (e.g., :pide, :ide, etc.)
	end
end

# Functions for testing domain transformations
function create_test_transform(dim::Int)
	return Transforms.NonlinearTransform(;
		forward_map = (x) -> test_transform_nd(x),
		inverse_map = (y) -> test_inverse_transform_nd(y),
		jacobian = (x) -> test_jacobian_nd(x),
	)
end

function create_test_affine_transform(dim::Int)
	matrix = randn(dim, dim)
	translation = randn(dim)
	return AffineTransform(matrix, translation)
end
# Transformation and Jacobian functions
function test_transform_nd(x::Union{AbstractVector, Tuple})
	n = length(x)
	transformed = zeros(n)
	for i in 1:n
		transformed[i] = x[i]^2 + 0.5 * sum(x[j]^2 for j in 1:i)
	end
	return transformed
end
function test_inverse_transform_nd(y::Union{AbstractVector, Tuple})
	n = length(y)
	x = zeros(n)
	for i in 1:n
		if i == 1
			x[i] = sqrt(y[i])
		else
			x[i] = sqrt(y[i] - 0.5 * sum(x[j]^2 for j in 1:(i - 1)))
		end
	end
	return x
end

function test_jacobian_nd(x::AbstractVector)
	n = length(x)
	J = zeros(n, n)
	for i in 1:n
		J[i, i] = 2 * x[i]
		for j in 1:(i - 1)
			J[i, j] = x[j]
		end
	end
	return J
end

# Test function evaluation (summation of sine function over multiple dimensions)
function evaluate_test_function(point::AbstractVector{T}) where {T <: Real}
	return sum(sin.(π * point))
end

# Sparse and SPD matrix generation functions for testing solvers
function create_spd_matrix(n::Int)
	A = randn(n, n)
	return A' * A + n * I
end

function create_sparse_matrix(n::Int, density::Float64)
	return sprandn(n, n, density)
end
# Helper function to create test cell
# Add to the top of test file, after existing utilities
function verify_node_coordinates(
	coords::NTuple{N, T}, cell::KSCell{T, N}, mesh::KSMesh{T, N}
) where {T, N}
	# Check if coordinates are within cell bounds
	cell_corners = get_cell_corners(cell)
	min_bounds = minimum(cell_corners)
	max_bounds = maximum(cell_corners)
	return all(min_bounds .<= coords .<= max_bounds)
end

function verify_node_coordinates(
	coords::SVector{N, T}, cell::KSCell{T, N}, mesh::KSMesh{T, N}
) where {T, N}
	cell_corners = get_cell_corners(cell)
	min_bounds = minimum(cell_corners)
	max_bounds = maximum(cell_corners)
	return all(min_bounds .<= coords .<= max_bounds)
end

function test_interpolation_function(x...)
	return sum(sin.(x))
end
