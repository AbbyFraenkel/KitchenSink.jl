module SpectralMethods

using LinearAlgebra, SparseArrays, FastGaussQuadrature
using ..KSTypes, ..CoordinateSystems
# Exported Functions
export create_legendre_nodes_and_weights, gauss_legendre_with_boundary_nd, barycentric_weights, Lagrange_polynomials
export barycentric_interpolation, interpolate_nd, derivative_matrix!, kth_derivative_matrix!
export derivative_matrix_nd, enforce_ck_continuity!, create_basis_functions, is_near_zero
export scale_derivative_matrix, create_differentiation_matrices, quadrature_matrix!, compute_integral
export quadrature_matrix_nd, compute_integral_nd, scale_quadrature_matrix!, create_ocfe_mesh
export map_nodes_to_physical_domain, get_coordinate_range
"""
	create_legendre_nodes_and_weights(p::Integer)

Create Legendre nodes and weights for the interval [-1, 1].

Returns:
- `Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`:
  Nodes with boundary, weights with boundary, interior nodes, interior weights.
"""
function create_legendre_nodes_and_weights(p::Integer)
	p >= 3 || throw(ArgumentError("Polynomial degree p must be at least 3."))

	x, w = gausslegendre(p - 2)
	x_with_boundary = vcat(-1.0, x, 1.0)
	w_with_boundary = vcat(0.0, w, 0.0)

	return x_with_boundary, w_with_boundary, x, w
end

"""
	gauss_legendre_with_boundary_nd(p::Integer, dim::Integer)

Generate Gauss-Legendre nodes and weights in multiple dimensions with boundary points included.

# Arguments
- `p::Integer`: The number of points in each dimension (including boundary points).
- `dim::Integer`: The number of dimensions.

# Returns
- `Tuple{Vector{Vector{Float64}}, Vector{Float64}, Vector{Vector{Float64}}, Vector{Float64}}`:
  Nodes with boundary, weights with boundary, interior nodes, interior weights.
"""
function gauss_legendre_with_boundary_nd(p::Integer, dim::Integer)
	p >= 3 || throw(ArgumentError("Polynomial degree p must be at least 3."))
	dim >= 1 || throw(ArgumentError("Number of dimensions must be at least 1."))

	nodes_1d, weights_1d, nodes_interior_1d, weights_interior_1d = create_legendre_nodes_and_weights(p)

	nodes = [collect(coords) for coords in Iterators.product(fill(nodes_1d, dim)...)]
	weights = [prod(weights_1d[i] for i in indices) for indices in Iterators.product(fill(1:p, dim)...)]

	nodes_interior = [collect(coords) for coords in Iterators.product(fill(nodes_interior_1d, dim)...)]
	weights_interior = [prod(weights_interior_1d[i] for i in indices) for indices in Iterators.product(fill(1:(p - 2), dim)...)]

	nodes = vec(nodes)
	weights = vec(weights)
	nodes_interior = vec(nodes_interior)
	weights_interior = vec(weights_interior)
	return nodes, weights, nodes_interior, weights_interior
end

"""
	barycentric_weights(points::AbstractVector{<:Real})

Compute the barycentric weights for a set of nodes.

# Arguments
- `points::AbstractVector{<:Real}`: A vector of nodes.

# Returns
- `Vector{Float64}`: The barycentric weights corresponding to the nodes.
"""
function barycentric_weights(points::AbstractVector{<:Real})
	n = length(points)
	n >= 3 || throw(ArgumentError("The length of nodes must be at least 3."))

	w = ones(eltype(points), n)
	for j in 1:n
		for k in 1:(j - 1)
			diff = points[j] - points[k]
			if is_near_zero(diff)
				w[j] = w[k]
				break
			else
				w[j] /= diff
				w[k] /= -diff
			end
		end
	end
	return w
end

"""
	Lagrange_polynomials(points::AbstractVector{<:Real})

Construct Lagrange polynomials based on the given points.

# Arguments
- `points::AbstractVector{<:Real}`: A vector of nodes.

# Returns
- `SparseMatrixCSC{Float64, Int}`: The sparse matrix representing the Lagrange polynomials.
"""
function Lagrange_polynomials(points::AbstractVector{<:Real})
	n = length(points)
	n >= 3 || throw(ArgumentError("The length of nodes must be at least 3."))
	return sparse(1:n, 1:n, ones(Float64, n))
end

"""
	barycentric_interpolation(points::AbstractVector{<:Real}, values::AbstractVector{<:Real}, x::Real)

Perform barycentric interpolation at a given point.

# Arguments
- `points::AbstractVector{<:Real}`: A vector of nodes.
- `values::AbstractVector{<:Real}`: A vector of values corresponding to the nodes.
- `x::Real`: The point at which to interpolate.

# Returns
- `Float64`: The interpolated value.
"""
function barycentric_interpolation(points::AbstractVector{<:Real}, values::AbstractVector{<:Real}, x::Real)
	n = length(points)
	n == length(values) || throw(ArgumentError("points and values must have the same length"))
	n >= 3 || throw(ArgumentError("At least 3 points are required for interpolation"))

	weights = barycentric_weights(points)
	numerator = zero(eltype(values))
	denominator = zero(eltype(points))

	for j in 1:n
		diff = x - points[j]
		if iszero(diff)
			return values[j]
		end
		t = weights[j] / diff
		numerator += t * values[j]
		denominator += t
	end

	return numerator / denominator
end

"""
	interpolate_nd(nodes::AbstractVector{<:AbstractVector{<:Real}}, values::AbstractArray, point::AbstractVector{<:Real})

Perform N-dimensional barycentric interpolation at a given point.

# Arguments
- `nodes::AbstractVector{<:AbstractVector{<:Real}}`: A vector of vectors, each representing nodes in a particular dimension.
- `values::AbstractArray`: The values corresponding to the nodes.
- `point::AbstractVector{<:Real}`: The point at which to interpolate.

# Returns
- `Float64`: The interpolated value.
"""
function interpolate_nd(nodes::AbstractVector{<:AbstractVector{<:Real}}, values::AbstractArray, point::AbstractVector{<:Real})
	dim = length(nodes)
	length(point) == dim || throw(ArgumentError("The number of dimensions in nodes must match the point dimensions."))

	function interpolate_recursive(current_dim::Int, indices::NTuple{N, Int}) where N
		if current_dim > dim
			return values[indices...]
		end
		interp_vals = [interpolate_recursive(current_dim + 1, ntuple(k -> k == current_dim ? i : indices[k], dim)) for i in 1:length(nodes[current_dim])]
		return barycentric_interpolation(nodes[current_dim], interp_vals, point[current_dim])
	end

	return interpolate_recursive(1, ntuple(_ -> 1, dim))
end

"""
	derivative_matrix!(points::AbstractVector{<:Real})

Construct the first derivative matrix for a set of nodes.

# Arguments
- `points::AbstractVector{<:Real}`: A vector of nodes.

# Returns
- `SparseMatrixCSC{Float64, Int}`: The sparse first derivative matrix.
"""
function derivative_matrix!(points::AbstractVector{<:Real})
	n = length(points)
	n >= 3 || throw(ArgumentError("The length of points must be at least 3."))

	D = spzeros(eltype(points), n, n)
	w = barycentric_weights(points)

	for i in 1:n
		for j in 1:n
			if i != j
				diff = points[i] - points[j]
				if !is_near_zero(diff)
					D[i, j] = w[j] / (w[i] * diff)
				end
			end
		end
		D[i, i] = -sum(D[i, k] for k in 1:n if k != i)
	end

	return D
end

"""
	kth_derivative_matrix!(points::AbstractVector{<:Real}, k_max::Integer)

Compute the k-th derivative matrices for a set of nodes.

# Arguments
- `points::AbstractVector{<:Real}`: A vector of nodes.
- `k_max::Integer`: The maximum order of the derivative.

# Returns
- `Vector{SparseMatrixCSC{Float64, Int}}`: A vector of derivative matrices up to the k-th order.
"""
function kth_derivative_matrix!(points::AbstractVector{<:Real}, k_max::Integer)
	k_max > 0 || throw(ArgumentError("k_max must be positive."))
	n = length(points)
	n >= 3 || throw(ArgumentError("The length of points must be at least 3."))

	T = eltype(points)
	D_matrices = [spzeros(T, n, n) for _ in 1:k_max]
	weights = barycentric_weights(points)
	D_prev = derivative_matrix!(points)
	D_matrices[1] = D_prev

	diff_recip = [i != j ? inv(points[i] - points[j]) : zero(T) for i in 1:n, j in 1:n]

	for k in 2:k_max
		D_k = D_matrices[k]
		for i in 1:n
			for j in 1:n
				if i != j
					D_k[i, j] = (k * diff_recip[i, j]) * (weights[j] / weights[i] * D_prev[i, i] - D_prev[i, j])
				end
			end
			D_k[i, i] = -sum(D_k[i, :])
		end
		D_prev = D_k
	end

	return D_matrices
end

"""
	derivative_matrix_nd(points::AbstractVector{<:AbstractVector{<:Real}}, dim::Integer, order::Integer = 1)

Construct the derivative matrix for each dimension independently, for an N-dimensional grid of points.

# Arguments
- `points::AbstractVector{<:AbstractVector{<:Real}}`: A vector of vectors representing the nodes in each dimension.
- `dim::Integer`: The number of dimensions.
- `order::Integer`: The order of the derivative (default is 1).

# Returns
- `Vector{SparseMatrixCSC{Float64, Int}}`: A vector of derivative matrices, one for each dimension.
"""
function derivative_matrix_nd(points::AbstractVector{<:AbstractVector{<:Real}}, dim::Integer, order::Integer = 1)
	dim == length(points) || throw(ArgumentError("The specified dimension dim must match the number of dimensions in the points."))
	order >= 1 || throw(ArgumentError("The derivative order must be at least 1."))

	# Initialize a vector to store derivative matrices
	T = eltype(points[1])
	derivative_matrices = Vector{SparseMatrixCSC{T, Int}}(undef, dim)

	# Compute derivative matrices for each dimension
	for d in 1:dim
		nodes_d = points[d]
		derivative_matrices[d] = kth_derivative_matrix!(nodes_d, order)[order]
	end

	return derivative_matrices
end

"""
	enforce_ck_continuity!(D::AbstractSparseMatrix, k::Integer)

Enforce C^k continuity on a derivative matrix by modifying its first and last `k` rows in place.

# Arguments
- `D::AbstractSparseMatrix`: The derivative matrix to modify.
- `k::Integer`: The order of continuity to enforce.

# Returns
- `AbstractSparseMatrix`: The modified derivative matrix.
"""
function enforce_ck_continuity!(D::AbstractMatrix, k::Integer)
	n = size(D, 1)
	0 <= k < n || throw(ArgumentError("Continuity order k must be between 0 and n-1."))

	for i in 1:k
		D[i, :] .= 0.0
		D[i, i] = 1.0
	end

	for i in (n - k + 1):n
		D[i, :] .= 0.0
		D[i, i] = 1.0
	end

	return D
end

"""
	is_near_zero(x::Number)

Check if a value is close to zero within machine precision.

# Arguments
- `x::Number`: The value to check.

# Returns
- `Bool`: True if the value is near zero, false otherwise.
"""
function is_near_zero(x::Number)
	return abs(x) < eps(typeof(x))
end

"""
	create_basis_functions(polynomial_degree::Integer, dim::Integer)

Create basis functions for the given polynomial degree and dimension.

# Arguments
- `polynomial_degree::Integer`: The polynomial degree for the basis functions.
- `dim::Integer`: The number of dimensions.

# Returns
- `Vector{KSBasisFunction}`: A vector of basis functions.
"""
function create_basis_functions(polynomial_degree::Integer, dim::Integer)
	polynomial_degree >= 1 && dim >= 1 || throw(ArgumentError("polynomial_degree and dim must be at least 1"))

	num_basis_functions = binomial(polynomial_degree + dim, dim)
	basis_functions = Vector{KSBasisFunction}(undef, num_basis_functions)
	basis_index = 1

	nodes = collect(0:polynomial_degree)
	lagrange_basis = [Lagrange_polynomials(nodes) for _ in 1:dim]

	for multi_index in Iterators.product(ntuple(_ -> 0:polynomial_degree, dim)...)
		if sum(multi_index) <= polynomial_degree
			basis_functions[basis_index] = KSBasisFunction(basis_index,
														   x -> prod(lagrange_basis[i][multi_index[i] + 1, 1] for i in 1:dim))
			basis_index += 1
		end
	end

	return basis_functions
end

"""
	scale_derivative_matrix(D::AbstractSparseMatrix, scale_factor::Number)

Scale the derivative matrix by a given factor.

# Arguments
- `D::AbstractSparseMatrix`: The derivative matrix to scale.
- `scale_factor::Number`: The scaling factor.

# Returns
- `AbstractSparseMatrix`: The scaled derivative matrix.
"""
function scale_derivative_matrix(D::AbstractSparseMatrix, scale_factor::Number)
	return D * scale_factor
end

"""
	create_differentiation_matrices(points::AbstractVector{<:Real}, dim::Integer)

Create differentiation matrices for the given points in the specified number of dimensions.

# Arguments
- `points::AbstractVector{<:Real}`: The points at which to create the differentiation matrices.
- `dim::Integer`: The number of dimensions.

# Returns
- `Vector{SparseMatrixCSC}`: A vector of differentiation matrices, one for each dimension.
"""
function create_differentiation_matrices(points::AbstractVector{<:Real}, dim::Integer)
	n = length(points)
	n >= 3 || throw(ArgumentError("The number of points must be at least 3."))
	dim >= 1 || throw(ArgumentError("The number of dimensions must be at least 1."))

	diff_matrices = Vector{SparseMatrixCSC{eltype(points), Int}}(undef, dim)

	for d in 1:dim
		diff_matrices[d] = derivative_matrix!(points)
	end

	return diff_matrices
end

"""
	quadrature_matrix!(points::AbstractVector{<:Real}, weights::AbstractVector{<:Real})

Constructs a quadrature matrix for numerical integration using specified points and weights.

# Arguments
- `points::AbstractVector{<:Real}`: A vector of nodes where the function is evaluated.
- `weights::AbstractVector{<:Real}`: A vector of quadrature weights associated with the nodes.

# Returns
- `SparseMatrixCSC{Float64, Int}`: The quadrature matrix that applies weights to function values for integration.
"""

function quadrature_matrix!(weights::AbstractVector{<:Real})
	n = length(weights)
	Q = spzeros(n, n)
	for i in 1:n
		Q[i, i] = weights[i]
	end
	return Q
end

"""
	compute_integral(values::AbstractVector{<:Real}, quad_matrix::AbstractMatrix)

Computes the integral of function values using the quadrature matrix.

# Arguments
- `values::AbstractVector{<:Real}`: A vector of function values evaluated at the nodes.
- `quad_matrix::AbstractMatrix`: The quadrature matrix that weights the function values for integration.

# Returns
- `Float64`: The computed integral value.
"""
function compute_integral(weights::AbstractVector{<:Real}, values::AbstractVector{<:Real})
	n = length(values)
	n == size(weights, 1) || throw(ArgumentError("The number of values must match the size of the quadrature matrix."))

	return dot(weights, values)
end

"""
	quadrature_matrix_nd(weights_input::Union{AbstractVector, Tuple}, dim::Int)

Constructs a multi-dimensional quadrature matrix using tensor products of 1D quadrature matrices.

# Arguments
- `weights_input::Union{AbstractVector, Tuple}`: A vector or tuple of vectors representing weights for each dimension.
- `dim::Int`: The number of dimensions.

# Returns
- `SparseMatrixCSC{Float64, Int}`: The multi-dimensional quadrature matrix.
"""
function quadrature_matrix_nd(weights_input::Union{AbstractVector, Tuple}, dim::Int)
	# Convert tuples to vectors to handle different input types uniformly
	weights_input = isa(weights_input, Tuple) ? collect(weights_input) : weights_input

	# Ensure the input dimensions match the expected number
	dim == length(weights_input) || throw(ArgumentError("Dimensions of weights must match."))

	# Extract weights as vectors, converting tuples if needed
	weights = [isa(weights_input[d], Tuple) ? collect(weights_input[d]) : weights_input[d] for d in 1:dim]

	# Create 1D quadrature matrices for each dimension
	Q_matrices = [quadrature_matrix!(weights[d]) for d in 1:dim]

	# Use Kronecker product to create the multi-dimensional quadrature matrix
	Q_nd = Q_matrices[1]
	for i in 2:dim
		Q_nd = kron(Q_nd, Q_matrices[i])  # Tensor product to combine matrices
	end

	return Q_nd
end

"""
	compute_integral_nd(values::AbstractVector{<:Real}, quad_matrix::AbstractMatrix)

Computes the multi-dimensional integral of function values using the quadrature matrix.

# Arguments
- `values::AbstractVector{<:Real}`: A vector of function values on a multi-dimensional grid.
- `quad_matrix::AbstractMatrix`: The multi-dimensional quadrature matrix.

# Returns
- `Float64`: The computed integral value.
"""
function compute_integral_nd(values::AbstractVector{<:Real}, quad_matrix::AbstractMatrix)
	n = length(values)
	n == size(quad_matrix, 1) || throw(ArgumentError("The number of values must match the size of the quadrature matrix."))

	return sum(quad_matrix * values)
end

"""
	create_quadrature_matrices(weights::AbstractVector{<:Real}, dim::Integer)

Create quadrature matrices for numerical integration in multiple dimensions.

# Arguments
- `weights::AbstractVector{<:Real}`: A vector of weights for the quadrature points.
- `dim::Integer`: The number of dimensions.

# Returns
An array of quadrature matrices, where each matrix represents the quadrature points for a specific dimension.

"""
function create_quadrature_matrices(weights::AbstractVector{<:Real}, dim::Integer)
	n = length(weights)
	n >= 3 || throw(ArgumentError("The number of points must be at least 3."))
	dim >= 1 || throw(ArgumentError("The number of dimensions must be at least 1."))

	quadrature_matrices = Vector{SparseMatrixCSC{eltype(weights), Int}}(undef, dim)

	for d in 1:dim
		quadrature_matrices[d] = quadrature_matrix!(weights)
	end

	return quadrature_matrices
end

"""
	scale_quadrature_matrix!(Q::AbstractSparseMatrix, scale_factor::Number)

Scales the quadrature matrix by a given factor.

# Arguments
- `Q::AbstractSparseMatrix`: The quadrature matrix to scale.
- `scale_factor::Number`: The scaling factor.

# Returns
- `AbstractSparseMatrix`: The scaled quadrature matrix.
"""
function scale_quadrature_matrix!(Q::AbstractSparseMatrix, scale_factor::Number)
	return Q * scale_factor
end

"""
	create_ocfe_mesh(coord_system::AbstractKSCoordinateSystem,
					 num_elements::AbstractVector{<:Integer},
					 poly_degree::Union{Integer, AbstractVector{<:Integer}},
					 continuity_order::Union{Integer, AbstractVector{<:Integer}},
					 dims::Int) -> KSMesh{T, N} where {T, N}

Create an Orthogonal Collocation on Finite Elements (OCFE) mesh for a given coordinate system.

# Arguments
- `coord_system::AbstractKSCoordinateSystem`: The coordinate system used for the mesh, which can be Cartesian, Polar, Cylindrical, or Spherical.
- `num_elements::AbstractVector{<:Integer}`: A vector specifying the number of elements in each dimension.
- `poly_degree::Union{Integer, AbstractVector{<:Integer}}`: The polynomial degree used for each element. It can be a single integer (uniform degree across dimensions) or a vector of integers specifying the degree for each dimension.
- `continuity_order::Union{Integer, AbstractVector{<:Integer}}`: Specifies the continuity order of the basis functions between elements. It can be a single integer or a vector of integers for each dimension.
- `dims::Int`: The number of dimensions in the problem.

# Returns
- `KSMesh{T, N}`: A mesh of type `KSMesh`, which includes the elements, tensor product masks, and location matrices for the problem's domain.

# Description
This function generates a mesh based on the specified coordinate system, polynomial degree, and number of elements in each dimension. It is designed to work with various coordinate systems such as Cartesian, Polar, Cylindrical, and Spherical coordinates. The function proceeds as follows:

1. **Determine the Element Type (`T`)**:
	- The type `T` is inferred from the range values of the coordinate system. This ensures compatibility with different coordinate systems.

2. **Initialize Structures**:
	- Various structures are prepared, including `global_nodes` (mapping unique nodes to global indices), `global_node_coords` (storing unique node coordinates), `elements` (a vector of mesh elements), `location_matrices` (storing the mapping of local to global node indices for each element), and `tensor_product_masks` (indicating tensor product structure of basis functions).

3. **Populate the Mesh**:
	- The function iterates over each element in the domain, creating nodes, differentiation matrices, quadrature matrices, and mapping nodes to their global indices. It also builds the location matrices and tensor product masks for each element.

4. **Return the Mesh**:
	- The function returns a `KSMesh` object containing the generated elements, tensor product masks, and location matrices.

# Example
```julia
# Example usage with Cartesian coordinates
cartesian_coords = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
num_elements = [4, 4, 4]
poly_degree = 3
continuity_order = 2
dims = 3

mesh = create_ocfe_mesh(cartesian_coords, num_elements, poly_degree, continuity_order, dims)
```

# Notes
- The function handles different coordinate systems by extracting the appropriate range values for each dimension.
- The `tensor_product_masks` are created as `BitArray` objects with dimensions determined by the polynomial degrees, which ensures the correct tensor product structure for the basis functions.
- This function is particularly useful for creating meshes in high-dimensional spaces where the problem domain is described by a non-Cartesian coordinate system.

# Errors
- Throws an `ArgumentError` if the coordinate system or dimension index is unsupported, or if there is an inconsistency in the dimensions of the input arguments.

"""

function create_ocfe_mesh(coord_system::AbstractKSCoordinateSystem,
						  num_elements::AbstractVector{<:Integer},
						  poly_degree::Union{Integer, AbstractVector{<:Integer}},
						  continuity_order::Union{Integer, AbstractVector{<:Integer}},
						  dims::Int)
	# Determine the element type T based on the coordinate system
	range = get_coordinate_range(coord_system, 1)
	T = eltype(range[1])  # Extract the element type from the first value in the range tuple

	# Ensure poly_degree is a vector of the correct length
	poly_degree_vec = typeof(poly_degree) <: Integer ? fill(poly_degree, dims) : collect(poly_degree)

	# Prepare structures for global nodes and elements
	global_nodes = Dict{Tuple, Int}()  # Map unique nodes to their global indices
	global_node_coords = Vector{Tuple}()  # List of unique node coordinates
	elements = Vector{KSElement{T, dims}}()  # Vector of mesh elements with correct type
	location_matrices = Vector{Dict{Int, Int}}()  # Location matrices for each element
	tensor_product_masks = Vector{AbstractArray{Bool, dims}}()  # Tensor product masks for each element

	# Function to add a node to the global node set and return its index
	function add_global_node(coord::Tuple, global_nodes, global_node_coords)
		if !haskey(global_nodes, coord)
			global_nodes[coord] = length(global_node_coords) + 1
			push!(global_node_coords, coord)
		end
		return global_nodes[coord]
	end

	# Loop over each element in the multi-dimensional space
	element_grids = Iterators.product((0:(num_elements[d] - 1) for d in 1:dims)...)

	for _ in element_grids
		element_nodes = Vector{Vector{T}}(undef, dims)
		differentiation_matrices = Vector{Vector{SparseMatrixCSC{T, Int}}}(undef, dims)
		quadrature_matrices = Vector{Vector{SparseMatrixCSC{T, Int}}}(undef, dims)
		node_indices = Vector{Int}()  # To store the indices of nodes for the current element

		# Add nodes and create differentiation and quadrature matrices
		for d in 1:dims
			range = get_coordinate_range(coord_system, d)
			nodes = range[1] .+ (range[2] - range[1]) .* (0:poly_degree_vec[d]) ./ poly_degree_vec[d]
			element_nodes[d] = nodes
			differentiation_matrices[d] = [derivative_matrix!(nodes)]
			quadrature_matrices[d] = [quadrature_matrix!(nodes)]
			for node in nodes
				global_index = add_global_node((node,), global_nodes, global_node_coords)
				push!(node_indices, global_index)
			end
		end

		# Create the element and add it to the elements vector
		element = KSElement(length(elements) + 1,
							1,
							NTuple{dims, Int}(poly_degree_vec))
		push!(elements, element)

		# Corrected creation of the location matrix
		push!(location_matrices, Dict(i => idx for (i, idx) in enumerate(node_indices)))

		# Create tensor product mask for the current element as a 3D BitArray
		num_nodes_per_dim = poly_degree_vec .+ 1
		mask = BitArray(undef, num_nodes_per_dim...)
		fill!(mask, true)
		push!(tensor_product_masks, mask)
	end

	# Create and return a KSMesh with global nodes, tensor product masks, and location matrices
	return KSMesh(elements, tensor_product_masks, location_matrices, zero(T))
end

"""
	get_coordinate_range(coord_system::AbstractKSCoordinateSystem, dim_index::Int) -> Union{Tuple{T, T}, Nothing} where T

Retrieve the coordinate range for a given dimension index from the provided coordinate system.

# Arguments
- `coord_system::AbstractKSCoordinateSystem`: The coordinate system from which to extract the range.
  Supported types include `KSCartesianCoordinates`, `KSPolarCoordinates`, `KSCylindricalCoordinates`, and `KSSphericalCoordinates`.
- `dim_index::Int`: The index of the dimension for which to retrieve the range. This index is 1-based.

# Returns
- `Union{Tuple{T, T}, Nothing}`: A tuple representing the range `[min, max]` of the specified dimension in the coordinate system.
  Returns `nothing` if the coordinate system or dimension index does not define a range.

# Supported Coordinate Systems
- `KSCartesianCoordinates`:
  - Uses `ranges` field and returns the range for the specified `dim_index`.
- `KSPolarCoordinates`:
  - `dim_index == 1`: Returns the radial range `r`.
  - `dim_index == 2`: Returns the angular range `theta`.
  - Throws an `ArgumentError` if `dim_index` is not 1 or 2.
- `KSCylindricalCoordinates`:
  - `dim_index == 1`: Returns the radial range `r`.
  - `dim_index == 2`: Returns the angular range `theta`.
  - `dim_index == 3`: Returns the height range `z`.
  - Throws an `ArgumentError` if `dim_index` is not 1, 2, or 3.
- `KSSphericalCoordinates`:
  - `dim_index == 1`: Returns the radial range `r`.
  - `dim_index == 2`: Returns the azimuthal angle range `theta`.
  - `dim_index == 3`: Returns the polar angle range `phi`.
  - Throws an `ArgumentError` if `dim_index` is not 1, 2, or 3.

# Errors
- Throws `ArgumentError` if the coordinate system or dimension index is unsupported, or if `dim_index` is out of bounds for the given coordinate system.

# Example
```julia
# Example usage with Cartesian coordinates
cartesian = KSCartesianCoordinates{Float64, 3}(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
range_x = get_coordinate_range(cartesian, 1)  # returns (0.0, 1.0)

# Example usage with Polar coordinates
polar = KSPolarCoordinates{Float64}((0.0, 1.0), (0.0, 2π))
range_r = get_coordinate_range(polar, 1)  # returns (0.0, 1.0)
range_theta = get_coordinate_range(polar, 2)  # returns (0.0, 2π)

# Example usage with Cylindrical coordinates
cylindrical = KSCylindricalCoordinates{Float64}((0.0, 1.0), (0.0, 2π), (0.0, 10.0))
range_r = get_coordinate_range(cylindrical, 1)  # returns (0.0, 1.0)
range_theta = get_coordinate_range(cylindrical, 2)  # returns (0.0, 2π)
range_z = get_coordinate_range(cylindrical, 3)  # returns (0.0, 10.0)

# Example usage with Spherical coordinates
spherical = KSSphericalCoordinates{Float64}((0.0, 1.0), (0.0, 2π), (0.0, π))
range_r = get_coordinate_range(spherical, 1)  # returns (0.0, 1.0)
range_theta = get_coordinate_range(spherical, 2)  # returns (0.0, 2π)
range_phi = get_coordinate_range(spherical, 3)  # returns (0.0, π)
"""

# Helper function to extract range based on coordinate system and dimension index
function get_coordinate_range(coord_system::AbstractKSCoordinateSystem, dim_index::Int)
	if isa(coord_system, KSCartesianCoordinates)
		return coord_system.ranges[dim_index]
	elseif isa(coord_system, KSPolarCoordinates)
		if dim_index == 1
			return coord_system.r
		elseif dim_index == 2
			return coord_system.theta
		else
			throw(ArgumentError("Invalid dimension index $dim_index for Polar Coordinates"))
		end
	elseif isa(coord_system, KSCylindricalCoordinates)
		if dim_index == 1
			return coord_system.r
		elseif dim_index == 2
			return coord_system.theta
		elseif dim_index == 3
			return coord_system.z
		else
			throw(ArgumentError("Invalid dimension index $dim_index for Cylindrical Coordinates"))
		end
	elseif isa(coord_system, KSSphericalCoordinates)
		if dim_index == 1
			return coord_system.r
		elseif dim_index == 2
			return coord_system.theta
		elseif dim_index == 3
			return coord_system.phi
		else
			throw(ArgumentError("Invalid dimension index $dim_index for Spherical Coordinates"))
		end
	else
		throw(ArgumentError("Unsupported coordinate system type $(typeof(coord_system))"))
	end
end

"""
	map_nodes_to_physical_domain(nodes::Vector{T}, coord_system::AbstractKSCoordinateSystem, dim_index::Int) -> Vector{T} where T

Maps a set of reference element nodes to their corresponding positions in the physical domain based on the provided coordinate system and dimension index.

# Arguments
- `nodes::Vector{T}`: A vector of nodes in the reference element, typically within a standard interval like `[-1, 1]` or `[0, 1]`.
- `coord_system::AbstractKSCoordinateSystem`: The coordinate system used to map the reference nodes to the physical domain. Supported types include `KSCartesianCoordinates`, `KSPolarCoordinates`, `KSCylindricalCoordinates`, and `KSSphericalCoordinates`.
- `dim_index::Int`: The dimension index indicating which physical dimension the nodes should be mapped to.

# Returns
- `Vector{T}`: A vector of the same type `T`, where each entry represents the mapped position of the corresponding reference node in the physical domain for the specified dimension.

# Description
The function takes nodes in the reference element and maps them to their corresponding positions in the physical domain using the provided coordinate system. The `dim_index` specifies which dimension of the physical domain the nodes should be mapped to. This is particularly useful in finite element methods where you need to transform local (reference) coordinates to global (physical) coordinates.

# Example
```julia
# Example usage with Cartesian coordinates
nodes = [-1.0, 0.0, 1.0]  # Reference element nodes
cartesian = KSCartesianCoordinates{Float64, 2}(((0.0, 1.0), (0.0, 1.0)))
mapped_nodes_x = map_nodes_to_physical_domain(nodes, cartesian, 1)  # Maps to the x-dimension
mapped_nodes_y = map_nodes_to_physical_domain(nodes, cartesian, 2)  # Maps to the y-dimension
```

# Notes
- The function uses `map_from_reference_element`, which is assumed to map a pair of reference nodes to their corresponding physical positions in a tuple form. The `dim_index` is used to extract the relevant dimension from the tuple.
- Ensure that `map_from_reference_element` correctly handles the mapping for the provided `coord_system` and that the tuple passed to it contains the correct number of elements.

# Errors
- Throws an `ArgumentError` if the `dim_index` is out of bounds or the coordinate system does not support the requested dimension.
"""

# Helper function to map nodes from reference to physical domain
function map_nodes_to_physical_domain(nodes::Vector{T}, coord_system::AbstractKSCoordinateSystem, dim_index::Int) where T
	mapped_nodes = T[]
	for node in nodes
		mapped_node = map_from_reference_element((node, node), coord_system)  # Ensure the tuple has 2 elements
		push!(mapped_nodes, mapped_node[dim_index])
	end
	return mapped_nodes
end

end # module SpectralMethods
