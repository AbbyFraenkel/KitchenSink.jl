module SpectralMethods

using LinearAlgebra, FastGaussQuadrature, StaticArrays
using BSplineKit: BSplineBasis, BSplineOrder, KnotVector, evaluate
using BSplineKit: BSplineKit  # Add this line to import the full module
using ..KSTypes, ..CoordinateSystems, ..CacheManagement, ..NumericUtilities

# Exported functions and types
export create_legendre_nodes_and_weights, gauss_legendre_with_boundary_nd
export create_standard_ks_cell, get_standard_spectral_properties
export derivative_matrix!, Lagrange_polynomials, quadrature_matrix, compute_integral
export kth_derivative_matrix!, barycentric_weights, barycentric_interpolation
export map_to_reference_cell, add_standard_spectral_properties!, apply_scaling_factors!
export quadrature_matrix_nd, compute_integral_nd, determine_scaling_factor
export get_or_create_standard_cell, create_ocfc_mesh, get_or_create_spectral_properties
export create_node_map, create_tensor_product_mask, derivative_matrix_nd
export interpolate_solution, is_point_in_cell, interpolate_in_cell, add_standard_cell!
export compute_quadrature_weights, spectral_jacobian, get_node_coordinates
export create_basis_functions, spectral_hessian, enforce_ck_continuity_bspline!
export interpolate_nd, enforce_ck_continuity_spectral!, compute_quadrature_weight
export get_node_coordinates, is_boundary_node
const STANDARD_CELL_CACHE = CacheManager{StandardKSCell}(100)
const SPECTRAL_PROPERTIES_CACHE = CacheManager{StandardSpectralProperties}(100)

@doc raw"""
	create_legendre_nodes_and_weights(p::Int) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}

Create Legendre-Gauss nodes and weights for a given polynomial order `p`.

Legendre-Gauss quadrature is a method for approximating the integral of a function,
particularly useful for polynomial functions. The nodes are the roots of the Legendre
polynomial \( P_p(x) \), and the weights are derived to ensure exact integration for
polynomials of degree up to \( 2p-1 \).

# Arguments
- `p::Int`: Polynomial order (must be at least 3)

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}`:
  (nodes_with_boundary, weights_with_boundary, nodes, weights)
  - `nodes_with_boundary`: Nodes including the boundary points -1 and 1.
  - `weights_with_boundary`: Weights including boundary weights (set to 0).
  - `nodes`: Legendre-Gauss nodes (roots of the Legendre polynomial \( P_p(x) \)).
  - `weights`: Corresponding weights for the nodes.

# Throws
- `ArgumentError`: If `p` is less than 3

# Mathematical Description
The Legendre-Gauss nodes \( x_i \) are the roots of the Legendre polynomial \( P_p(x) \),
which is defined by the recurrence relation: \[ (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
\] with initial conditions \( P_0(x) = 1 \) and \( P_1(x) = x \).

The weights \( w_i \) are given by: \[ w_i = \frac{2}{(1 - x_i^2)[P'_p(x_i)]^2} \] where \(
P'_p(x) \) is the derivative of the Legendre polynomial \( P_p(x) \).
"""
function create_legendre_nodes_and_weights(p::Int)
	p < 3 && throw(ArgumentError("Polynomial order p must be at least 3, but got p = $p"))
	nodes, weights = gausslegendre(p)
	nodes_with_boundary = vcat(-1.0, nodes, 1.0)
	weights_with_boundary = vcat(0.0, weights, 0.0)
	return nodes_with_boundary, weights_with_boundary, nodes, weights
end
@doc raw"""
	gauss_legendre_with_boundary_nd(polynomial_degree::Union{Int, NTuple{N, Int}}; dim::Int = 1) where N -> Tuple

Create multi-dimensional Gauss-Legendre nodes and weights with boundary points.

# Arguments
- `polynomial_degree::Union{Int, NTuple{N, Int}}`: Polynomial degree(s) for each dimension
- `dim::Int = 1`: Number of dimensions (default: 1)

# Returns
- `Tuple`: (nodes, weights, nodes_interior, weights_interior)
  - `nodes`: Multi-dimensional Gauss-Legendre nodes including boundary points.
  - `weights`: Multi-dimensional Gauss-Legendre weights including boundary points.
  - `nodes_interior`: Interior Gauss-Legendre nodes.
  - `weights_interior`: Interior Gauss-Legendre weights.

# Throws
- `ArgumentError`: If dimensions or polynomial degrees are invalid

# Mathematical Description
For each dimension \( d \), the Gauss-Legendre nodes and weights are computed using the
Legendre polynomial \( P_{p_d}(x) \). The nodes \( x_i \) are the roots of \( P_{p_d}(x) \),
and the weights \( w_i \) are given by: \[ w_i = \frac{2}{(1 - x_i^2)[P'_{p_d}(x_i)]^2} \]
where \( P'_{p_d}(x) \) is the derivative of the Legendre polynomial \( P_{p_d}(x) \).

The multi-dimensional nodes and weights are then formed by taking the tensor product of the
one-dimensional nodes and weights.
"""
function gauss_legendre_with_boundary_nd(
	polynomial_degree::Union{Int, NTuple{N, Int}};
	dim::Int = 1,
) where {N}
	# Convert polynomial_degree to tuple if single Int
	p_vec = if isa(polynomial_degree, Int)
		ntuple(_ -> polynomial_degree, dim)
	else
		polynomial_degree
	end

	# Validate inputs
	length(p_vec) == dim || throw(DimensionMismatch(
		"Length of polynomial_degree must match dimension $dim"))
	all(p_i >= 3 for p_i in p_vec) || throw(ArgumentError(
		"Polynomial degree must be at least 3 in each dimension"))

	# Generate nodes and weights for each dimension with proper type inference
	T = eltype(create_legendre_nodes_and_weights(first(p_vec))[1])
	nodes = Vector{Vector{T}}(undef, dim)
	weights = Vector{Vector{T}}(undef, dim)
	nodes_interior = Vector{Vector{T}}(undef, dim)
	weights_interior = Vector{Vector{T}}(undef, dim)

	for d in 1:dim
		# Create nodes and weights for this dimension with its specific p value
		n_d, w_d, n_i_d, w_i_d = create_legendre_nodes_and_weights(p_vec[d])

		# Store with proper dimensioning
		nodes[d] = reshape(n_d, p_vec[d] + 2)
		weights[d] = reshape(w_d, p_vec[d] + 2)
		nodes_interior[d] = reshape(n_i_d, p_vec[d])
		weights_interior[d] = reshape(w_i_d, p_vec[d])
	end

	return nodes, weights, nodes_interior, weights_interior
end

@doc raw"""
	get_or_create_spectral_properties(p::Int, continuity_order::Int = 2) -> StandardSpectralProperties

Retrieve or create spectral properties for a given polynomial order and continuity order.

# Arguments
- `p::Int`: Polynomial order.
- `continuity_order::Int = 2`: Continuity order (default: 2).

# Returns
- `StandardSpectralProperties`: Spectral properties for the given polynomial and continuity
  order.

# Throws
- `ArgumentError`: If the polynomial order or continuity order is invalid.

# Mathematical Description
The spectral properties include:
- Legendre-Gauss nodes and weights, both with and without boundary points.
- Differentiation matrices \( D \) for the given polynomial order \( p \).
- Quadrature matrix \( Q \) derived from the weights.

The differentiation matrix \( D \) is computed such that: \[ D_{ij} =
\frac{P'_p(x_i)}{P_p(x_i)} \] where \( P_p(x) \) is the Legendre polynomial of order \( p \).

The quadrature matrix \( Q \) is a diagonal matrix with the weights \( w_i \) on the
diagonal: \[ Q_{ii} = w_i \]
"""
function get_or_create_spectral_properties(p::Int, continuity_order::Int = 2)
	key = (p, continuity_order)
	return get_or_create_cached_item(SPECTRAL_PROPERTIES_CACHE, key) do
		nodes_with_boundary, weights_with_boundary, nodes_interior, weights_interior = create_legendre_nodes_and_weights(
			p
		)
		D_with_boundary, D_interior = derivative_matrix!(p)
		quadrature_matrix_val = quadrature_matrix(weights_interior)
		D_matrices_with_boundary, D_matrices_interior = kth_derivative_matrix!(
			p, max(1, continuity_order)
		)
		higher_order_diff_matrices_with_boundary = D_matrices_with_boundary[1:continuity_order]
		higher_order_diff_matrices_interior = D_matrices_interior[1:continuity_order]
		StandardSpectralProperties(;
			p = p,
			continuity_order = continuity_order,
			nodes_with_boundary = nodes_with_boundary,
			nodes_interior = nodes_interior,
			weights_with_boundary = weights_with_boundary,
			weights_interior = weights_interior,
			differentiation_matrix_with_boundary = D_with_boundary,
			differentiation_matrix_interior = D_interior,
			quadrature_matrix = quadrature_matrix_val,
			higher_order_diff_matrices_with_boundary = higher_order_diff_matrices_with_boundary,
			higher_order_diff_matrices_interior = higher_order_diff_matrices_interior,
		)
	end
end

@doc raw"""
	add_standard_cell!(cell::StandardKSCell)

Add a standard cell to the global set of standard cells.
````
# Arguments
- `cell::StandardKSCell`: The standard cell to add.

# Throws
- `ArgumentError`: If the cell already exists in the global set.
"""
function add_standard_cell!(cell::StandardKSCell)
	# Create key from cell's properties
	key = (cell.p, cell.level, cell.continuity_order)

	# Check if key exists in cache
	if haskey(STANDARD_CELL_CACHE.items, key)
		throw(
			ArgumentError(
				"Standard cell with p=$(cell.p), level=$(cell.level), and continuity_order=$(cell.continuity_order) already exists."
			),
		)
	end

	# Add to cache
	STANDARD_CELL_CACHE.items[key] = cell
	return cell
end

@doc raw"""
	get_or_create_standard_cell(p::NTuple{N, Int}, level::Int; continuity_order::NTuple{N, Int} = ntuple(_ -> 2, N)) where N -> StandardKSCell

Retrieve or create a standard cell for given polynomial degrees, level, and continuity order.

# Arguments
- `p::NTuple{N, Int}`: Polynomial degrees for each dimension.
- `level::Int`: Refinement level.
- `continuity_order::NTuple{N, Int} = ntuple(_ -> 2, N)`: Continuity order for each dimension
  (default: 2).

# Returns
- `StandardKSCell`: The standard cell for the given parameters.

# Throws
- `ArgumentError`: If the polynomial degrees, level, or continuity order are invalid.

# Mathematical Description
A standard cell is defined by its polynomial degrees \( p \), refinement level \( l \), and
continuity order \( k \). The polynomial degrees determine the order of the basis functions
used within the cell. The continuity order specifies the smoothness of the basis functions
across cell boundaries.

The standard cell is created using the function `create_standard_ks_cell`, which generates
the necessary spectral properties and basis functions for the cell.
"""
function get_or_create_standard_cell(
	p::NTuple{N, Int}, level::Int; continuity_order::NTuple{N, Int} = ntuple(_ -> 1, N)
) where {N}
	all(p_i >= 1 for p_i in p) ||
		throw(ArgumentError("All polynomial degrees must be at least 1"))
	level >= 1 || throw(ArgumentError("Level must be at least 1"))
	all(k >= 0 for k in continuity_order) ||
		throw(ArgumentError("Continuity order must be non-negative"))

	key = (p, level, continuity_order)

	# Create a parametric standard cell
	return get_or_create_cached_item(STANDARD_CELL_CACHE, key) do
		create_standard_ks_cell(p, level, length(p); continuity_order = continuity_order)
	end
end
@doc raw"""
	create_standard_ks_cell(p::NTuple{N, Int}, level::Int, dim::Int; continuity_order::NTuple{N, Int} = ntuple(_ -> 2, N)) where N -> StandardKSCell

Create a standard KS cell for given polynomial degrees, level, dimensions, and continuity
order.

# Arguments
- `p::NTuple{N, Int}`: Polynomial degrees for each dimension.
- `level::Int`: Refinement level.
- `dim::Int`: Number of dimensions.
- `continuity_order::NTuple{N, Int} = ntuple(_ -> 2, N)`: Continuity order for each dimension
  (default: 2).

# Returns
- `StandardKSCell`: The created standard KS cell.

# Mathematical Description
A standard KS cell is created by scaling the nodes and weights based on the refinement level.
The differentiation matrices and quadrature matrix are also scaled accordingly.

The scaling factor for the refinement level \( l \) is given by: \[
\text{level\_scale\_factor} = 2^{(-l + 1)} \]

The nodes and weights are scaled as: \[ \text{nodes\_with\_boundary} =
\text{nodes\_with\_boundary} \times \text{level\_scale\_factor} \] \[
\text{weights\_with\_boundary} = \text{weights\_with\_boundary} \times
\text{level\_scale\_factor} \]

The differentiation matrices are scaled as: \[ D_{\text{with\_boundary}} =
\frac{D_{\text{with\_boundary}}}{\text{level\_scale\_factor}} \] \[ D_{\text{interior}} =
\frac{D_{\text{interior}}}{\text{level\_scale\_factor}} \]

The quadrature matrix is scaled as: \[ Q = Q \times \text{level\_scale\_factor} \]

Higher-order differentiation matrices are scaled as: \[ D_{\text{higher\_order}} =
\frac{D_{\text{higher\_order}}}{\text{level\_scale\_factor}^k} \] where \( k \) is the order
of the differentiation matrix.
"""
function create_standard_ks_cell(
	p::NTuple{N, Int},
	level::Int,
	dim::Int;
	continuity_order::NTuple{N, Int} = ntuple(_ -> 2, N)) where {N}
	spectral_props = ntuple(
		d -> get_or_create_spectral_properties(p[d], continuity_order[d]), dim)

	# Scale the nodes based on the refinement level
	level_scale_factor = 2.0^(-level + 1)

	nodes_with_boundary = ntuple(
		d -> spectral_props[d].nodes_with_boundary * level_scale_factor, dim)
	nodes_interior = ntuple(d -> spectral_props[d].nodes_interior * level_scale_factor, dim)
	weights_with_boundary = ntuple(
		d -> spectral_props[d].weights_with_boundary * level_scale_factor, dim)
	weights_interior = ntuple(
		d -> spectral_props[d].weights_interior * level_scale_factor, dim)

	differentiation_matrix_with_boundary = ntuple(
		d -> spectral_props[d].differentiation_matrix_with_boundary / level_scale_factor,
		dim,
	)
	differentiation_matrix_interior = ntuple(
		d -> spectral_props[d].differentiation_matrix_interior / level_scale_factor,
		dim,
	)
	quadrature_matrix_val = ntuple(
		d -> spectral_props[d].quadrature_matrix * level_scale_factor, dim)

	higher_order_diff_matrices_with_boundary = ntuple(
		d -> [
			m / (level_scale_factor^k)
			for (k, m) in
			enumerate(spectral_props[d].higher_order_diff_matrices_with_boundary)
		],
		dim,
	)
	higher_order_diff_matrices_interior = ntuple(
		d -> [
			m / (level_scale_factor^k)
			for
			(k, m) in enumerate(spectral_props[d].higher_order_diff_matrices_interior)
		],
		dim,
	)

	return StandardKSCell(;
		p = p,
		level = level,
		continuity_order = continuity_order,
		nodes_with_boundary = nodes_with_boundary,
		nodes_interior = nodes_interior,
		weights_with_boundary = weights_with_boundary,
		weights_interior = weights_interior,
		differentiation_matrix_with_boundary = differentiation_matrix_with_boundary,
		differentiation_matrix_interior = differentiation_matrix_interior,
		quadrature_matrix = quadrature_matrix_val,
		higher_order_diff_matrices_with_boundary = higher_order_diff_matrices_with_boundary,
		higher_order_diff_matrices_interior = higher_order_diff_matrices_interior,
	)
end
@doc raw"""
	derivative_matrix!(p::Int; tol::Float64 = 1e-12) -> Tuple{Matrix{Float64}, Matrix{Float64}}

Compute the derivative matrix for Legendre polynomials of order `p`.

# Arguments
- `p::Int`: Polynomial order.
- `tol::Float64 = 1e-12`: Tolerance for numerical comparisons.

# Returns
- `Tuple{Matrix{Float64}, Matrix{Float64}}`: (D_with_boundary, D_interior)

# Mathematical Description
The derivative matrix \( D \) for Legendre polynomials is computed using barycentric
interpolation. For nodes \( x_i \) and barycentric weights \( w_i \), the off-diagonal
elements of the derivative matrix are given by: \[ D_{ij} = \frac{w_j}{w_i (x_i - x_j)} \quad
\text{for} \quad i \neq j \]

The diagonal elements are computed to ensure that the sum of each row is zero: \[ D_{ii} =
-\sum_{j \neq i} D_{ij} \]

The function returns two matrices:
- `D_with_boundary`: Derivative matrix including boundary points.
- `D_interior`: Derivative matrix for interior points.
"""
function derivative_matrix!(p::Int; tol::Float64 = 1e-12)
	nodes_with_boundary, _, nodes_interior, _ = create_legendre_nodes_and_weights(p)

	barycentric_weights_with_boundary = barycentric_weights(nodes_with_boundary)
	barycentric_weights_interior = barycentric_weights(nodes_interior)

	recip_weights_with_boundary = 1 ./ barycentric_weights_with_boundary
	recip_weights_interior = 1 ./ barycentric_weights_interior

	node_diffs_with_boundary = [
		nodes_with_boundary[i] - nodes_with_boundary[j]
		for i in 1:(p + 2), j in 1:(p + 2)
	]
	node_diffs_interior = [nodes_interior[i] - nodes_interior[j] for i in 1:p, j in 1:p]

	D_with_boundary = zeros(p + 2, p + 2)
	D_interior = zeros(p, p)

	for i in 1:(p + 2), j in 1:(p + 2)
		if i != j
			D_with_boundary[i, j] =
				barycentric_weights_with_boundary[j] *
				recip_weights_with_boundary[i] /
				node_diffs_with_boundary[i, j]
		end
	end

	for i in 1:p, j in 1:p
		if i != j
			D_interior[i, j] =
				barycentric_weights_interior[j] * recip_weights_interior[i] /
				node_diffs_interior[i, j]
		end
	end

	for i in 1:(p + 2)
		D_with_boundary[i, i] = -sum(D_with_boundary[i, j] for j in 1:(p + 2) if j != i)
	end

	for i in 1:p
		D_interior[i, i] = -sum(D_interior[i, j] for j in 1:p if j != i)
	end

	return D_with_boundary, D_interior
end
@doc raw"""
	kth_derivative_matrix!(p::Int, k_max::Int; tol::Float64 = 1e-12, max_condition_number::Float64 = 1e20) -> Tuple{Vector{Matrix{Float64}}, Vector{Matrix{Float64}}}

Compute the kth derivative matrices up to `k_max` for Legendre polynomials of order `p`.

# Arguments
- `p::Int`: Polynomial order.
- `k_max::Int`: Maximum derivative order.
- `tol::Float64 = 1e-12`: Tolerance for numerical comparisons.
- `max_condition_number::Float64 = 1e20`: Maximum allowed condition number.

# Returns
- `Tuple{Vector{Matrix{Float64}}, Vector{Matrix{Float64}}}`: (D_matrices_with_boundary,
  D_matrices_interior)

# Throws
- `ArgumentError`: If the condition number exceeds `max_condition_number`.

# Mathematical Description
The kth derivative matrix \( D^{(k)} \) is computed iteratively using the first derivative
matrix \( D \): \[ D^{(k)} = (D^{(k-1)}) \cdot D \]

The condition number of the resulting matrix is checked to ensure numerical stability. If the
condition number exceeds `max_condition_number`, the function throws an `ArgumentError`.

The function returns two vectors of matrices:
- `D_matrices_with_boundary`: kth derivative matrices including boundary points.
- `D_matrices_interior`: kth derivative matrices for interior points.
"""
function kth_derivative_matrix!(
	p::Int,
	k_max::Int;
	tol::Float64 = 1e-12,
	max_condition_number::Float64 = 1e20,
)
	D_with_boundary, D_interior = derivative_matrix!(p; tol = tol)

	D_matrices_with_boundary = [D_with_boundary]
	D_matrices_interior = [D_interior]

	for k in 2:min(k_max, p)  # Limit k_max to p to avoid ill-conditioning
		D_with_boundary_k = D_matrices_with_boundary[end] * D_with_boundary
		D_interior_k = D_matrices_interior[end] * D_interior

		cond_with_boundary = cond(D_with_boundary_k)
		cond_interior = cond(D_interior_k)

		# if cond_with_boundary > max_condition_number || cond_interior >
		# max_condition_number throw(ArgumentError("Condition number exceeds
		#   max_condition_number at order $k")) end

		push!(D_matrices_with_boundary, D_with_boundary_k)
		push!(D_matrices_interior, D_interior_k)
	end

	return D_matrices_with_boundary, D_matrices_interior
end

@doc raw"""
	barycentric_weights(nodes::Vector{T}) where T <: AbstractFloat -> Vector{T}

Compute barycentric weights for given nodes.

# Arguments
- `nodes::Vector{T}`: Vector of node positions.

# Returns
- `Vector{T}`: Barycentric weights.

# Throws
- `ArgumentError`: If the number of nodes is less than 3.

# Mathematical Description
Barycentric weights \( w_i \) for nodes \( x_i \) are computed as: \[ w_i = \frac{1}{\prod_{j
\neq i} (x_i - x_j)} \]

These weights are used in barycentric interpolation to compute the interpolating polynomial
efficiently.
"""
function barycentric_weights(nodes::Vector{T}) where {T <: AbstractFloat}
	length(nodes) < 3 && throw(ArgumentError("The number of nodes must be at least 3"))
	n = length(nodes)
	weights = ones(T, n)
	for j in 1:n
		for k in 1:n
			if j != k
				weights[j] *= (nodes[j] - nodes[k])
			end
		end
		weights[j] = 1 / weights[j]
	end
	return weights
end

@doc raw"""
	barycentric_interpolation(points::AbstractVector{T}, values::AbstractVector{S}, x; tol::Real = 1e-14) where {T <: Real, S <: Real} -> S

Perform barycentric interpolation at a given point.

# Arguments
- `points::AbstractVector{T}`: Vector of interpolation points.
- `values::AbstractVector{S}`: Vector of function values at the interpolation points.
- `x`: The point at which to interpolate.
- `tol::Real = 1e-14`: Tolerance for numerical comparisons (default: 1e-14).

# Returns
- `S`: Interpolated value at the given point.

# Mathematical Description
Barycentric interpolation is a method of polynomial interpolation that avoids the numerical
instability of other interpolation methods. Given interpolation points \( x_i \) and
corresponding function values \( f_i \), the interpolated value \( f(x) \) at a point \( x \)
is computed as:

\[ f(x) = \frac{\sum_{j=1}^{n} \frac{w_j}{x - x_j} f_j}{\sum_{j=1}^{n} \frac{w_j}{x - x_j}}
\]

where \( w_j \) are the barycentric weights, defined as:

\[ w_j = \frac{1}{\prod_{k \neq j} (x_j - x_k)} \]

If \( x \) exactly matches one of the interpolation points \( x_i \), the function returns
the corresponding function value \( f_i \) to avoid division by zero.

# Throws
- `ArgumentError`: If the number of points and values do not match.

# Example
```julia
points = [0.0, 1.0, 2.0]
values = [1.0, 2.0, 0.0]
x = 1.5
interpolated_value = barycentric_interpolation(points, values, x)
```
"""
function barycentric_interpolation(
	points::AbstractVector{T},
	values::AbstractVector{S},
	x;
	tol::Real = 1e-14,
) where {T <: Real, S <: Real}
	weights = barycentric_weights(points)
	n = length(points)
	numerator = zero(promote_type(T, S))
	denominator = zero(T)

	exact_match = findfirst(p -> abs(x - p) < tol, points)
	if exact_match !== nothing
		return values[exact_match]
	end

	for j in 1:n
		diff = x - points[j]
		if abs(diff) < tol
			return values[j]
		end
		t = weights[j] / diff
		numerator += t * values[j]
		denominator += t
	end

	result = numerator / denominator
	return abs(result) < eps(typeof(result)) ? zero(result) : result
end
@doc raw"""
	quadrature_matrix(weights::AbstractVector) -> Diagonal{T, Vector{T}} where T

Create a quadrature matrix from given weights.

# Arguments
- `weights::AbstractVector`: Vector of quadrature weights.

# Returns
- `Diagonal{T, Vector{T}}`: Diagonal matrix with the weights on the diagonal.

# Mathematical Description
The quadrature matrix \( Q \) is a diagonal matrix where the diagonal elements are the
quadrature weights \( w_i \): \[ Q_{ii} = w_i \] This matrix is used to perform numerical
integration by matrix multiplication.
"""
function quadrature_matrix(weights::AbstractVector)
	return Diagonal(weights)
end

@doc raw"""
	compute_integral(weights::AbstractVector, values::AbstractVector) -> Float64

Compute the integral of a function given its values and quadrature weights.

# Arguments
- `weights::AbstractVector`: Vector of quadrature weights.
- `values::AbstractVector`: Vector of function values at the quadrature nodes.

# Returns
- `Float64`: The computed integral.

# Mathematical Description
The integral \( I \) of a function \( f(x) \) can be approximated using quadrature weights \(
w_i \) and function values \( f(x_i) \) at the quadrature nodes \( x_i \): \[ I \approx
\sum_{i} w_i f(x_i) \] This is computed as the dot product of the weights and the values: \[
I = \mathbf{w} \cdot \mathbf{f} \]
"""
function compute_integral(weights::AbstractVector, values::AbstractVector)
	return dot(weights, values)
end

@doc raw"""
	compute_integral_nd(values::AbstractArray, quad_matrices::NTuple{N, AbstractMatrix}) where N -> Float64

Compute the multi-dimensional integral of a function given its values and quadrature
matrices.

# Arguments
- `values::AbstractArray`: Array of function values at the quadrature nodes.
- `quad_matrices::NTuple{N, AbstractMatrix}`: Tuple of quadrature matrices for each
  dimension.

# Returns
- `Float64`: The computed multi-dimensional integral.

# Throws
- `DimensionMismatch`: If the number of dimensions in values does not match the number of
  quadrature matrices.

# Mathematical Description
The multi-dimensional integral \( I \) of a function \( f(\mathbf{x}) \) over an
N-dimensional domain can be approximated using quadrature matrices \( Q_d \) for each
dimension \( d \): \[ I \approx \sum_{i_1} \sum_{i_2} \cdots \sum_{i_N} Q_{i_1} Q_{i_2}
\cdots Q_{i_N} f(x_{i_1}, x_{i_2}, \ldots, x_{i_N}) \]

This is computed by iteratively applying the quadrature matrices to the function values: \[
\mathbf{F} = Q_1 \mathbf{F} \] \[ \mathbf{F} = Q_2 \mathbf{F} \] \[ \vdots \] \[ \mathbf{F} =
Q_N \mathbf{F} \]

The final result is a scalar value representing the integral.
"""
function compute_integral_nd(
	values::AbstractArray,
	quad_matrices::NTuple{N, AbstractMatrix},
) where {N}
	if ndims(values) != N
		throw(
			DimensionMismatch(
				"The number of dimensions in values ($(ndims(values))) does not match the number of quadrature matrices ($N)"
			),
		)
	end

	result = values
	for (dim, Q) in enumerate(quad_matrices)
		result = dropdims(sum(Q * reshape(result, size(Q, 2), :); dims = 1); dims = 1)
		result = reshape(result, size(values)[(dim + 1):end]...)
	end

	return result[]  # Return the scalar value
end
@doc raw"""
	interpolate_nd(standard_cell::StandardKSCell{T, N}, values::AbstractArray{T}, point::NTuple{N, T}) where {T <: Real, N} -> T

Interpolate the solution at a given point in the standard cell.

# Arguments
- `standard_cell::StandardKSCell{T, N}`: The standard cell containing the solution.
- `values::AbstractArray{T}`: Array of solution values at the cell nodes.
- `point::NTuple{N, T}`: The point at which to interpolate.

# Returns
- `T`: Interpolated value at the given point.

# Throws
- `ArgumentError`: If the interpolation point is outside the reference cell bounds.

# Mathematical Description
The interpolation is performed using barycentric interpolation. For each dimension \( d \),
the interpolation at point \( x_d \) is computed as: \[ f(x_d) = \sum_{i} \frac{w_i}{x_d -
x_i} f(x_i) \] where \( w_i \) are the barycentric weights and \( x_i \) are the nodes.

The interpolation is performed recursively for each dimension until the final interpolated
value is obtained.

# Example
```julia
standard_cell = StandardKSCell{Float64, 2}(...)
values = rand(4, 4)  # Example values at the cell nodes
point = (0.5, 0.5)  # Point at which to interpolate
interpolated_value = interpolate_nd(standard_cell, values, point)
"""
function interpolate_nd(
	standard_cell::StandardKSCell{T, N},
	values::AbstractArray{T},
	point::NTuple{N, T},
) where {T, N}
	# Check if the point is within the bounds of the standard cell
	for dim in 1:N
		min_bound, max_bound = extrema(standard_cell.nodes_with_boundary[dim])
		(min_bound <= point[dim] <= max_bound) || throw(
			ArgumentError(
				"Interpolation point must be within the reference cell bounds for dimension $dim: [$min_bound, $max_bound]"
			),
		)
	end

	function interpolate_recursive(current_dim::Int, indices::Vector{Int})
		if current_dim > N
			return values[CartesianIndex(Tuple(indices))]
		end
		nodes = standard_cell.nodes_with_boundary[current_dim]
		interp_vals = [
			begin
				new_indices = copy(indices)
				new_indices[current_dim] = i
				interpolate_recursive(current_dim + 1, new_indices)
			end
			for i in 1:length(nodes)
		]
		return barycentric_interpolation(collect(nodes), interp_vals, point[current_dim])
	end

	result = interpolate_recursive(1, ones(Int, N))
	return isapprox(result, zero(T); atol = eps(T)) ? zero(T) : result
end
@doc raw"""
	interpolate_solution(mesh::KSMesh{T, N}, u::AbstractVector{T}, point::AbstractVector{T}) where {T <: Real, N} -> T

Interpolate the solution at a given point in the mesh.

# Arguments
- `mesh::KSMesh{T, N}`: The mesh containing the solution.
- `u::AbstractVector{T}`: Vector of solution values at the mesh nodes.
- `point::AbstractVector{T}`: The point at which to interpolate.

# Returns
- `T`: Interpolated value at the given point.

# Throws
- `ArgumentError`: If the point is not found in any cell of the mesh.

# Mathematical Description
The interpolation is performed by first identifying the cell that contains the point. Once
the cell is identified, the interpolation is performed using the `interpolate_in_cell`
function, which uses barycentric interpolation within the cell.

The interpolation within the cell is computed as: \[ f(\mathbf{x}) = \sum_{i}
\frac{w_i}{\mathbf{x} - \mathbf{x}_i} f(\mathbf{x}_i) \] where \( w_i \) are the barycentric
weights and \( \mathbf{x}_i \) are the nodes within the cell.
"""
function interpolate_solution(
	mesh::KSMesh{T, N},
	u::AbstractVector{T},
	point::AbstractVector{T},
) where {T, N}
	# First find the cell containing the point
	point_tuple = tuple(point...)

	for cell in mesh.cells
		if is_point_in_cell(cell, point, mesh)
			# Get the standard cell
			standard_cell = get_or_create_standard_cell(
				cell.p,
				cell.level;
				continuity_order = cell.continuity_order,
			)

			# Map point to reference coordinates
			ref_coords = get_node_coordinates(
				mesh.coord_system, cell, point, mesh.num_cells_per_dim
			)

			# Reshape solution values for the cell
			cell_values = reshape(
				@view(u[collect(values(cell.node_map))]),
				ntuple(i -> cell.p[i] + 2, N),
			)

			# Perform interpolation in reference coordinates
			return interpolate_nd(standard_cell, cell_values, ref_coords)
		end
	end
	throw(ArgumentError("Point $point not found in any cell of the mesh"))
end


@doc raw"""
	is_point_in_cell(cell::KSCell{T, N}, point::AbstractVector{T}, mesh::KSMesh{T, N}) where {T <: Real, N} -> Bool

Check if a point is within the bounds of a given cell in the mesh.

# Arguments
- `cell::KSCell{T, N}`: The cell to check.
- `point::AbstractVector{T}`: The point to check.
- `mesh::KSMesh{T, N}`: The mesh containing the cell.

# Returns
- `Bool`: `true` if the point is within the cell bounds and the cell is a leaf and not
  fictitious, `false` otherwise.

# Mathematical Description
The function checks if the point \( \mathbf{x} \) lies within the bounds of the cell. The
cell bounds are defined by the minimum and maximum coordinates of the cell corners: \[
\text{cell\_min} = \min(\text{cell\_corners}) \] \[ \text{cell\_max} =
\max(\text{cell\_corners}) \]

The point is within the cell if: \[ \text{cell\_min} \leq \mathbf{x} \leq \text{cell\_max} \]

Additionally, the cell must be a leaf, not fictitious, and the point must lie within the
physical domain of the mesh.
"""
function is_point_in_cell(
	cell::KSCell{T, N},
	point::AbstractVector{T},
	mesh::KSMesh{T, N},
) where {T, N}
	cell_corners = get_cell_corners(cell)
	cell_min = minimum(cell_corners)
	cell_max = maximum(cell_corners)

	in_bounds = all(x -> isapprox(x, true; atol = eps(T)), cell_min .<= point .<= cell_max)
	return in_bounds && cell.is_leaf && !cell.is_fictitious && mesh.physical_domain(point)
end

@doc raw"""
	interpolate_in_cell(cell::KSCell{T, N}, u::AbstractVector{T}, point::AbstractVector{T}, mesh::KSMesh{T, N}) where {T <: Real, N} -> T

Interpolate the solution at a given point within a cell.

# Arguments
- `cell::KSCell{T, N}`: The cell containing the solution.
- `u::AbstractVector{T}`: Vector of solution values at the cell nodes.
- `point::AbstractVector{T}`: The point at which to interpolate.
- `mesh::KSMesh{T, N}`: The mesh containing the cell.

# Returns
- `T`: Interpolated value at the given point.

# Mathematical Description
The interpolation is performed using barycentric interpolation within the standard cell. The
steps are:
1. Map the point to the standard cell coordinates.
2. Reshape the solution values to match the standard cell structure.
3. Perform the interpolation using the `interpolate_nd` function.

The mapping to the standard cell coordinates is given by: \[ \text{local\_coords} = \frac{2
(\mathbf{point} - \text{cell\_min})}{\text{cell\_max} - \text{cell\_min}} - 1 \]

The interpolation within the standard cell is computed as: \[ f(\mathbf{x}) = \sum_{i}
\frac{w_i}{\mathbf{x} - \mathbf{x}_i} f(\mathbf{x}_i) \] where \( w_i \) are the barycentric
weights and \( \mathbf{x}_i \) are the nodes within the cell.
"""
function interpolate_in_cell(
	cell::KSCell{T, N},
	u::AbstractVector{T},
	point::AbstractVector{T},
	mesh::KSMesh{T, N},
) where {T, N}
	standard_cell = get_or_create_standard_cell(
		cell.p,
		cell.level;
		continuity_order = cell.continuity_order,
	)
	local_coords = get_node_coordinates(
		mesh.coord_system, cell, point, mesh.num_cells_per_dim
	)
	cell_values = reshape(
		@view(u[collect(values(cell.node_map))]), ntuple(i -> cell.p[i] + 2, N))
	return interpolate_nd(standard_cell, cell_values, local_coords)
end

@doc raw"""
	create_tensor_product_mask(p_vec::NTuple{N, Int}, continuity_order::NTuple{N, Int}, is_boundary::Bool, is_fictitious::Bool, dim::Int) where N -> NTuple{N, BitVector}

Create a tensor product mask for a given polynomial degree vector, continuity order, boundary
status, and dimension.

# Arguments
- `p_vec::NTuple{N, Int}`: Polynomial degrees for each dimension.
- `continuity_order::NTuple{N, Int}`: Continuity order for each dimension.
- `is_boundary::Bool`: Whether the cell is on the boundary.
- `is_fictitious::Bool`: Whether the cell is fictitious.
- `dim::Int`: Number of dimensions.

# Returns
- `NTuple{N, BitVector}`: Tensor product mask.

# Mathematical Description
The tensor product mask is created by iterating over each dimension and setting the mask
values based on the continuity order and boundary status. For each dimension \( d \):
- If the cell is on the boundary, the mask values at the boundaries are set to `false` based
  on the continuity order.
- If the cell is fictitious, the mask values for the interior points are set to `false`.

The resulting mask is a tuple of bit vectors, where each bit vector represents the mask for
one dimension.

# Logical Operations
- The mask values are initialized to `true` for all points.
- For boundary cells, the mask values at the boundaries are set to `false`: \[
  \text{mask}[d][1:\min(\text{continuity\_order}[d], p\_vec[d])] \leftarrow \text{false} \]
  \[ \text{mask}[d][(\text{end} - \min(\text{continuity\_order}[d], p\_vec[d]) +
  1):\text{end}] \leftarrow \text{false} \]
- For fictitious cells, the mask values for the interior points are set to `false`: \[
  \text{mask}[d][2:(\text{end} - 1)] \leftarrow \text{false} \]
"""
function create_tensor_product_mask(
	p_vec::NTuple{N, Int},
	continuity_order::NTuple{N, Int},
	is_boundary::Bool,
	is_fictitious::Bool,
	dim::Int,
) where {N}
	mask = ntuple(d -> trues(p_vec[d] + 2), dim)
	for d in 1:dim
		if is_boundary
			mask[d][1:min(continuity_order[d], p_vec[d])] .= false
			mask[d][(end - min(continuity_order[d], p_vec[d]) + 1):end] .= false
		else
			mask[d][1:min(continuity_order[d], p_vec[d])] .= false
			mask[d][(end - min(continuity_order[d], p_vec[d]) + 1):end] .= false
		end
		if is_fictitious
			mask[d][2:(end - 1)] .= false
		end
	end
	return mask
end

@doc raw"""
	quadrature_matrix_nd(weights_tuple::NTuple{N, AbstractVector}) where N -> NTuple{N, Diagonal{T, Vector{T}}} where T

Create multi-dimensional quadrature matrices from given weights.

# Arguments
- `weights_tuple::NTuple{N, AbstractVector}`: Tuple of quadrature weights for each dimension.

# Returns
- `NTuple{N, Diagonal{T, Vector{T}}}`: Tuple of diagonal matrices with the weights on the
  diagonal for each dimension.

# Mathematical Description
The quadrature matrix \( Q \) for each dimension is a diagonal matrix where the diagonal
elements are the quadrature weights \( w_i \): \[ Q_{ii} = w_i \]

For multi-dimensional integration, the quadrature matrices are created for each dimension
separately and returned as a tuple.
"""
function quadrature_matrix_nd(weights_tuple::NTuple{N, AbstractVector}) where {N}
	return ntuple(i -> quadrature_matrix(weights_tuple[i]), N)
end

@doc raw"""
	create_node_map(standard_cell::StandardKSCell{T, N}, start_index::Int) where {T <: Number, N} -> Dict{NTuple{N, Int}, Int}

Create a node map for a standard cell starting from a given index.

# Arguments
- `standard_cell::StandardKSCell{T, N}`: The standard cell containing the nodes.
- `start_index::Int`: The starting index for the node map.

# Returns
- `Dict{NTuple{N, Int}, Int}`: A dictionary mapping multi-dimensional indices to linear
  indices.

# Mathematical Description
The node map is created by iterating over all possible combinations of node indices in each
dimension. The multi-dimensional indices are mapped to linear indices starting from
`start_index`.

The multi-dimensional indices are generated using the Cartesian product of the node indices
in each dimension: \[ \text{indices} =
\text{product}(1:\text{length}(\text{nodes\_with\_boundary}[d]) \, \text{for} \, d \,
\text{in} \, 1:N) \]

The node map is then created by assigning each multi-dimensional index to a linear index: \[
\text{node\_map}[\text{multi\_idx}] = \text{start\_index} + \text{idx} - 1 \]
"""
function create_node_map(
	standard_cell::StandardKSCell{T, N},
	start_index::Int,
) where {T <: Number, N}
	node_map = Dict{NTuple{N, Int}, Int}()
	indices = Iterators.product(
		ntuple(
			d -> 1:length(standard_cell.nodes_with_boundary[d]), N)...,
	)
	for (idx, multi_idx) in enumerate(indices)
		node_map[NTuple{N, Int}(multi_idx)] = start_index + idx - 1
	end
	return node_map
end
@doc raw"""
	enforce_ck_continuity_spectral!(D::AbstractMatrix, k::Int, nodes::AbstractVector) -> AbstractMatrix

Enforce C^k continuity on the spectral differentiation matrix.

# Arguments
- `D::AbstractMatrix`: The spectral differentiation matrix.
- `k::Int`: The order of continuity to enforce.
- `nodes::AbstractVector`: The nodes at which the differentiation matrix is defined.

# Returns
- `AbstractMatrix`: The modified differentiation matrix with enforced C^k continuity.

# Throws
- `ArgumentError`: If `k` is less than 0 or greater than or equal to the number of nodes
  minus 1.

# Mathematical Description
C^k continuity is enforced by modifying the first `k` rows and the last `k` rows of the
differentiation matrix. For the first `k` rows, the entries are set to ensure that the
polynomial interpolant matches the function values and derivatives up to order `k` at the
first node. For the last `k` rows, the entries are set to ensure the same at the last node.

For the first `k` rows: \[ D[i, j] = \frac{nodes[i]^{j-1}}{(j-1)!} \quad \text{for} \quad i =
1, \ldots, k \quad \text{and} \quad j = 1, \ldots, k+1 \]

For the last `k` rows: \[ D[i, j] = \frac{(nodes[end] - nodes[i])^{j-1}}{(j-1)!} \quad
\text{for} \quad i = n-k+1, \ldots, n \quad \text{and} \quad j = n-k, \ldots, n \]

# Logical Operations
- Check if `k` is valid: \[ \text{if} \quad k < 0 \quad \text{or} \quad k + 1 \geq n \quad
  \text{throw} \quad \text{ArgumentError} \]
- Set the first `k` rows: \[ D[i, :] \leftarrow 0.0 \quad \text{for} \quad i = 1, \ldots, k
  \] \[ D[i, j] = \frac{nodes[i]^{j-1}}{(j-1)!} \quad \text{for} \quad j = 1, \ldots, k+1 \]
- Set the last `k` rows: \[ D[i, :] \leftarrow 0.0 \quad \text{for} \quad i = n-k+1, \ldots,
  n \] \[ D[i, j] = \frac{(nodes[end] - nodes[i])^{j-1}}{(j-1)!} \quad \text{for} \quad j =
  n-k, \ldots, n \]
"""
function enforce_ck_continuity_spectral!(D::AbstractMatrix, k::Int, nodes::AbstractVector)
	n = size(D, 1)
	if k < 0 || k + 1 >= n
		throw(ArgumentError("Invalid value for k"))
	end

	# For first k rows (left boundary)
	for i in 1:k
		D[i, :] .= 0.0
		for j in 1:(k + 1)
			# Use nodes[i] directly for the left boundary
			D[i, j] = nodes[i]^(j - 1) / factorial(j - 1)
		end
	end

	# For last k rows (right boundary)
	for i in (n - k + 1):n
		D[i, :] .= 0.0
		for j in 1:(k + 1)
			# Use (nodes[end] - nodes[i]) for the right boundary
			D[i, end - k + j - 1] = (nodes[end] - nodes[i])^(j - 1) / factorial(j - 1)
		end
	end

	return D
end

@doc raw"""
	enforce_ck_continuity_bspline!(D::AbstractMatrix, k::Int, nodes::AbstractVector, degree::Int) -> AbstractMatrix

Enforce C^k continuity on the B-spline differentiation matrix.

# Arguments
- `D::AbstractMatrix`: The B-spline differentiation matrix.
- `k::Int`: The order of continuity to enforce.
- `nodes::AbstractVector`: The nodes at which the differentiation matrix is defined.
- `degree::Int`: The degree of the B-spline basis functions.

# Returns
- `AbstractMatrix`: The modified differentiation matrix with enforced C^k continuity.

# Throws
- `ArgumentError`: If `k` is less than 0 or greater than or equal to the number of nodes.

# Mathematical Description
C^k continuity is enforced by modifying the first `k + 1` rows and the last `k` rows of the
differentiation matrix. For the first `k + 1` rows, the entries are set to ensure that the
B-spline basis functions match the function values and derivatives up to order `k` at the
first node. For the last `k` rows, the entries are set to ensure the same at the last node.

For the first `k + 1` rows: \[ D[i, j] = B[i](x_j) \cdot \frac{x_j^{i-1}}{(i-1)!} \quad
\text{for} \quad i = 1, \ldots, k+1 \quad \text{and} \quad j = 1, \ldots, n \]

For the last `k` rows: \[ D[i, j] = \frac{x_i^{j-1}}{(j-1)!} \quad \text{for} \quad i =
n-k+1, \ldots, n \quad \text{and} \quad j = 1, \ldots, \min(k+1, n) \]

# Logical Operations
- Check if `k` is valid: \[ \text{if} \quad k < 0 \quad \text{or} \quad k \geq n \quad
  \text{throw} \quad \text{ArgumentError} \]
- Set the first `k + 1` rows: \[ D[i, :] \leftarrow 0.0 \quad \text{for} \quad i = 1, \ldots,
  k+1 \] \[ D[i, j] = B[i](x_j) \cdot \frac{x_j^{i-1}}{(i-1)!} \quad \text{for} \quad j = 1,
  \ldots, n \]
- Set the last `k` rows: \[ D[i, :] \leftarrow 0.0 \quad \text{for} \quad i = n-k+1, \ldots,
  n \] \[ D[i, idx] = \frac{x_i^{j-1}}{(j-1)!} \quad \text{for} \quad j = 1, \ldots,
  \min(k+1, n) \]
"""
function enforce_ck_continuity_bspline!(
	D::AbstractMatrix,
	k::Int,
	nodes::AbstractVector,
	degree::Int,
)
	n = size(D, 1)
	if k < 0 || k >= n
		throw(ArgumentError("Invalid value for k"))
	end

	# Use fully qualified names with BSplineKit
	knots = BSplineKit.KnotVector(BSplineKit.UniformKnotInterval(degree), length(nodes))
	basis = BSplineKit.BSplineBasis(BSplineKit.BSplineOrder(degree + 1), knots)

	# Rest of implementation
	for i in 1:(k + 1)
		D[i, :] .= 0.0
		for j in eachindex(nodes)
			B = BSplineKit.evaluate(basis, nodes[j])  # Use fully qualified evaluate
			D[i, j] = B[i] * (nodes[j]^(i - 1)) / factorial(i - 1)
		end
	end

	for i in (n - k + 1):n
		D[i, :] .= 0.0
		for j in 1:(k + 1)
			D[i, end - k + j - 1] = (nodes[end] - nodes[i])^(j - 1) / factorial(j - 1)
		end
	end

	return D
end

@doc raw"""
	create_basis_functions(p::Int, continuity_order::Int) -> Vector{Function}

Create basis functions for a given polynomial order and continuity order.

# Arguments
- `p::Int`: Polynomial order.
- `continuity_order::Int`: Continuity order.

# Returns
- `Vector{Function}`: Vector of basis functions.

# Mathematical Description
The basis functions are created using Lagrange interpolation. For each node \( x_i \), the
basis function \( \phi_i(x) \) is defined as: \[ \phi_i(x) = \frac{\prod_{j \neq i} (x -
x_j)}{\prod_{j \neq i} (x_i - x_j)} \]

The numerator is the product of the differences between \( x \) and all other nodes \( x_j
\): \[ \text{numerator} = \prod_{j \neq i} (x - x_j) \]

The denominator is the product of the differences between \( x_i \) and all other nodes \(
x_j \): \[ \text{denominator} = \prod_{j \neq i} (x_i - x_j) \]

The basis function is then given by: \[ \phi_i(x) =
\frac{\text{numerator}}{\text{denominator}} \]
"""
function create_basis_functions(p::Int, continuity_order::Int)
	nodes, _, _, _ = create_legendre_nodes_and_weights(p)
	basis_functions = Vector{Function}(undef, p + 2)

	for i in 1:(p + 2)
		basis_functions[i] =
			x -> begin
				numerator = prod((x - nodes[j]) for j in 1:(p + 2) if j != i)
				denominator = prod((nodes[i] - nodes[j]) for j in 1:(p + 2) if j != i)
				return numerator / denominator
			end
	end

	return basis_functions
end
@doc raw"""
	Lagrange_polynomials(x::AbstractVector{T}, nodes::AbstractVector{T}, i::Int) where T <: Real -> AbstractVector{T}

Compute the Lagrange polynomial \( L_i(x) \) at given points for a specified node.

# Arguments
- `x::AbstractVector{T}`: Points at which to evaluate the polynomial.
- `nodes::AbstractVector{T}`: Nodes of the interpolation.
- `i::Int`: Index of the node for which to compute the polynomial.

# Returns
- `AbstractVector{T}`: Values of the Lagrange polynomial at the given points.

# Throws
- `ArgumentError`: If `i` is not a valid node index.

# Mathematical Description
The Lagrange polynomial \( L_i(x) \) is defined as: \[ L_i(x) = \prod_{j \neq i} \frac{x -
x_j}{x_i - x_j} \]

The polynomial is computed by iterating over all nodes \( x_j \) except \( x_i \) and
multiplying the terms: \[ \text{result} = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j} \]

# Logical Operations
- Check if `i` is valid: \[ \text{if} \quad i < 1 \quad \text{or} \quad i > n \quad
  \text{throw} \quad \text{ArgumentError} \]
- Compute the polynomial: \[ \text{result} \leftarrow 1.0 \] \[ \text{for} \quad j \neq i
  \quad \text{result} \leftarrow \text{result} \cdot \frac{x - x_j}{x_i - x_j} \]
"""
function Lagrange_polynomials(
	x::AbstractVector{T},
	nodes::AbstractVector{T},
	i::Int,
) where {T <: Real}
	n = length(nodes)
	if i < 1 || i > n
		throw(ArgumentError("Invalid index i. Must be between 1 and $(n)."))
	end

	result = ones(T, length(x))
	for j in 1:n
		if j != i
			result .*= (x .- nodes[j]) ./ (nodes[i] - nodes[j])
		end
	end

	return result
end

# ToDo: Make clear that this scaling is fictitious domain and level scaling

@doc raw"""
	determine_scaling_factor(a::Real, b::Real) -> Tuple{Real, Real}

Determine the scaling factors for differentiation and quadrature.

# Arguments
- `a::Real`: Lower bound of the interval.
- `b::Real`: Upper bound of the interval.

# Returns
- `Tuple{Real, Real}`: Differentiation scaling factor and quadrature scaling factor.

# Throws
- `ArgumentError`: If `a` is not less than `b`.

# Mathematical Description
The scaling factors are determined based on the interval \([a, b]\). The differentiation
scaling factor is given by: \[ \text{diff\_scaling} = \frac{2}{b - a} \]

The quadrature scaling factor is given by: \[ \text{quad\_scaling} = \frac{b - a}{2} \]

# Logical Operations
- Check if `a` is less than `b`: \[ \text{if} \quad a \geq b \quad \text{throw} \quad
  \text{ArgumentError} \]
- Compute the scaling factors: \[ \text{diff\_scaling} = \frac{2}{b - a} \] \[
  \text{quad\_scaling} = \frac{b - a}{2} \]
"""
function determine_scaling_factor(a::Real, b::Real)
	if a >= b
		throw(ArgumentError("Invalid arguments: a must be less than b"))
	end
	diff_scaling = 2 / (b - a)
	quad_scaling = (b - a) / 2
	return diff_scaling, quad_scaling
end

# ToDo: Make clear that this scaling is fictitious domain and level scaling

@doc raw"""
	apply_scaling_factors!(D::AbstractMatrix, Q::AbstractMatrix, diff_scaling::Real, quad_scaling::Real) -> Tuple{AbstractMatrix, AbstractMatrix}

Apply scaling factors to the differentiation and quadrature matrices.

# Arguments
- `D::AbstractMatrix`: Differentiation matrix.
- `Q::AbstractMatrix`: Quadrature matrix.
- `diff_scaling::Real`: Scaling factor for the differentiation matrix.
- `quad_scaling::Real`: Scaling factor for the quadrature matrix.

# Returns
- `Tuple{AbstractMatrix, AbstractMatrix}`: Scaled differentiation and quadrature matrices.

# Mathematical Description
The differentiation matrix \( D \) is scaled by the factor \( \text{diff\_scaling} \): \[ D
\leftarrow D \cdot \text{diff\_scaling} \]

The quadrature matrix \( Q \) is scaled by the factor \( \text{quad\_scaling} \): \[ Q
\leftarrow Q \cdot \text{quad\_scaling} \]

# Logical Operations
- Scale the differentiation matrix: \[ D \leftarrow D \cdot \text{diff\_scaling} \]
- Scale the quadrature matrix: \[ Q \leftarrow Q \cdot \text{quad\_scaling} \]
"""
function apply_scaling_factors!(
	D::AbstractMatrix,
	Q::AbstractMatrix,
	diff_scaling::Real,
	quad_scaling::Real,
)
	D .*= diff_scaling
	Q .*= quad_scaling
	return D, Q
end
@doc raw"""
	create_ocfc_mesh(coord_system::KSCartesianCoordinates{T, N},
					 num_cells::NTuple{N, Int},
					 p_vec::NTuple{N, Int},
					 continuity_order::NTuple{N, Int},
					 dim::Int,
					 max_level::Int) where {T <: Number, N} -> KSMesh{T, N}

Create an octree-based Cartesian finite cell (OCFC) mesh.

# Arguments
- `coord_system::KSCartesianCoordinates{T, N}`: Cartesian coordinate system.
- `num_cells::NTuple{N, Int}`: Number of cells in each dimension.
- `p_vec::NTuple{N, Int}`: Polynomial degrees for each dimension.
- `continuity_order::NTuple{N, Int}`: Continuity order for each dimension.
- `dim::Int`: Number of dimensions.
- `max_level::Int`: Maximum refinement level.

# Returns
- `KSMesh{T, N}`: The created OCFC mesh.

# Mathematical Description
The mesh is created by iterating over each level of refinement and each cell within the
level. For each cell, the following steps are performed:
1. Compute the cell size: \[ \text{cell\_size}[d] = \frac{\text{domain}[d][2] -
\text{domain}[d][1]}{\text{num\_cells}[d]} \]

2. Compute the cell start and end points: \[ \text{cell\_start}[d] = \text{domain}[d][1] +
(\text{multi\_idx}[d] - 1) \cdot \text{cell\_size}[d] \] \[ \text{cell\_end}[d] =
\text{cell\_start}[d] + \text{cell\_size}[d] \]

3. Determine if the cell is on the boundary: \[ \text{is\_boundary} =
\text{any}(\text{multi\_idx}[d] == 1 \, \text{or} \, \text{multi\_idx}[d] ==
\text{num\_cells}[d] \, \text{for} \, d \, \text{in} \, 1:N) \]

4. Create the tensor product mask: \[ \text{mask} =
\text{create\_tensor\_product\_mask}(\text{p\_vec}, \text{continuity\_order},
\text{is\_boundary}, \text{is\_fictitious}, \text{dim}) \]

5. Create the standard cell and node map: \[ \text{standard\_cell} =
\text{get\_or\_create\_standard\_cell}(\text{p\_vec}, \text{level}; \text{continuity\_order}
= \text{continuity\_order}) \] \[ \text{cell\_node\_map} =
\text{create\_node\_map}(\text{standard\_cell}, \text{global\_node\_counter}) \]

6. Update the global node counter: \[ \text{global\_node\_counter} +=
\text{length}(\text{cell\_node\_map}) \]

7. Determine the neighbors of the cell: \[ \text{neighbors}[\text{Symbol}("dim$(d)_neg")] =
\text{if} \, \text{multi\_idx}[d] == 1 \, \text{then} \, -1 \, \text{else} \, \text{cell\_id}
- \text{level\_num\_cells\_vec}[d]^{(d - 1)} \] \[
\text{neighbors}[\text{Symbol}("dim$(d)_pos")] = \text{if} \, \text{multi\_idx}[d] ==
\text{level\_num\_cells\_vec}[d] \, \text{then} \, -1 \, \text{else} \, \text{cell\_id} +
\text{level\_num\_cells\_vec}[d]^{(d - 1)} \]

8. Create the cell instance and update the node map: \[ \text{cell\_instance} =
\text{KSCell}(\text{id} = \text{cell\_id}, \ldots) \] \[ \text{node\_map}[(\text{cell\_id},
\text{local\_idx})] = \text{global\_idx} \]

9. Update the boundary cells if the cell is on the boundary: \[ \text{if} \,
\text{multi\_idx}[d] == 1 \, \text{then} \, \text{side} = \text{Symbol}("left_dim_$d") \,
\text{else} \, \text{side} = \text{Symbol}("right_dim_$d") \] \[
\text{boundary\_cells}[\text{side}] = \text{get}(\text{boundary\_cells}, \text{side},
\text{Int}[]) \] \[ \text{push!}(\text{boundary\_cells}[\text{side}], \text{cell\_id}) \]

# Logical Operations
- Iterate over each level and each cell within the level.
- Compute the cell size, start, and end points.
- Determine if the cell is on the boundary.
- Create the tensor product mask, standard cell, and node map.
- Update the global node counter.
- Determine the neighbors of the cell.
- Create the cell instance and update the node map.
- Update the boundary cells if the cell is on the boundary.
"""
function create_ocfc_mesh(
	coord_system::KSCartesianCoordinates{T, N},
	num_cells::NTuple{N, Int},
	p_vec::NTuple{N, Int},
	continuity_order::NTuple{N, Int},
	dim::Int,
	max_level::Int,
) where {T <: Number, N}
	cells = Vector{KSCell{T, N}}()
	global_node_counter = 1
	cell_id = 1
	node_map = Dict{Tuple{Int, NTuple{N, Int}}, Int}()
	boundary_cells = Dict{Symbol, Vector{Int}}()

	for level in 1:max_level
		level_num_cells_vec = num_cells

		cell_sizes = ntuple(
			d ->
				(coord_system.domains[d][2] - coord_system.domains[d][1]) /
				level_num_cells_vec[d],
			N,
		)

		for idx in Iterators.product(ntuple(d -> 1:level_num_cells_vec[d], N)...)
			multi_idx = NTuple{N, Int}(idx)
			cell_start = ntuple(
				d -> coord_system.domains[d][1] + (multi_idx[d] - 1) * cell_sizes[d],
				N,
			)
			cell_end = ntuple(d -> cell_start[d] + cell_sizes[d], N)
			cell_domain = ntuple(d -> (cell_start[d], cell_end[d]), N)

			is_boundary = any(
				multi_idx[d] == 1 || multi_idx[d] == level_num_cells_vec[d] for d in 1:N
			)
			is_fictitious = false

			mask = create_tensor_product_mask(
				p_vec,
				continuity_order,
				is_boundary,
				is_fictitious,
				dim,
			)

			standard_cell = get_or_create_standard_cell(
				p_vec,
				level;
				continuity_order = continuity_order,
			)

			cell_node_map = create_node_map(standard_cell, global_node_counter)
			global_node_counter += length(cell_node_map)

			neighbors = Dict{Symbol, Int}()
			for d in 1:N
				if multi_idx[d] == 1
					neighbors[Symbol("dim$(d)_neg")] = -1  # Boundary
				else
					neighbors[Symbol("dim$(d)_neg")] =
						cell_id -
						level_num_cells_vec[d]^(d - 1)
				end
				if multi_idx[d] == level_num_cells_vec[d]
					neighbors[Symbol("dim$(d)_pos")] = -1  # Boundary
				else
					neighbors[Symbol("dim$(d)_pos")] =
						cell_id +
						level_num_cells_vec[d]^(d - 1)
				end
			end

			# Convert StandardKSCell properties to KSCell properties
			cell_instance = KSCell{T, N}(
				cell_id,                # id
				p_vec,                 # p
				level,                 # level
				continuity_order,      # continuity_order
				(p_vec, level),       # standard_cell_key
				neighbors,            # neighbors
				cell_node_map,        # node_map
				mask,                 # tensor_product_mask
				Dict{Symbol, Int}(),  # boundary_connectivity
				zero(T),             # error_estimate
				zero(T),             # legendre_decay_rate
				true,                # is_leaf
				is_fictitious,       # is_fictitious
				nothing,             # refinement_options
				nothing,              # parent_id
				nothing,              # child_ids
			)

			push!(cells, cell_instance)

			for (local_idx, global_idx) in cell_node_map
				node_map[(cell_id, local_idx)] = global_idx
			end

			if is_boundary
				for d in 1:N
					if multi_idx[d] == 1
						side = Symbol("left_dim_$d")
						boundary_cells[side] = get(boundary_cells, side, Int[])
						push!(boundary_cells[side], cell_id)
					elseif multi_idx[d] == level_num_cells_vec[d]
						side = Symbol("right_dim_$d")
						boundary_cells[side] = get(boundary_cells, side, Int[])
						push!(boundary_cells[side], cell_id)
					end
				end
			end

			cell_id += 1
		end
	end

	mesh = KSMesh(;
		cells = cells,
		global_error_estimate = zero(T),
		boundary_cells = boundary_cells,
		physical_domain = x -> true,
	)

	return mesh
end
@doc raw"""
	derivative_matrix_nd(polynomial_degree::AbstractVector{Int}, dim::Int, k_max::Union{Int, AbstractVector{Int}}) -> Vector{Matrix{Float64}}

Compute the multi-dimensional derivative matrices for given polynomial degrees and
dimensions.

# Arguments
- `polynomial_degree::AbstractVector{Int}`: Polynomial degrees for each dimension.
- `dim::Int`: Number of dimensions.
- `k_max::Union{Int, AbstractVector{Int}}`: Maximum derivative order for each dimension.

# Returns
- `Vector{Matrix{Float64}}`: Vector of derivative matrices for each dimension.

# Throws
- `ArgumentError`: If the dimension does not match the number of polynomial degrees.

# Mathematical Description
The derivative matrices are computed for each dimension separately. For each dimension \( d
\), the kth derivative matrix \( D^{(k)} \) is computed using the function
`kth_derivative_matrix!`.

The kth derivative matrix \( D^{(k)} \) is computed iteratively using the first derivative
matrix \( D \): \[ D^{(k)} = (D^{(k-1)}) \cdot D \]

# Logical Operations
- Check if the dimension matches the number of polynomial degrees: \[ \text{if} \quad
  \text{dim} \neq \text{length}(\text{polynomial\_degree}) \quad \text{throw} \quad
  \text{ArgumentError} \]
- Initialize the derivative matrices: \[ \text{derivative\_matrices} =
  [\text{Vector{Matrix}}(\text{undef}, \text{k\_max}[d]) \quad \text{for} \quad d \quad
  \text{in} \quad 1:\text{dim}] \]
- Compute the kth derivative matrices for each dimension: \[ \text{for} \quad d \quad
  \text{in} \quad 1:\text{dim} \] \[ \quad \text{nodes\_d} = \text{polynomial\_degree}[d] \]
  \[ \quad \text{kth\_matrices} = \text{kth\_derivative\_matrix!}(\text{nodes\_d},
  \text{k\_max}[d]) \] \[ \quad \text{derivative\_matrices}[d] = \text{kth\_matrices}[1]
  \quad \text{Use the matrices with boundary} \]
"""
function derivative_matrix_nd(
	polynomial_degree::AbstractVector{Int},
	dim::Int,
	k_max::Union{Int, AbstractVector{Int}},
)
	dim == length(polynomial_degree) || throw(
		ArgumentError(
			"Dimension 'dim' must match the number of dimensions in 'polynomial_degree'."
		),
	)
	k_max = isa(k_max, Int) ? fill(k_max, dim) : k_max

	derivative_matrices = [Vector{Matrix}(undef, k_max[d]) for d in 1:dim]
	for d in 1:dim
		nodes_d = polynomial_degree[d]
		kth_matrices = kth_derivative_matrix!(nodes_d, k_max[d])
		derivative_matrices[d] = kth_matrices[1]  # Use the matrices with boundary
	end

	return derivative_matrices
end
@doc raw"""
	compute_quadrature_weight(mesh::KSMesh{T, N}, global_idx::Int) where {T <: Number, N} -> T

Compute the quadrature weight for a given global node index in the mesh.

# Arguments
- `mesh::KSMesh{T, N}`: The mesh containing the cells.
- `global_idx::Int`: The global node index.

# Returns
- `T`: The quadrature weight for the given global node index.

# Throws
- `ArgumentError`: If the global node index is not found in any cell's node map.

# Mathematical Description
The quadrature weight \( w \) for a given global node index is determined by iterating
through the cells in the mesh and finding the corresponding local node index. The quadrature
weight is then retrieved from the cell's spectral properties.

# Logical Operations
- Iterate over each cell in the mesh: \[ \text{for each cell in mesh.cells} \]
- Iterate over the node map of the cell to find the global node index: \[ \text{for each
  (local_coords, g_idx) in cell.node_map} \] \[ \quad \text{if} \quad g_idx == global_idx \]
  \[ \quad \quad \text{for each (local_idx, node_g_idx) in enumerate(values(cell.node_map))}
  \] \[ \quad \quad \quad \text{if} \quad node_g_idx == g_idx \] \[ \quad \quad \quad \quad
  \text{return} \quad cell.spectral_properties.quadrature_weights[local_idx] \]
- Throw an error if the global node index is not found: \[ \text{throw} \quad
  \text{ArgumentError("Global node index $global_idx not found in any cell's node_map.")} \]
"""
function compute_quadrature_weight(
	mesh::KSMesh{T, N}, global_idx::Int) where {T <: Number, N}
	for cell in mesh.cells
		for (local_coords, g_idx) in cell.node_map
			if g_idx == global_idx
				# Get standard cell from cache using the cell's standard_cell_key
				standard_cell = get_or_create_standard_cell(cell.standard_cell_key...)

				# Find corresponding weight index
				for (local_idx, node_g_idx) in enumerate(values(cell.node_map))
					if node_g_idx == g_idx
						# Return quadrature weight from standard cell
						return standard_cell.weights_with_boundary[1][local_idx]
					end
				end
			end
		end
	end
	throw(ArgumentError("Global node index $global_idx not found in any cell's node_map."))
end

function compute_quadrature_weight(
	cell_key::Tuple{NTuple{N, Int}, Int},
	quad_point::CartesianIndex) where N

	# Get standard cell to access weights
	standard_cell = get_or_create_standard_cell(cell_key...)

	# Compute tensor product weight
	weight = one(eltype(standard_cell.weights_with_boundary[1]))
	for d in 1:N
		idx = quad_point[d]
		if idx <= length(standard_cell.weights_with_boundary[d])
			weight *= standard_cell.weights_with_boundary[d][idx]
		end
	end

	return weight
end
@doc raw"""
	get_node_coordinates(coord_system::KSCartesianCoordinates{T, N}, cell::KSCell{T, N}, node_index::NTuple{N, Int}) where {T <: Real, N} -> NTuple{N, T}

Get the coordinates of a node within a cell.

# Arguments
- `coord_system::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.
- `cell::KSCell{T, N}`: The cell containing the node.
- `node_index::NTuple{N, Int}`: The multi-dimensional index of the node within the cell.

# Returns
- `NTuple{N, T}`: The coordinates of the node.

# Mathematical Description
The coordinates of a node within a cell are determined by the cell's position in the mesh and
the node's position within the cell. The steps are:
1. Calculate the number of cells per dimension.
2. Determine the cell's multi-dimensional index based on its ID.
3. Calculate the cell size in each dimension.
4. Calculate the start coordinates of the cell.
5. Calculate the node coordinates within the cell.

# Logical Operations
- Calculate the number of cells per dimension: \[ \text{num\_cells\_per\_dim} =
  \text{coord\_system.num\_cells} \]
- Determine the cell's multi-dimensional index: \[ \text{multi\_idx} = \text{ntuple}(d
  \rightarrow \text{div}(\text{rem}(cell.id - 1,
  \text{prod}(\text{num\_cells\_per\_dim}[1:d])),
  \text{prod}(\text{num\_cells\_per\_dim}[1:(d - 1)])) + 1, N) \]
- Calculate the cell size in each dimension: \[ \text{cell\_sizes} = \text{ntuple}(d
  \rightarrow (\text{cell\_corners}[d][2] - \text{cell\_corners}[d][1]) /
  \text{num\_cells\_per\_dim}[d], N) \]
- Calculate the start coordinates of the cell: \[ \text{cell\_start} = \text{ntuple}(d
  \rightarrow \text{coord\_system.domains}[d][1] + (\text{multi\_idx}[d] - 1) *
  \text{cell\_sizes}[d], N) \]
- Calculate the node coordinates within the cell: \[ \text{node\_coords} = \text{ntuple}(d
  \rightarrow \text{cell\_start}[d] + (\text{node\_index}[d] - 1) * \text{cell\_sizes}[d] /
  (p[d] + 1), N) \]
"""
function get_node_coordinates(
	coord_system::KSCartesianCoordinates{T, N},
	cell::KSCell{T, N},
	point_or_index::Union{AbstractVector{T}, NTuple{N, T}, Tuple},
	p_or_num_cells::NTuple{N, Int}) where {T, N}
	# Calculate cell position based on cell ID
	cell_position = if N == 1
		(cell.id,)
	else
		ntuple(d -> begin
				denominator = 1
				for i in 1:(d - 1)
					denominator *= (cell.p[i] + 2)
				end
				div((cell.id - 1), denominator) % (cell.p[d] + 2)
			end, N)
	end

	# Convert point to tuple if it's a vector
	point_tuple =
		isa(point_or_index, AbstractVector) ? tuple(point_or_index...) : point_or_index

	# Calculate cell start coordinates
	cell_start = ntuple(
		d -> begin
			domain_size = coord_system.domains[d][2] - coord_system.domains[d][1]
			coord_system.domains[d][1] +
			cell_position[d] * domain_size / (p_or_num_cells[d] + 2)
		end, N)

	# If point_or_index is a node index, calculate node coordinates
	if length(point_tuple) == N && all(x -> isa(x, Integer), point_tuple)
		cell_size = ntuple(
			d -> begin
				domain_size = coord_system.domains[d][2] - coord_system.domains[d][1]
				domain_size / (p_or_num_cells[d] + 2)
			end, N)

		return ntuple(
			d ->
				cell_start[d] +
				(point_tuple[d] - 1) * cell_size[d] / (p_or_num_cells[d] + 1),
			N,
		)
	end

	# If point_or_index is a point, return it as is
	return point_tuple
end
function get_node_coordinates(
	coord_system::KSCartesianCoordinates{T, N},
	cell::KSCell{T, N},
	point::AbstractVector{T},
	num_cells::NTuple{N, Int},
) where {T, N}
	# Calculate cell indices
	cell_indices = ntuple(
		d -> begin
			# Use proper cell indexing based on cell ID
			idx = div(cell.id - 1, prod(num_cells[1:(d - 1)])) % num_cells[d] + 1
			idx
		end, N)

	# Calculate cell sizes
	cell_sizes = ntuple(
		d -> begin
			domain_size = coord_system.domains[d][2] - coord_system.domains[d][1]
			domain_size / num_cells[d]
		end, N)

	# Calculate cell bounds
	cell_starts = ntuple(
		d -> begin
			coord_system.domains[d][1] + (cell_indices[d] - 1) * cell_sizes[d]
		end, N)

	cell_ends = ntuple(d -> cell_starts[d] + cell_sizes[d], N)

	# Map point to reference coordinates [-1, 1]
	reference_coords = ntuple(
		d -> begin
			# Linear mapping from physical to reference coordinates
			2.0 * (point[d] - cell_starts[d]) / cell_sizes[d] - 1.0
		end, N)

	return reference_coords
end
@doc raw"""
	interpolate_in_cell(cell::KSCell{T, N}, u::AbstractVector{T}, point::NTuple{N, T}, mesh::KSMesh{T, N}) where {T <: Real, N} -> T

Interpolate the solution at a given point within a cell.

# Arguments
- `cell::KSCell{T, N}`: The cell containing the solution.
- `u::AbstractVector{T}`: Vector of solution values at the cell nodes.
- `point::NTuple{N, T}`: The point at which to interpolate.
- `mesh::KSMesh{T, N}`: The mesh containing the cell.

# Returns
- `T`: Interpolated value at the given point.

# Mathematical Description
The interpolation is performed using barycentric interpolation within the standard cell. The
- Reshape the solution values to match the standard cell structure: \[ \text{cell\_values} =
  \text{reshape}(@\text{view}(\text{u}[\text{collect}(\text{values}(\text{cell.node_map}))]),
  \text{ntuple}(i \rightarrow \text{cell.p}[i] + 2, N)) \]
- Perform the interpolation using the `interpolate_nd` function: \[ \text{result} =
  \text{interpolate_nd}(\text{standard\_cell}, \text{cell\_values}, \text{local\_coords}) \]
"""
function interpolate_in_cell(
	cell::KSCell{T, N},
	u::AbstractVector{T},
	point::NTuple{N, T},
	mesh::KSMesh{T, N},
) where {T, N}
	standard_cell = get_or_create_standard_cell(
		cell.p,
		cell.level;
		continuity_order = cell.continuity_order,
	)
	local_coords = get_node_coordinates(
		mesh.coord_system, cell, point, mesh.num_cells_per_dim
	)
	cell_values = reshape(
		@view(u[collect(values(cell.node_map))]), ntuple(i -> cell.p[i] + 2, N))
	return interpolate_nd(standard_cell, cell_values, local_coords)
end

# Fix get_node_coordinates to handle both mesh and node inputs
function get_node_coordinates(node_idx::Int, mesh::KSMesh{T,N}) where {T,N}
    # Find cell containing this node
    for cell in mesh.cells
        if haskey(cell.node_map, node_idx)
            # Get the standard cell
            std_cell = get_or_create_standard_cell(cell.standard_cell_key...)

            # Get local coordinates
            local_coords = findfirst(==(node_idx), values(cell.node_map))

            # Map to physical coordinates
            return get_node_coordinates(mesh.coord_system, cell, local_coords, mesh.num_cells_per_dim)
        end
    end
    throw(ArgumentError("Node index $node_idx not found in mesh"))
end

# Add function to get boundary nodes from mesh
function get_boundary_nodes(mesh::KSMesh{T,N}) where {T,N}
    boundary_nodes = Set{Int}()
    for (side, cells) in mesh.boundary_cells
        for cell_id in cells
            cell = mesh.cells[cell_id]
            # Add nodes from boundary side
            for (local_idx, global_idx) in cell.node_map
                if is_boundary_node(local_idx, side, cell.p)
                    push!(boundary_nodes, global_idx)
                end
            end
        end
    end
    return sort(collect(boundary_nodes))
end

function is_boundary_node(local_idx::NTuple{N,Int}, side::Symbol, p::NTuple{N,Int}) where N
    dim, direction = parse_direction(side)
    return direction == :pos ? local_idx[dim] == p[dim] + 2 : local_idx[dim] == 1
end

end # module SpectralMethods
