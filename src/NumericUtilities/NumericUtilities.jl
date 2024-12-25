module NumericUtilities

# Update imports to explicitly import nnz
using LinearAlgebra, StaticArrays, SparseArrays, FastGaussQuadrature, Statistics
using ..KSTypes
import SparseArrays: nnz

# Remove exports that are no longer needed
export regularize_matrix!, regularize_sparse_matrix!, estimate_condition_number
export estimate_condition_number_power_method, compute_stability_metrics
export estimate_sparse_condition_number, validate_numerical_computation
export is_diagonally_dominant, diagdom!, analyze_matrix_properties
export compute_base_matrix_properties, determine_best_algorithm_condition_number
export verify_matrix_property, get_matrix_property, ensure_property_exists
export verify_matrix_dimensions, get_derivative_coefficients, verify_matrix_property
export compute_stability_metrics, is_well_conditioned, validate_numerical_computation
export is_numerically_stable, check_metrics_stability
export is_reasonable_sparsity, has_valid_block_structure
export count_nonzeros  # Add to exports

"""
	count_nonzeros(A::AbstractMatrix) -> Int

Count the number of non-zero elements in a matrix efficiently.

For sparse matrices, uses the built-in nnz function.
For dense matrices, counts elements that are not exactly zero.
"""
function count_nonzeros(A::AbstractMatrix)
	if issparse(A)
		return nnz(A)
	else
		return count(!iszero, A)
	end
end

"""
	get_matrix_property(props::Dict{Symbol,Any}, key::Symbol, default::Any=false)
Safe accessor for matrix properties with default value.
"""
function get_matrix_property(props::Dict{Symbol, Any}, key::Symbol, default::Any = false)
	return get(props, key, default)
end

"""
	ensure_property_exists(props::Dict{Symbol,Any})
Ensures all required properties exist in the dictionary.
"""
function ensure_property_exists(props::Dict{Symbol, Any})
	required = [:size, :sparse, :symmetric, :diag_dominant, :posdef,
		:ill_conditioned, :nearly_singular, :cond_num]
	for key in required
		if !haskey(props, key)
			props[key] = key == :size ? 0 : false
		end
	end
	return props
end

"""
	compute_base_matrix_properties(A::AbstractMatrix{T}) where T
Compute basic matrix properties that are commonly needed across functions.
Internal helper function to reduce redundant computations.
"""
function compute_base_matrix_properties(A::AbstractMatrix{T}) where T
	props = Dict{Symbol, Any}()

	# Basic properties
	n = size(A, 1)
	props[:size] = n
	props[:sparse] = issparse(A)
	props[:norm_1] = opnorm(A, 1)
	props[:norm_inf] = opnorm(A, Inf)

	# Structure properties
	props[:symmetric] = issymmetric(A)
	props[:diag_dominant] = is_diagonally_dominant(A)

	# Additional properties for sparse matrices
	if props[:sparse]
		props[:sparsity] = nnz(A) / (n * n)
	end

	# Ensure required properties are set
	props = ensure_property_exists(props)

	return props
end

"""
	analyze_matrix_properties(A::AbstractMatrix{T}) where T
Returns properties to guide algorithm selection with improved integration.
"""

function analyze_matrix_properties(A::AbstractMatrix{T}) where T
	props = Dict{Symbol, Any}()
	n = size(A, 1)

	# Check for empty or invalid matrix
	if n == 0 || n != size(A, 2)
		throw(ArgumentError("Matrix must be non-empty and square"))
	end

	# Type-specific tolerances
	tol = sqrt(eps(real(T)))
	T_eps = eps(real(T))

	# Basic properties
	props[:size] = n
	props[:sparse] = issparse(A)

	# Compute norms safely
	try
		props[:norm_inf] = norm(A, Inf)
		props[:norm_one] = norm(A, 1)
	catch
		# Fallback for matrices where norm fails
		props[:norm_inf] = maximum(sum(abs, A; dims = 1))
		props[:norm_one] = maximum(sum(abs, A; dims = 2))
	end

	# Sparsity analysis
	if props[:sparse]
		nnz_A = nnz(A)
	else
		nnz_A = count(!iszero, A)
	end
	props[:sparsity] = nnz_A / (n * n)

	# Symmetry check with improved accuracy
	if n > 0
		if issparse(A)
			# For sparse matrices, check pattern symmetry first
			pattern_symmetric = SparseMatrixCSC(A) == SparseMatrixCSC(A')
			props[:symmetric] =
				pattern_symmetric && norm(A - A', Inf) <= tol * props[:norm_inf]
		else
			props[:symmetric] = norm(A - A', Inf) <= tol * props[:norm_inf]
		end
	end

	# Condition number estimation with fallbacks
	props[:cond_num] = try
		if props[:sparse]
			estimate_sparse_condition_number(A)
		else
			κ = estimate_condition_number(A)
			# Make estimation more conservative
			if !isfinite(κ) || κ > 1e12
				convert(real(T), Inf)
			else
				κ
			end
		end
	catch
		convert(real(T), Inf)  # Default to worst case
	end

	props[:ill_conditioned] = props[:cond_num] >= 1e6

	# Update criteria for near-singularity
	# A matrix is considered nearly singular if:
	# 1. Condition number > 1e12
	# 2. Contains very small diagonal elements
	# 3. Determinant is close to zero
	props[:nearly_singular] = false
	if props[:cond_num] > 1e12 || !isfinite(props[:cond_num])
		props[:nearly_singular] = true
	else
		# Check diagonal elements
		diag_elements = diag(A)
		matrix_norm = opnorm(A, Inf)
		smallest_diag = minimum(abs.(diag_elements))

		if smallest_diag < eps(real(T)) * matrix_norm
			props[:nearly_singular] = true
		else
			# For small matrices, check determinant
			if size(A, 1) <= 100  # Only for reasonably sized matrices
				try
					det_A = abs(det(A))
					if det_A < eps(real(T)) * matrix_norm^size(A, 1)
						props[:nearly_singular] = true
					end
				catch
					# If det fails, rely on condition number
				end
			end
		end
	end

	# Positive definiteness check with improved robustness
	if props[:symmetric]
		props[:posdef] = try
			if n > 100
				# For large matrices, use Cholesky with a sample
				sample_size = min(10, n)
				principal_submatrix = A[1:sample_size, 1:sample_size]
				if cholesky(Symmetric(principal_submatrix)).info == 0
					# Verify with a few random principal minors
					for _ in 1:3
						idx = sort(randperm(n)[1:sample_size])
						if cholesky(Symmetric(A[idx, idx])).info != 0
							return false
						end
					end
					true
				else
					false
				end
			else
				# For smaller matrices, use direct Cholesky
				cholesky(Symmetric(A)).info == 0
			end
		catch
			false
		end
	else
		props[:posdef] = false
	end

	# Diagonal dominance check with improved efficiency
	diag_abs = abs.(diag(A))
	if props[:sparse]
		# Efficient sparse implementation
		row_sums = zeros(real(T), n)
		cols = rowvals(A)
		vals = nonzeros(A)
		for j ∈ 1:n
			for k in nzrange(A, j)
				i = cols[k]
				if i != j
					row_sums[i] += abs(vals[k])
				end
			end
		end
		props[:diag_dominant] = all(diag_abs .>= row_sums)
	else
		row_sums = sum(abs, A; dims = 2) .- diag_abs
		props[:diag_dominant] = all(diag_abs .>= vec(row_sums))
	end

	# Additional properties
	props[:spd] = props[:symmetric] && props[:posdef]
	props[:suitable_for_cg] = props[:spd] && !props[:ill_conditioned]
	props[:needs_robust_precond] = props[:ill_conditioned] || props[:nearly_singular]
	props[:parallel_favorable] = n > 1000 && Threads.nthreads() > 1

	return props
end

"""
	determine_best_algorithm_condition_number(props::Dict{Symbol,Any})
Determines the best algorithm for condition number estimation based on matrix properties.
"""
function determine_best_algorithm_condition_number(props::Dict{Symbol, <:Any})
	# Input validation
	haskey(props, :size) || throw(ArgumentError("Matrix size not found in properties"))
	n = Int(props[:size])
	n > 0 || throw(ArgumentError("Invalid matrix size"))

	# Fast path for tiny matrices
	if n <= 4
		return :direct
	end

	# Pre-compute commonly used properties with default values
	is_sparse = get(props, :sparse, false)
	is_symmetric = get(props, :symmetric, false)
	is_posdef = get(props, :posdef, false)
	is_diag_dom = get(props, :diag_dominant, false)
	sparsity = get(props, :sparsity, 1.0)

	# Decision tree for algorithm selection
	if n < 100
		return :direct
	elseif is_sparse
		if is_symmetric && is_posdef
			return :cholesky_sparse
		elseif sparsity < 0.1
			return :iterative_sparse
		elseif is_diag_dom
			return :power_iteration
		end
	end

	# Dense matrix specializations
	if is_symmetric
		if is_posdef
			return :cholesky_dense
		elseif is_diag_dom
			return :power_iteration
		end
	end

	return n > 1000 ? :power_iteration : :direct
end

"""
	estimate_condition_number(A::AbstractMatrix{T}; method::Symbol = :direct) where T
Estimates the condition number of a matrix using the specified method.
"""
function estimate_condition_number(A::AbstractMatrix{T}; method::Symbol = :direct) where T
	isempty(A) && throw(ArgumentError("Cannot estimate condition number of empty matrix"))

	# Handle SubArray views by converting to proper matrix type
	if A isa SubArray
		if A.parent isa SparseMatrixCSC
			return estimate_sparse_condition_number(sparse(A))
		else
			return estimate_condition_number(Matrix(A); method = method)
		end
	end

	props = compute_base_matrix_properties(A)

	# Handle sparse matrices separately
	if issparse(A)
		return estimate_sparse_condition_number(A)
	end

	if method == :direct && size(A) == (2, 2)
		return cond(Array(A))
	elseif method == :cholesky_sparse && props[:sparse] && props[:symmetric] &&
		props[:posdef]
		F = cholesky(Symmetric(A))
		return opnorm(F) * opnorm(inv(F))
	elseif method == :power_iteration || (method == :power && props[:size] > 1000)
		return estimate_condition_number_power_method(A)
	else
		return cond(Array(A))
	end
end
"""
	estimate_condition_number_power_method(A::AbstractMatrix{T};
		max_iter::Int = 100,
		tol::Real = sqrt(eps(real(T)))) where T

Estimates the condition number using the power method for large matrices.
Uses power iteration to estimate largest and smallest singular values.

Parameters:
- `A`: Input matrix
- `max_iter`: Maximum number of iterations for power method
- `tol`: Convergence tolerance

Returns:
- Estimated condition number
"""

function estimate_condition_number_power_method(A::AbstractMatrix{T};
	max_iter::Int = 100,
	tol::Real = sqrt(eps(real(T)))) where T
	n = size(A, 1)
	props = compute_base_matrix_properties(A)

	if n <= 4
		return cond(A)
	end

	# For symmetric matrices, use more robust eigenvalue estimation
	if props[:symmetric]
		v1 = normalize!(randn(n))
		v2 = normalize!(randn(n))
		λ_max = zero(real(T))
		λ_min = convert(real(T), Inf)

		# Increase iterations for symmetric indefinite matrices
		actual_max_iter = 2 * max_iter

		# Power iteration for largest eigenvalue
		for _ in 1:actual_max_iter
			v1_new = A * v1
			λ = dot(v1_new, v1)
			v1 = normalize!(v1_new)
			λ_max = max(λ_max, abs(λ))
		end

		# Inverse iteration with shift for smallest eigenvalue
		B = A + λ_max * I / n  # More robust shift
		for _ in 1:actual_max_iter
			v2_new = try
				B \ v2
			catch
				# Fallback to QR if LU fails
				qr(B) \ v2
			end
			v2 = normalize!(v2_new)
			λ = abs(dot(v2, A * v2))
			λ_min = min(λ_min, λ)
		end

		return λ_max / max(λ_min, eps(real(T)) * λ_max)
	end

	# SVD-based estimation for non-symmetric matrices
	v = normalize!(randn(n))
	σ_max = zero(real(T))

	# Power iteration for largest singular value
	for _ in 1:max_iter
		v_new = A * v
		σ_new = norm(v_new)
		if σ_new == 0
			break
		end
		v = normalize!(v_new)
		if abs(σ_max - σ_new) < tol * σ_new
			σ_max = σ_new
			break
		end
		σ_max = max(σ_max, σ_new)
	end

	# Inverse iteration for smallest singular value
	w = normalize!(randn(n))
	σ_min = convert(real(T), Inf)

	# Try inverse iteration with normal equations
	try
		for _ in 1:max_iter
			w_new = (A'A) \ w
			w = normalize!(w_new)
			σ = sqrt(abs(dot(w, A'A * w)))
			if abs(σ_min - σ) < tol * σ
				σ_min = σ
				break
			end
			σ_min = min(σ_min, σ)
		end
	catch
		# Fallback to minimum singular value estimation
		σ_min = try
			svds(A; nsv = 1, which = :SM)[1].S[1]
		catch
			eps(real(T)) * σ_max
		end
	end

	return σ_max / max(σ_min, eps(real(T)) * σ_max)
end

"""
	estimate_sparse_condition_number(A::AbstractMatrix{T}; max_iter::Int=100, tol::Real=sqrt(eps(T))) where T
Handle both sparse matrices and sparse views
"""
function estimate_sparse_condition_number(A::AbstractMatrix{T};
	max_iter::Int = 100, tol::Real = sqrt(eps(T))) where T

	# Handle SubArray views of sparse matrices
	if A isa SubArray && A.parent isa SparseMatrixCSC
		A = sparse(A)  # Convert view to proper sparse matrix
	elseif !(A isa SparseMatrixCSC)
		A = sparse(A)  # Convert to sparse if needed
	end

	n = size(A, 1)
	if n == 0
		return convert(real(T), Inf)
	end

	# Enhanced largest singular value estimation
	v = normalize!(randn(n))
	σ_max = 0.0

	for _ in 1:max_iter
		v_new = A * v
		σ_new = norm(v_new)
		if abs(σ_max - σ_new) < tol * σ_new
			σ_max = σ_new
			break
		end
		σ_max = max(σ_max, σ_new)
		v = normalize!(v_new)
	end

	# Enhanced smallest singular value estimation
	σ_min = Inf
	v = normalize!(randn(n))

	# Try multiple methods for robustness
	methods = [
		:inverse_iteration,
		:normal_equations,
		:direct_solver,
		:power_method,
	]

	for method in methods
		try
			if method == :inverse_iteration
				F = lu(A)
				for _ in 1:max_iter
					v = normalize!(F \ v)
					σ = norm(A * v)
					if abs(σ_min - σ) < tol * σ
						σ_min = σ
						break
					end
					σ_min = min(σ_min, σ)
				end
			elseif method == :normal_equations
				w = A' * (A * v)
				σ = sqrt(norm(w))
				σ_min = min(σ_min, σ)
			elseif method == :direct_solver
				F = factorize(A)
				v = F \ randn(n)
				σ_min = min(σ_min, 1.0 / norm(v))
			else # :power_method fallback
				w = A' * v
				σ = norm(w)
				σ_min = min(σ_min, σ)
			end

			# If we found a good estimate, break the loop
			if σ_min < Inf
				break
			end
		catch
			continue
		end
	end

	# Ensure we don't return too small values that could cause overflow
	σ_min = max(σ_min, eps(real(T)) * σ_max)

	return σ_max / σ_min
end

"""
	regularize_matrix!(matrix::AbstractMatrix{T};
		max_cond::Real = 1e6,
		tol::Real = sqrt(eps(real(T))),
		max_iterations::Int = 100) where T
Enhances the numerical stability of a matrix by regularizing it.
"""
function regularize_matrix!(matrix::AbstractMatrix{T};
	max_cond::Real = 1e6,
	tol::Real = sqrt(eps(real(T))),
	max_iterations::Int = 100) where T

	# Input validation
	size(matrix, 1) == size(matrix, 2) || throw(ArgumentError("Matrix must be square"))
	isempty(matrix) && throw(ArgumentError("Cannot regularize empty matrix"))

	# Handle sparse matrices separately
	if issparse(matrix)
		return regularize_sparse_matrix!(matrix, max_cond, tol, max_iterations)
	end

	props = compute_base_matrix_properties(matrix)
	initial_cond = estimate_condition_number(matrix)

	if initial_cond <= max_cond
		return matrix
	end

	# Use more aggressive initial regularization
	λ = sqrt(eps(T)) * props[:norm_1] * sqrt(props[:size])

	# Store original diagonal
	diag_orig = diag(matrix)

	# Progressive regularization
	best_cond = initial_cond
	best_λ = λ

	iteration = 0
	while iteration < max_iterations
		iteration += 1
		# Apply regularization in-place
		@inbounds for i in 1:props[:size]
			matrix[i, i] = diag_orig[i] + λ
		end

		current_cond = estimate_condition_number(matrix)

		if current_cond < best_cond
			best_cond = current_cond
			best_λ = λ

			if current_cond <= max_cond
				break
			end
		end

		λ *= 2.0

		if λ > props[:norm_1] * 1e6
			break
		end
	end

	# Apply best regularization found
	@inbounds for i in 1:props[:size]
		matrix[i, i] = diag_orig[i] + best_λ
	end

	return matrix
end

"""
	regularize_sparse_matrix!(matrix::SparseMatrixCSC{T},
		max_cond::Real,
		tol::Real,
		max_iterations::Int) where T
Improves the numerical stability of a sparse matrix through in-place operations.
"""
function regularize_sparse_matrix!(matrix::SparseMatrixCSC{T},
	max_cond::Real,
	tol::Real,
	max_iterations::Int) where T
	n = size(matrix, 1)
	if n == 0
		return matrix
	end

	props = compute_base_matrix_properties(matrix)
	λ = tol * max(props[:norm_1], one(T))

	# Store original diagonal for restoration if needed
	orig_diag = diag(matrix)

	# Initial condition number
	κ_init = estimate_sparse_condition_number(matrix)

	if κ_init <= max_cond
		return matrix
	end

	# Track best regularization
	best_κ = κ_init
	best_λ = λ

	iteration = 0
	while iteration < max_iterations
		iteration += 1

		# Modify diagonal in-place
		for i in 1:n
			# Find diagonal element index in sparse structure
			row_start = matrix.colptr[i]
			row_end = matrix.colptr[i + 1] - 1
			diag_idx =
				searchsortedfirst(matrix.rowval[row_start:row_end], i) + row_start - 1

			if diag_idx <= row_end && matrix.rowval[diag_idx] == i
				# Update existing diagonal element
				matrix.nzval[diag_idx] = orig_diag[i] + λ
			else
				# Insert new diagonal element if needed
				insert!(matrix.rowval, diag_idx, i)
				insert!(matrix.nzval, diag_idx, λ)
				for j in (i + 1):(n + 1)
					matrix.colptr[j] += 1
				end
			end
		end

		# Check condition number
		κ = estimate_sparse_condition_number(matrix)

		if κ < best_κ
			best_κ = κ
			best_λ = λ

			if κ <= max_cond
				break
			end
		end

		λ *= 2.0

		# Break if regularization parameter becomes too large
		if λ > props[:norm_1] * 1e6
			# Restore best found regularization
			for i in 1:n
				row_start = matrix.colptr[i]
				row_end = matrix.colptr[i + 1] - 1
				diag_idx =
					searchsortedfirst(matrix.rowval[row_start:row_end], i) + row_start - 1

				if diag_idx <= row_end && matrix.rowval[diag_idx] == i
					matrix.nzval[diag_idx] = orig_diag[i] + best_λ
				end
			end
			break
		end
	end

	return matrix
end

function validate_numerical_computation(A::AbstractMatrix{T},
	b::AbstractVector{T};
	condition_threshold::Real = 1e8) where T
	issues = String[]

	# Basic validation
	if isempty(A)
		push!(issues, "Empty input detected")
		return false, issues
	end
	if size(A, 1) != length(b)
		push!(issues, "Matrix and vector dimensions mismatch")
		return false, issues
	end
	if !all(isfinite, A) || !all(isfinite, b)
		push!(issues, "Non-finite values detected")
		return false, issues
	end

	# Strict diagonal dominance check first
	if !is_diagonally_dominant(A)
		push!(
			issues, "Matrix lacks diagonal dominance - this may cause numerical instability"
		)
	end

	# Condition number check
	κ = estimate_condition_number(A)
	if κ > condition_threshold
		push!(issues, "High condition number detected (κ ≈ $(round(κ, sigdigits=3)))")
	end

	# Return false if any issues were found
	return isempty(issues), issues
end
"""
	is_diagonally_dominant(A::AbstractMatrix)
Checks if matrix A is strictly diagonally dominant.
Returns true if |A[i,i]| > ∑_{j≠i} |A[i,j]| for all i.
"""
function is_diagonally_dominant(A::AbstractMatrix)
	m, n = size(A)
	m == n || return false  # Only square matrices can be diagonally dominant

	for i in 1:m
		# Compute row sum excluding diagonal
		row_sum = sum(j -> j != i ? abs(A[i, j]) : zero(eltype(A)), 1:n)
		if abs(A[i, i]) <= row_sum
			return false
		end
	end
	return true
end

# Update verify_matrix_property to be more robust for posdef check
function verify_matrix_property(A::AbstractMatrix, property::Symbol)
	if property == :diag_dominant || property == :diagonally_dominant
		return is_diagonally_dominant(A)
	elseif property == :symmetric
		return issymmetric(A)
	elseif property == :posdef
		return try
			# First check symmetry
			if !issymmetric(A)
				return false
			end
			# Use Cholesky for better numerical stability
			cholesky(Symmetric(A)).info == 0
		catch
			false
		end
	elseif property == :sparse
		return issparse(A)
	elseif property == :ill_conditioned
		κ = estimate_condition_number(A)
		return κ >= 1e6
	elseif property == :nearly_singular
		κ = estimate_condition_number(A)
		return κ > 1e12
	else
		return false
	end
end

"""
	diagdom!(A::Matrix)
Make a matrix diagonally dominant in place.
"""
function diagdom!(A::Matrix)
	n = size(A, 1)
	n == size(A, 2) || throw(ArgumentError("Matrix must be square"))

	@inbounds for i in 1:n
		row_sum = sum(j -> i != j ? abs(A[i, j]) : zero(eltype(A)), 1:n)
		if abs(A[i, i]) ≤ row_sum
			A[i, i] = sign(A[i, i]) * (row_sum + one(eltype(A)))
		end
	end
	return A
end
"""
	compute_stability_metrics(A::AbstractMatrix{T}, b::AbstractVector{S}=Vector{T}()) where {T <: Number, S <: Number}
Computes various stability metrics for matrix `A` and optionally vector `b`.
"""
function compute_stability_metrics(
	A::AbstractMatrix{T},
	b::AbstractVector{S} = Vector{T}()) where {T <: Number, S <: Number}

	# Get base properties and initialize metrics
	props = compute_base_matrix_properties(A)
	metrics = Dict{Symbol, Float64}()

	if isempty(A)
		metrics[:norm_A] = 0.0
		metrics[:zero_ratio] = 0.0
		return metrics
	end

	# Base matrix metrics
	metrics[:rows], metrics[:cols] = size(A)
	metrics[:is_square] = size(A, 1) == size(A, 2) ? 1.0 : 0.0
	metrics[:sparsity] = get(props, :sparsity, count(!iszero, A) / length(A))
	metrics[:norm_A] = props[:norm_1]
	metrics[:zero_ratio] = count(iszero, A) / length(A)

	# Get nonzero values using SparseArrays.nnz for sparse matrices
	nonzeros_A = if issparse(A)
		[A.nzval[i] for i in 1:nnz(A)]
	else
		filter(!iszero, A)
	end

	# Matrix value range metrics
	metrics[:min_abs] = isempty(nonzeros_A) ? 0.0 : minimum(abs, nonzeros_A)
	metrics[:max_abs] = maximum(abs, A)
	metrics[:dynamic_range] = metrics[:max_abs] / max(metrics[:min_abs], eps(real(T)))

	# Square matrix specific metrics
	if metrics[:is_square] == 1.0
		metrics[:is_symmetric] = props[:symmetric] ? 1.0 : 0.0
		metrics[:diag_dominant] = props[:diag_dominant] ? 1.0 : 0.0
		metrics[:cond_num] = estimate_condition_number(A)

		diag_vals = diag(A)
		row_sums = vec(sum(abs, A; dims = 2))
		metrics[:diag_dominance] = minimum(abs.(diag_vals) ./ (row_sums .- abs.(diag_vals)))
		metrics[:zero_diag_ratio] = count(iszero, diag_vals) / size(A, 1)
		metrics[:symm_diff] = norm(A - A', Inf) / max(props[:norm_inf], eps(real(T)))
	end

	# Add vector metrics if b is provided
	if !isempty(b)
		metrics[:norm_b] = norm(b, Inf)
		metrics[:vector_zeros] = count(iszero, b) / length(b)
		metrics[:vector_max_abs] = maximum(abs, b)
		nonzeros_b = filter(!iszero, b)
		metrics[:vector_min_abs] = isempty(nonzeros_b) ? 0.0 : minimum(abs, nonzeros_b)
		metrics[:vector_dynamic_range] =
			metrics[:vector_max_abs] /
			max(metrics[:vector_min_abs], eps(real(promote_type(T, S))))
	end

	return metrics
end

"""
	compute_stability_metrics(A::AbstractVector{T}) where T <: Number
Computes stability metrics for a vector `A`.
"""
function compute_stability_metrics(A::AbstractVector{T}) where T <: Number
	metrics = Dict{Symbol, Float64}()

	if isempty(A)
		metrics[:norm_A] = 0.0
		metrics[:zero_ratio] = 0.0
		return metrics
	end

	# Basic vector metrics
	metrics[:size] = length(A)
	metrics[:zero_ratio] = count(iszero, A) / length(A)
	metrics[:norm_A] = norm(A, Inf)

	# Vector value range metrics
	nonzeros_A = filter(!iszero, A)
	metrics[:min_abs] = isempty(nonzeros_A) ? 0.0 : minimum(abs, nonzeros_A)
	metrics[:max_abs] = maximum(abs, A)
	metrics[:dynamic_range] = metrics[:max_abs] / max(metrics[:min_abs], eps(real(T)))
	metrics[:sparsity] = count(!iszero, A) / length(A)

	return metrics
end

"""
	is_numerically_stable(metrics::Union{Dict{Symbol,Any}, Dict{Symbol,<:Real}}, threshold::Real = 1e8)

Checks if a matrix is numerically stable based on its computed metrics.

Returns true if:
1. Condition number is below threshold
2. Matrix is well-conditioned
3. No numerical instability indicators are present
"""
function is_numerically_stable(
	metrics::Union{Dict{Symbol, Any}, Dict{Symbol, <:Real}}, threshold::Real = 1e8
)
	# Check condition number if available
	if haskey(metrics, :cond_num)
		if metrics[:cond_num] > threshold
			return false
		end
	end

	# Check for near-singularity
	if haskey(metrics, :nearly_singular)
		if metrics[:nearly_singular] == true
			return false
		end
	end

	# Check for diagonal dominance - convert numeric value to boolean if needed
	if haskey(metrics, :diag_dominant)
		if metrics[:diag_dominant] isa Real
			# If it's a number (e.g. 0.0 or 1.0), convert to bool
			if metrics[:diag_dominant] ≈ 0.0
				return false
			end
		elseif metrics[:diag_dominant] isa Bool
			# If it's already a bool, use directly
			if !metrics[:diag_dominant]
				return false
			end
		end
	end

	# Check for other stability indicators
	if haskey(metrics, :dynamic_range)
		if metrics[:dynamic_range] > 1e15
			return false
		end
	end

	return true
end

"""
	is_numerically_stable(A::AbstractMatrix{T}, condition_threshold::Real = 1e8) where T
Convenience method that computes metrics first then checks stability.
"""
function is_numerically_stable(
	A::AbstractMatrix{T}, condition_threshold::Real = 1e8
) where T
	metrics = compute_stability_metrics(A)
	return is_numerically_stable(metrics, condition_threshold)
end

"""
	is_numerically_stable(v::AbstractVector{T}, threshold::Real = 1e8) where T

Check if a vector is numerically stable based on its values.

Returns true if:
1. Values are all finite (not NaN/Inf)
2. Values are within reasonable magnitude bounds
3. Dynamic range is reasonable
"""
function is_numerically_stable(v::AbstractVector{T}, threshold::Real = 1e8) where T
	# Check for empty vector
	isempty(v) && return false

	# Check for NaN/Inf values
	all(isfinite, v) || return false

	# Check maximum magnitude
	max_val = maximum(abs, v)
	min_val = minimum(abs, filter(!iszero, v))

	# Check magnitude bounds
	if max_val > threshold || (min_val > zero(T) && min_val < one(T) / threshold)
		return false
	end

	# Check dynamic range
	if min_val > zero(T) && max_val / min_val > threshold
		return false
	end

	return true
end

function validate_nan_result(y::AbstractMatrix{T}, method_name::String) where T
	# Check each column
	for j in axes(y, 2)
		validate_nan_result(view(y, :, j), method_name)
	end
	return y
end

function verify_matrix_dimensions(A::AbstractMatrix, b::AbstractVector)
	return size(A, 1) == size(A, 2) && size(A, 1) == length(b)
end

"""
	check_metrics_stability(metrics::Dict{Symbol,T}, threshold::T=1e8) where T<:Real

Check stability metrics against threshold. Returns true if metrics indicate stability.
"""
function check_metrics_stability(
	metrics::Dict{Symbol, T}, threshold::T = convert(T, 1e8)
) where T <: Real
	# Direct checks on metrics without calling other stability functions
	!get(metrics, :singular, true) || return false
	!get(metrics, :ill_conditioned, true) || return false
	get(metrics, :cond_num, convert(T, Inf)) <= threshold || return false
	get(metrics, :norm_A, zero(T)) > eps(T) || return false
	all(isfinite, values(metrics)) || return false
	return true
end

"""
	check_metrics_stability(A::AbstractMatrix{T}, b::AbstractVector{T}) where T

Check stability metrics for a matrix-vector system.
"""
function check_metrics_stability(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
	metrics = Dict{Symbol, T}(
		:cond_num => compute_safe_condition_number(A, T),
		:norm_A => compute_matrix_norm(A, T),
		:singular => !isposdef(Symmetric(A'A)),
		:ill_conditioned => false,
	)
	return check_metrics_stability(metrics)
end

"""
	check_metrics_stability(metrics::Dict{Symbol,T}, problem::AbstractKSProblem) where T<:Real

Check stability metrics against problem-specific threshold.
"""
function check_metrics_stability(
	metrics::Dict{Symbol, T}, problem::AbstractKSProblem
) where T <: Real
	threshold = convert(T, get(problem.properties, :stability_threshold, 1e8))
	return check_metrics_stability(metrics, threshold)
end

# Helper functions
"""
	is_reasonable_sparsity(A::AbstractMatrix, problem::AbstractKSProblem)

Check if matrix sparsity pattern is reasonable for the given problem type.
"""
function is_reasonable_sparsity(A::AbstractMatrix, ::AbstractKSProblem)
	matrix_size = size(A, 1)
	nonzeros_per_row = count(!iszero, A) ÷ matrix_size
	max_reasonable = matrix_size ÷ 2
	return nonzeros_per_row <= max_reasonable
end

"""
	has_valid_block_structure(A::AbstractMatrix, block_size::Int)

Check if matrix has valid block structure for given block size.
"""
function has_valid_block_structure(A::AbstractMatrix, block_size::Int)
	dims = size(A, 1)
	if dims % block_size != 0
		return false
	end

	A_sparse = sparse(A)
	I, J, _ = findnz(A_sparse)
	max_bandwidth = 3 * block_size
	return !any(abs.(I .- J) .> max_bandwidth)
end
end  # module NumericUtilities
