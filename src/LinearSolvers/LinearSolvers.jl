module LinearSolvers

using LinearAlgebra, SparseArrays, Statistics, IterativeSolvers, LinearOperators
using Base.Threads
using IncompleteLU
import AlgebraicMultigrid: ruge_stuben, aspreconditioner, GaussSeidel

using ..KSTypes, ..Preconditioners, ..CacheManagement, ..NumericUtilities

# Exports
export solve_linear_system, solve_direct, solve_iterative, solve_amg
export adaptive_solve, iterative_refinement
export clear_solver_cache!, invalidate_cache_for_matrix!
export check_solution_quality

# Use CacheManagement's existing infrastructure
const SOLVER_CACHE = CacheManager{Any}(1000; strategy = :lru)

function solve_linear_system(
	A::AbstractArray{T, 2}, b::AbstractVector{T}, solver::AbstractKSLinearSolver
) where T
	# Convert special array types to regular matrix
	A_mat = convert(Matrix{T}, A)

	# Get matrix properties using NumericUtilities
	props = analyze_matrix_properties(A_mat)

	# Validate dimensions
	n, m = size(A_mat)
	n != m && throw(ArgumentError("Matrix must be square"))
	length(b) != n && throw(ArgumentError("Matrix and vector dimensions mismatch"))

	# Check condition number and regularize if needed using NumericUtilities
	κ = get(props, :cond_num, Inf)
	if κ > 1e12
		A_reg = copy(A_mat)
		regularize_matrix!(A_reg)
		if estimate_condition_number(A_reg) < κ
			A_mat = A_reg
		end
	end

	# Select solver based on matrix properties
	if isa(solver, KSDirectSolver)
		if get(props, :near_singular, false) && solver.method == :lu
			@warn "Matrix is near singular, switching to QR decomposition"
			return solve_direct(A_mat, b, :qr)
		end
		return solve_direct(A_mat, b, solver.method)
	elseif isa(solver, KSIterativeSolver)
		# Use Preconditioners module to select preconditioner
		P = select_preconditioner(A_mat)
		return solve_iterative(A_mat, b, solver.method, solver.maxiter, solver.tolerance, P)
	elseif isa(solver, KSAMGSolver)
		return solve_amg(A_mat, b, solver.maxiter, solver.tolerance, solver.preconditioner)
	end

	throw(ArgumentError("Unknown solver type: $(typeof(solver))"))
end

# Add method for AMG solver
function solve_linear_system(A::AbstractMatrix{T}, b::AbstractVector{T},
                           solver::KSAMGSolver{T}) where T
    # Create AMG preconditioner if not provided
    precond = if isnothing(solver.preconditioner)
        algebraicmultigrid(A)
    else
        solver.preconditioner
    end

    # Create GMRES parameters
    gmres_params = (
        maxiter=solver.maxiter,
        tol=solver.tolerance,
        Pl=precond
    )

    # Use GMRES with AMG preconditioning
    x, gmres_stats = gmres(A, b; gmres_params...)

    if !gmres_stats.isconverged
        @warn "AMG solver did not converge" iterations=gmres_stats.niter residual=gmres_stats.residual
    end

    return x
end

# Add helper method to handle sparse matrices specifically
function solve_linear_system(A::SparseMatrixCSC{T}, b::Vector{T},
                           solver::KSAMGSolver{T}) where T
    # Handle sparse case the same way as general case
    solve_linear_system(AbstractMatrix{T}(A), AbstractVector{T}(b), solver)
end

function solve_direct(A::AbstractMatrix{T}, b::AbstractVector{T}, method::Symbol) where T
	cache_key = (size(A), method, hash(round.(A, digits = 10)))

	# Use CacheManagement's get_or_create_cached_item
	decomp = get_or_create_cached_item(SOLVER_CACHE, cache_key) do
		if method == :lu
			lu(A)
		elseif method == :qr
			qr(A)
		else
			throw(ArgumentError("Unsupported method: $method"))
		end
	end

	x = decomp \ b
	check_solution_quality(A, x, b)
	return x
end

# Fix the iterative solver interface
function solve_iterative(A::AbstractMatrix{T}, b::AbstractVector{T},
    method::Symbol, maxiter::Int, tol::Real, preconditioner::Function) where T

    n = length(b)
    x = zeros(T, n)

    # Create temporary vectors for the preconditioner
    temp1 = similar(b)
    temp2 = similar(b)

    # Create a preconditioner operator that implements both mul! and ldiv!
    function precond_mul!(y, x)
        copyto!(temp1, x)
        temp2 .= preconditioner(temp1)
        copyto!(y, temp2)
        return y
    end

    function precond_ldiv!(y, x)
        copyto!(temp1, x)
        temp2 .= preconditioner(temp1)
        copyto!(y, temp2)
        return y
    end

    P = LinearOperator(T, n, n, true, true,
                      (y,x) -> precond_mul!(y,x),
                      (y,x) -> precond_ldiv!(y,x),
                      (y,x) -> precond_ldiv!(y,x))

    try
        if method == :cg && issymmetric(A)
            x = cg!(x, A, b; Pl=P, maxiter=maxiter, reltol=tol)
        else
            # Use correct GMRES parameters
            x = gmres!(x, A, b;
                Pl=P,
                maxiter=maxiter,
                restart=min(20,n),
                reltol=tol,
                initially_zero=true)
        end

        # Check solution quality
        rel_residual = norm(A * x - b) / norm(b)
        props = analyze_matrix_properties(A)
        actual_tol = get(props, :ill_conditioned, false) ? 1e-4 : tol

        if rel_residual > actual_tol
            @debug "Iterative solver did not achieve desired accuracy, trying direct solve"
            return A \ b
        end

        return x
    catch e
        @warn "Iterative method failed, falling back to direct solve" exception=e
        return A \ b
    end
end

function solve_amg(A::AbstractMatrix{T}, b::AbstractVector{T},
	maxiter::Int, tol::Real, preconditioner::Any) where T
	props = analyze_matrix_properties(A)
	A_sparse = props[:sparse] ? A : sparse(A)

	try
		ml = ruge_stuben(A_sparse)
		P = aspreconditioner(ml)

		# Test preconditioner using NumericUtilities
		converged, _ = test_preconditioner_convergence(A_sparse, b, x -> P \ x)
		if !converged
			@warn "AMG preconditioner may be ineffective"
		end

		x = zeros(T, length(b))
		cg!(x, A_sparse, b; maxiter = maxiter, reltol = tol, Pl = P)

		metrics = compute_stability_metrics(A_sparse, x)
		if metrics[:dynamic_range] > 1e8
			x = iterative_refinement(A_sparse, b, x)
		end

		check_solution_quality(A_sparse, x, b)
		return x
	catch e
		@warn "AMG solver failed, falling back to direct solver" exception = e
		return solve_direct(A, b, :lu)
	end
end

function adaptive_solve(A::AbstractMatrix{T}, b::AbstractVector{T}) where T
	# Convert sparse matrix to the right type for analysis
	A_dense = issparse(A) ? Matrix(A) : A

	# Use NumericUtilities to analyze matrix
	props = analyze_matrix_properties(A_dense)
	metrics = compute_stability_metrics(A_dense)

	# Add missing properties with defaults
	props[:posdef] = get(props, :posdef, false)
	props[:symmetric] = get(props, :symmetric, false)
	props[:diag_dominant] = get(props, :diag_dominant, false)

	# Choose solver based on matrix properties
	solver = if get(props, :sparse, false)
		if get(props, :symmetric, false) && get(props, :posdef, false)
			if get(props, :condition_number, Inf) < 1e6
				KSIterativeSolver(:cg, 1000, sqrt(eps(T)))
			else
				KSAMGSolver(1000, sqrt(eps(T)), GaussSeidel())
			end
		elseif get(props, :diag_dominant, false)
			KSIterativeSolver(:gmres, 1000, sqrt(eps(T)))
		else
			KSDirectSolver(:qr)
		end
	else
		if get(props, :posdef, false) && !get(props, :ill_conditioned, false)
			KSDirectSolver(:lu)
		elseif get(props, :symmetric, false)
			KSIterativeSolver(:gmres, 1000, sqrt(eps(T)))
		else
			KSDirectSolver(:qr)
		end
	end

	return solve_linear_system(A, b, solver)
end

function iterative_refinement(A::AbstractMatrix{T}, b::AbstractVector{T},
	x_initial::AbstractVector{T};
	max_refinements::Int = 3) where T
	x = copy(x_initial)
	best_x = copy(x)
	best_residual = norm(A * x - b)

	for _ in 1:max_refinements
		r = b - A * x
		dx = solve_direct(A, r, :lu)
		x += dx
		current_residual = norm(A * x - b)
		if current_residual < best_residual
			best_residual = current_residual
			best_x .= x
		end
		if current_residual < sqrt(eps(T)) * norm(b)
			break
		end
	end

	return best_x
end

# Update check_solution_quality with relaxed tolerances
function check_solution_quality(A::AbstractMatrix{T}, x::AbstractVector{T},
	b::AbstractVector{T};
	tol::Real = sqrt(eps(T))) where T
	metrics = compute_stability_metrics(A, x)
	props = analyze_matrix_properties(A)

	# Use larger tolerance for ill-conditioned matrices
	actual_tol = get(props, :ill_conditioned, false) ? 1e-1 : tol

	if !all(isfinite, x)
		throw(ArgumentError("Solution contains non-finite values"))
	end

	residual_norm = norm(A * x - b)
	norm_b = norm(b)

	if residual_norm > actual_tol * (norm_b + eps(T))
		@warn "Large residual" residual = residual_norm / norm_b
	end

	return true
end

function clear_solver_cache!()
	# Use CacheManagement's thread_safe_cache_operation
	thread_safe_cache_operation() do
		empty!(SOLVER_CACHE.items)
		empty!(SOLVER_CACHE.metadata)
	end
end

function invalidate_cache_for_matrix!(A::AbstractMatrix)
	matrix_hash = hash(round.(A, digits = 10))
	thread_safe_cache_operation() do
		for (key, metadata) in SOLVER_CACHE.metadata
			if get(metadata.custom, :matrix_hash, nothing) == matrix_hash
				delete!(SOLVER_CACHE.items, key)
				delete!(SOLVER_CACHE.metadata, key)
			end
		end
	end
end

using LinearOperators, LinearAlgebra

"""
    ldiv!(Y, A::LinearOperator, B)

Implements left division for LinearOperators. This method is required for iterative solvers.
"""
function LinearAlgebra.ldiv!(Y::AbstractVecOrMat, A::LinearOperator, B::AbstractVecOrMat)
    mul!(Y, A.inverse_op, B)
    return Y
end

"""
    ldiv!(A::LinearOperator, b::AbstractVector)

In-place left division for LinearOperators with vectors.
"""
function LinearAlgebra.ldiv!(A::LinearOperator, b::AbstractVector)
    temp = similar(b)
    mul!(temp, A.inverse_op, b)
    copyto!(b, temp)
    return b
end

"""
    ldiv!(A::LinearOperator, B::AbstractMatrix)

In-place left division for LinearOperators with matrices.
"""
function LinearAlgebra.ldiv!(A::LinearOperator, B::AbstractMatrix)
    temp = similar(B)
    for j in 1:size(B, 2)
        @views mul!(temp[:, j], A.inverse_op, B[:, j])
    end
    copyto!(B, temp)
    return B
end

# Helper method to create inverse operator
function create_inverse_operator(A::LinearOperator)
    n = size(A, 1)
    return LinearOperator(Float64, n, n,
        false, false,
        (y, x) -> ldiv!(y, factorize(Matrix(A)), x),
        nothing,
        nothing)
end

# Update the solve_linear_system method to handle LinearOperators
function solve_linear_system(A::LinearOperator, b::AbstractVecOrMat;
                           preconditioner=nothing,
                           solver_type=:iterative,
                           kwargs...)
    if solver_type == :iterative
        try
            return iterative_solve(A, b; preconditioner=preconditioner, kwargs...)
        catch e
            @warn "Iterative method failed, falling back to direct solve" exception=e
            return direct_solve(A, b; kwargs...)
        end
    else
        return direct_solve(A, b; kwargs...)
    end
end

end # module LinearSolvers
