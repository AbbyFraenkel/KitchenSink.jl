module ErrorEstimation

using LinearAlgebra
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods

export estimate_error, compute_error_indicator, compute_smoothness_indicator, compute_residual
export apply_differential_operator, compute_expansion_coefficients, estimate_smoothness
export estimate_multilevel_error, compute_multilevel_error_indicator

"""
    estimate_error(element::KSElement{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::T where {T<:Real,N}

Estimate the error for a given element using a residual-based error estimator.

# Arguments
- `element::KSElement{T,N}`: The element to estimate the error for.
- `solution::AbstractArray{T}`: The current solution vector.
- `problem::KSProblem{T,N}`: The problem being solved.

# Returns
- `T`: The estimated error for the element.
"""
function estimate_error(element::KSElement{T, N}, solution::AbstractArray{T}, problem::KSProblem{T, N}) where {T <: Real, N}
    residual = compute_residual(element, solution, problem)
    h = CommonMethods.element_size(element)
    return norm(residual) * sqrt(prod(h)) / sqrt(length(residual))  # Normalizing by the number of residuals
end

"""
    compute_residual(element::KSElement{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::AbstractArray{T} where {T<:Real,N}

Compute the residual for a given element.

# Arguments
- `element::KSElement{T,N}`: The element to compute the residual for.
- `solution::AbstractArray{T}`: The current solution vector.
- `problem::KSProblem{T,N}`: The problem being solved.

# Returns
- `AbstractArray{T}`: The residual vector for the element.
"""
function compute_residual(element::KSElement{T, N}, solution::AbstractArray{T}, problem::KSProblem{T, N})::AbstractArray{T} where {T <: Real, N}
    Lu = apply_differential_operator(solution, element, problem)
    f = [problem.equation(x.coordinates...) for x in element.collocation_points]
    return f - Lu
end

"""
    compute_error_indicator(element::KSElement{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::Tuple{T,T} where {T<:Real,N}

Compute an error indicator for hp-adaptive refinement.

# Arguments
- `element::KSElement{T,N}`: The element to compute the error indicator for.
- `solution::AbstractArray{T}`: The current solution vector.
- `problem::KSProblem{T,N}`: The problem being solved.

# Returns
- `Tuple{T, T}`: A tuple containing the error indicator and a smoothness measure.
"""
function compute_error_indicator(element::KSElement{T, N}, solution::AbstractArray{T}, problem::KSProblem{T, N})::Tuple{T, T} where {T <: Real, N}
    error = estimate_error(element, solution, problem)
    smoothness = compute_smoothness_indicator(element, solution)
    return error, smoothness
end

"""
    compute_smoothness_indicator(element::KSElement{T, N}, solution::AbstractArray{S})::T where {T <: Real, S <: Real, N}

Compute the smoothness indicator for a given element and solution.

# Arguments
- `element::KSElement{T,N}`: The element to compute the smoothness indicator for.
- `solution::AbstractArray{S}`: The current solution vector.

# Returns
- `T`: The smoothness indicator for the element.
"""
function compute_smoothness_indicator(element::KSElement{T, N}, solution::AbstractArray{S})::T where {T <: Real, S <: Real, N}
    coeffs = compute_expansion_coefficients(element, solution)
    return -log(norm(coeffs[end-1:end])) / sum(element.polynomial_degree)
end

"""
    apply_differential_operator(u::AbstractArray{T}, element::KSElement{T,N}, problem::KSProblem{T,N})::AbstractArray{T} where {T<:Real,N}

Apply the differential operator to the solution vector `u` for a given element using spectral methods.

# Arguments
- `u::AbstractArray{T}`: The solution vector.
- `element::KSElement{T,N}`: The element to which the operator is applied.
- `problem::KSProblem{T,N}`: The problem being solved, which defines the differential operator.

# Returns
- `AbstractArray{T}`: The resulting vector after applying the operator.
"""
function apply_differential_operator(u::AbstractArray{T}, element::KSElement{T, N}, problem::KSProblem{T, N})::AbstractArray{T} where {T <: Real, N}
    D = element.differentiation_matrices
    result = similar(u)

    for i in eachindex(u)
        derivatives = [D[d] * u for d in 1:N]  # Apply each differential operator dimension-wise
        result[i] = problem.equation(element.collocation_points[i].coordinates..., derivatives...)
    end

    return result
end

"""
    compute_expansion_coefficients(element::KSElement{T, N}, solution::Vector{T}; lambda::T = 1e-6) where {T <: Real}

Compute the expansion coefficients for the solution on the given element.

# Arguments
- `element::KSElement{T,N}`: The `KSElement` object on which to compute the expansion coefficients.
- `solution::Vector{T}`: The solution vector.
- `lambda::T = 1e-6`: Regularization parameter for the linear system.

# Returns
- `Vector{T}`: A vector of expansion coefficients.

# Throws
- `ArgumentError`: If the regularization parameter is negative.
- `DimensionMismatch`: If the solution length doesn't match the number of collocation points.
- `LinearAlgebra.SingularException`: If the resulting matrix is singular or nearly singular.
"""
function compute_expansion_coefficients(element::KSElement{T, N}, solution::Vector{T}; lambda::T = 1e-6) where {T <: Real, N}
    if lambda < 0
        throw(ArgumentError("Regularization parameter lambda must be non-negative"))
    end

    if length(solution) != length(element.collocation_points)
        throw(DimensionMismatch("Solution length does not match the number of collocation points"))
    end

    # Retrieve the standard element for this element's polynomial degree
    std_elem = SpectralMethods.create_standard_element(element.polynomial_degree)

    basis_functions = element.basis_functions
    collocation_points = element.collocation_points

    # Construct the system to solve for the expansion coefficients
    A = zeros(T, length(basis_functions), length(basis_functions))
    b = zeros(T, length(basis_functions))

    for i in eachindex(basis_functions)
        for j in eachindex(basis_functions)
            A[i, j] = sum(basis_functions[i].function_handle(p.coordinates) * basis_functions[j].function_handle(p.coordinates) for p in collocation_points)
        end
        b[i] = sum(solution[k] * basis_functions[i].function_handle(collocation_points[k].coordinates) for k in 1:length(solution))
    end

    if lambda > 0.0
        A += lambda * I
    end

    if cond(A) > 1e12  # Check condition number instead of determinant
        throw(LinearAlgebra.SingularException(length(A)))
    end

    coeffs = A \ b

    return coeffs
end

"""
    estimate_smoothness(element::KSElement{T, N}, solution::AbstractVector{T}) where {T <: Real, N}

Estimate the smoothness of the solution on the given element.

# Arguments
- `element`: The `KSElement` object on which to estimate smoothness.
- `solution`: The solution vector.

# Returns
- A scalar value representing the estimated smoothness.
"""
function estimate_smoothness(element::KSElement{T, N}, solution::AbstractVector{T}) where {T <: Real, N}
    coeffs = compute_expansion_coefficients(element, solution)
    return -log(norm(coeffs[end-1:end])) / sum(element.polynomial_degree)
end

"""
    estimate_multilevel_error(mesh::KSMesh{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::AbstractArray{T} where {T<:Real,N}

Estimate the error for all elements in a multi-level mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The multi-level mesh.
- `solution::AbstractArray{T}`: The current solution vector.
- `problem::KSProblem{T,N}`: The problem being solved.

# Returns
- `AbstractArray{T}`: An array of error estimates for each element in the mesh.
"""
function estimate_multilevel_error(mesh::KSMesh{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::AbstractArray{T} where {T<:Real,N}
    error_estimates = similar(solution, length(mesh.elements))

    for (i, element) in enumerate(mesh.elements)
        local_solution = solution[CommonMethods.get_active_indices(mesh, i)]
        error_estimates[i] = estimate_error(element, local_solution, problem)
    end

    return error_estimates
end

"""
    compute_multilevel_error_indicator(mesh::KSMesh{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::AbstractArray{Tuple{T,T}} where {T<:Real,N}

Compute error indicators for all elements in a multi-level mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The multi-level mesh.
- `solution::AbstractArray{T}`: The current solution vector.
- `problem::KSProblem{T,N}`: The problem being solved.

# Returns
- `AbstractArray{Tuple{T,T}}`: An array of tuples containing error indicators and smoothness measures for each element in the mesh.
"""
function compute_multilevel_error_indicator(mesh::KSMesh{T,N}, solution::AbstractArray{T}, problem::KSProblem{T,N})::AbstractArray{Tuple{T,T}} where {T<:Real,N}
    error_indicators = Array{Tuple{T,T}}(undef, length(mesh.elements))

    for (i, element) in enumerate(mesh.elements)
        local_solution = solution[CommonMethods.get_active_indices(mesh, i)]
        error_indicators[i] = compute_error_indicator(element, local_solution, problem)
    end

    return error_indicators
end

end # module ErrorEstimation
