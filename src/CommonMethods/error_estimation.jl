"""
    estimate_error(element::KSElement{T}, solution::AbstractVector{T}, problem::KSProblem{T,N,C})::T where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}

Estimate the error for a given element using a residual-based error estimator.

# Arguments
- `element::KSElement{T}`: The element to estimate the error for.
- `solution::AbstractVector{T}`: The current solution vector.
- `problem::KSProblem{T,N,C}`: The problem being solved.

# Returns
- `T`: The estimated error for the element.
"""
function estimate_error(element::KSElement{T}, solution::AbstractVector{T}, problem::KSProblem{T,N,C})::T where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}
    residual = compute_residual(element, solution, problem)
    h = element_size(element)
    return h * norm(residual)
end

"""
    compute_residual(element::KSElement{T}, solution::AbstractVector{T}, problem::KSProblem{T,N,C})::AbstractVector{T} where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}

Compute the residual for a given element.

# Arguments
- `element::KSElement{T}`: The element to compute the residual for.
- `solution::AbstractVector{T}`: The current solution vector.
- `problem::KSProblem{T,N,C}`: The problem being solved.

# Returns
- `AbstractVector{T}`: The residual vector for the element.
"""
function compute_residual(element::KSElement{T}, solution::AbstractVector{T}, problem::KSProblem{T,N,C})::AbstractVector{T} where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}
    Lu = apply_differential_operator(solution, element, problem)
    f = [problem.equation(solution[i], element.points[i].coordinates, []) for i in eachindex(solution)]
    return f - Lu
end

"""
    compute_error_indicator(element::KSElement{T}, solution::AbstractVector{T}, problem::KSProblem{T,N,C})::Tuple{T, T} where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}

Compute an error indicator for hp-adaptive refinement.

# Arguments
- `element::KSElement{T}`: The element to compute the error indicator for.
- `solution::AbstractVector{T}`: The current solution vector.
- `problem::KSProblem{T,N,C}`: The problem being solved.

# Returns
- `Tuple{T, T}`: A tuple containing the error indicator and a smoothness measure.
"""
function compute_error_indicator(element::KSElement{T}, solution::AbstractVector{T}, problem::KSProblem{T,N,C})::Tuple{T,T} where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}
    error = estimate_error(element, solution, problem)
    smoothness = estimate_smoothness(element, solution)
    return error, smoothness
end

"""
    apply_differential_operator(u::AbstractVector{T}, element::KSElement{T}, problem::KSProblem{T,N,C})::AbstractVector{T} where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}

Apply the differential operator to the solution vector `u` for a given element using spectral methods.

# Arguments
- `u::AbstractVector{T}`: The solution vector.
- `element::KSElement{T}`: The element to which the operator is applied.
- `problem::KSProblem{T,N,C}`: The problem being solved, which defines the differential operator.

# Returns
- `AbstractVector{T}`: The resulting vector after applying the operator.
"""
function apply_differential_operator(u::AbstractVector{T}, element::KSElement{T}, problem::KSProblem{T,N,C})::AbstractVector{T} where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}
    D = element.differentiation_matrices
    result = similar(u)

    for i in eachindex(u)
        derivatives = [D[d] * u for d in 1:length(D)]
        result[i] = problem.equation(u[i], element.points[i].coordinates, derivatives)
    end

    return result
end


"""
    compute_expansion_coefficients(element::KSElement{T}, solution::AbstractVector{T})::AbstractVector{T} where {T<:Real}

Compute the expansion coefficients for the solution on a given element.

# Arguments
- `element::KSElement{T}`: The element to compute the coefficients for.
- `solution::AbstractVector{T}`: The current solution vector.

# Returns
- `AbstractVector{T}`: The expansion coefficients.
"""
function compute_expansion_coefficients(element::KSElement{T}, solution::AbstractVector{T})::AbstractVector{T} where {T<:Real}
    basis_matrix = hcat([bf.function_handle.(element.points) for bf in element.basis_functions]...)
    return basis_matrix \ solution
end
