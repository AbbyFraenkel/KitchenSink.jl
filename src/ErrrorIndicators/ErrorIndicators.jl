module ErrorIndicators

function residual_error_estimator(element::Element, problem::AbstractProblem, dt::Float64)
    residuals = element.residuals
    u = element.solution
    u_prev = copy(u)  # Assuming you store the previous solution

    # Compute the residuals
    if isa(problem, PDEProblem)
        problem.equation(residuals, u, element)
    elseif isa(problem, DAEProblem)
        problem.equation(residuals, u, element)
    elseif isa(problem, HybridProblem)
        problem.pde_equation(residuals, u, element)
        problem.dae_equation(residuals, u, element)
    end

    # Include the time derivative component
    residuals .= (u .- u_prev) / dt .+ residuals

    return norm(residuals)  # Norm of the residuals as the error estimator
end


function energy_norm_error_estimator(element::Element, problem::AbstractProblem)
    residuals = element.residuals
    u = element.solution
    J = compute_jacobian(element)

    if isa(problem, PDEProblem)
        problem.equation(residuals, u, element)
    elseif isa(problem, DAEProblem)
        problem.equation(residuals, u, element)
    elseif isa(problem, HybridProblem)
        problem.pde_equation(residuals, u, element)
        problem.dae_equation(residuals, u, element)
    end
    # Energy norm: ||e||_E = (J e, e)^(1/2), where e is the error estimate
    e = residuals
    energy_norm = sqrt(dot(J * e, e))
    return energy_norm
end


function estimate_error(
        element::Element, problem::AbstractProblem, dt::Float64, method::Symbol = :residual)
    if method == :residual
        return residual_error_estimator(element, problem, dt)
    elseif method == :energy
        return energy_norm_error_estimator(element, problem)
    else
        error("Unknown error estimation method: $method")
    end
end

end

"""
Calculate the residual error between the right-hand side vector `b` and the matrix-vector product `A*x`.

# Arguments
- `A::AbstractMatrix`: The coefficient matrix.
- `x::AbstractVector`: The solution vector.
- `b::AbstractVector`: The right-hand side vector.

# Returns
- `error::Real`: The estimated error.

"""

function residialError(A, x, b)
    # A is the coefficient matrix
    # x is the solution vector
    # b is the right-hand side vector
    # Calculate the residual
    r = b - A * x
    # Estimate error
    error = norm(r)
    return error
end
