@inline function estimate_residual(
    A::Matrix{Float64},
    x::Vector{Float64},
    b::Vector{Float64},
)::Float64
    residual = b - A * x
    return norm(residual)
end

function compute_residuals(
    elements::Vector{Element},
    solution::Vector{Float64},
    A::Function,
    f::Function,
)::Vector{Float64}
    residuals = Vector{Float64}()
    for el in elements
        push!(residuals, compute_residual(el, solution, A, f))
    end
    return residuals
end

@inline function evaluate_at_superconvergent_points(
    solution::Vector{Float64},
    superconvergent_points::Vector{Float64},
)::Vector{Float64}
    return [solution[pt] for pt in superconvergent_points]
end

function recover_derivatives(values::Vector{Float64}, method::Symbol)::Vector{Float64}
    if method == :superconvergence
        return superconvergence_recovery(values)
    else
        error("Unknown method: $method")
    end
end

function superconvergence_recovery(values::Vector{Float64})::Vector{Float64}
    n = length(values)
    recovered_derivatives = zeros(Float64, n)

    Threads.@threads for i = 2:n-1
        recovered_derivatives[i] = (values[i+1] - values[i-1]) / 2
    end

    recovered_derivatives[1] = values[2] - values[1]
    recovered_derivatives[n] = values[n] - values[n-1]

    return recovered_derivatives
end


@inline function compute_smoothness_indicator(coefficients::Vector{Float64})::Float64
    decay_rate = norm(coefficients[end]) / norm(coefficients[1])
    return decay_rate
end


@inline function determine_refinement_strategy(
    smoothness_indicator::Float64,
    threshold::Float64,
)::Symbol
    if smoothness_indicator < threshold
        return :h_refinement
    else
        return :p_refinement
    end
end
