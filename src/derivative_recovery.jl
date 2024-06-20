module DerivativeRecovery

export evaluate_at_superconvergent_points, recover_derivatives, superconvergence_recovery

using LinearAlgebra

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

end # module
