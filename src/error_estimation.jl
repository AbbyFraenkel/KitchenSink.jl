module ErrorEstimation

export estimate_error, compute_residuals

using LinearAlgebra
using ..Types

@inline function estimate_error(
    A::Matrix{Float64},
    x::Vector{Float64},
    b::Vector{Float64},
)::Float64
    residual = b - A * x
    return norm(residual) / norm(b)
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

# Fix
function compute_residual(
    element::Element,
    solution::Vector{Float64},
    A::Function,
    f::Function,
)::Float64
    residual = 0.0
    for pt in element.nodes
        differential = A(pt, solution)
        exact_value = f(pt)
        residual += abs(differential - exact_value)^2
    end
    return sqrt(residual)
end

end # module
