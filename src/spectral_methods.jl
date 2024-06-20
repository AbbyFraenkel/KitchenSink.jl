module SpectralMethods

export legendre_basis_functions,
    kth_derivative_matrix, barycentric_weights, shifted_adapted_gauss_legendre
using FastGaussQuadrature
using LinearAlgebra

function legendre_basis_functions(
    p::Int,
)::Tuple{Vector{Function},Vector{Float64},Vector{Float64}}
    roots, weights = gausslegendre(p)
    basis_functions = [x -> legendreP(i, x) for i = 0:p]
    return basis_functions, roots, weights
end

@inline function legendreP(n::Int, x::Float64)::Float64
    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        P0 = 1.0
        P1 = x
        for k = 2:n
            P2 = ((2k - 1) * x * P1 - (k - 1) * P0) / k
            P0, P1 = P1, P2
        end
        return P1
    end
end

function kth_derivative_matrix(x::Vector{Float64}, k::Int)::Matrix{Float64}
    n = length(x)
    D = zeros(Float64, n, n)
    for i = 1:n
        for j = 1:n
            if i != j
                D[i, j] = (-1)^(i + j) / (x[i] - x[j])
            end
        end
        D[i, i] = -sum(D[i, :])
    end
    return D^k
end

@inline function barycentric_weights(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    w = ones(Float64, n)
    for j = 1:n
        for k = 1:n
            if j != k
                w[j] *= (x[j] - x[k])
            end
        end
        w[j] = 1 / w[j]
    end
    return w
end

function shifted_adapted_gauss_legendre(
    N::Int,
    a::Float64 = 0.0,
    b::Float64 = 1.0,
)::Tuple{Vector{Float64},Vector{Float64}}
    x, w = gausslegendre(N - 2)
    x = vcat(-1, x, 1)
    x = (b - a) * (x .+ 1) / 2 + a
    w_endpoint = 0.0
    w = vcat(w_endpoint, w, w_endpoint)
    w = (b - a) * w / 2
    return x, w
end


end # module
