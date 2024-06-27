function legendre_basis_functions(
    p::Int,
)::Tuple{Vector{Function},Vector{Float64},Vector{Float64}}
    roots, weights = gausslegendre(p)
    basis_functions = [x -> legendreP(i, x) for i = 0:p]
    return basis_functions, roots, weights
end



function EO_matrix_derivative(x)
    n = length(x)
    D = zeros(n, n)
    w = barycentric_weights(x)

    for i = 2:(n-1)
        for j = 1:n
            if i != j
                D[i, j] = w[j] / ((x[i] - x[j]) * w[i])
            end
        end
        D[i, i] = -sum(D[i, :])
    end

    for j = 2:n
        D[1, j] = w[j] / ((x[1] - x[j]) * w[1])
    end
    D[1, 1] = -sum(D[1, :])

    for j = 1:(n-1)
        D[n, j] = w[j] / ((x[n] - x[j]) * w[n])
    end
    D[n, n] = -sum(D[n, :])

    return D
end

function kth_derivative_matrix(x, k_max)
    n = length(x)
    D_prev = EO_matrix_derivative(x)
    Dks = [D_prev]
    for k = 2:k_max
        D_k = zeros(n, n)
        for i = 1:n
            for j = 1:n
                if i != j
                    diff_x = x[i] - x[j]
                    D_k[i, j] = (k / diff_x) * (Dks[k-1][j, j] - Dks[k-1][i, j])
                end
            end
            D_k[i, i] = -sum(D_k[i, :])
        end
        push!(Dks, D_k)
    end
    return Dks
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

function barycentric_interpolation(x_points, y_points)
    n = length(x_points)

    return function (x)
        result = 0.0

        for i = 1:n
            basis = 1.0
            for j = 1:n
                if i != j
                    basis = basis .* (x .- x_points[j]) ./ (x_points[i] .- x_points[j])
                end
            end
            result = result .+ y_points[i] .* basis
        end

        return result
    end
end
