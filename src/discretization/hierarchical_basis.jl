
function hierarchical_Legendre_interpolation(
    coords::Vector{Float64}, basis_functions::Array{BasisFunction,1},
    data_points::Array{Tuple{Vector{Float64},Float64},1})
    numerator = 0.0
    denominator = 0.0
    for basis in basis_functions
        for j in 1:length(data_points)
            point_j, z_j = data_points[j]
            weight_product = 1.0
            for dim in 1:length(coords)
                weight_product *= basis.contribution / (coords[dim] - point_j[dim])
            end
            numerator += weight_product * z_j
            denominator += weight_product
        end
    end
    return numerator / denominator
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


"""
    hierarchical_Legendre(x::Vector{T}, x_nodes::Vector{T}) where {T <: Real}

Computes the hierarchical barycentric basis functions for a given set of points x
and nodes x_nodes.

# Arguments
- `x`: Vector of evaluation points
- `x_nodes`: Vector of nodes

# Returns
- `basis_functions`: Vector of vectors, each inner vector represents the values of a basis function at the points in `x`
"""
function hierarchical_Legendre(x::Vector{T}, x_nodes::Vector{T}) where {T<:Real}
    n = length(x_nodes)
    w_b = barycentric_weights(x_nodes)

    basis_functions = []

    # Linear basis functions
    for i in 1:n
        φ = [(w_b[i] / (ξ - x_nodes[i])) /
             sum(w_b[j] / (ξ - x_nodes[j]) for j in 1:n if j != i) for ξ in x]
        push!(basis_functions, φ)
    end

    # Higher-order basis functions (Legendre polynomials)
    for p in n:(2*n-2)
        φ = [legendre_polynomial(p, ξ) for ξ in x]
        push!(basis_functions, φ)
    end

    return basis_functions
end
