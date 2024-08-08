"""
    create_nodes(p::Int, a::T=0.0, b::T=1.0)::Tuple{Vector{T}, Vector{T}} where T

Compute the shifted and adapted Gauss-Legendre nodes and weights for polynomial degree `p`.

# Arguments
- `p::Int`: Polynomial degree. Must be at least 3.
- `a::T=0.0`: Lower bound of the interval (default: 0.0).
- `b::T=1.0`: Upper bound of the interval (default: 1.0).

# Returns
- `Tuple{Vector{T}, Vector{T}}`: Shifted nodes and adapted weights.

# Example
```julia
nodes, weights = create_nodes(5)
println("Nodes: ", nodes)
println("Weights: ", weights)
```
"""
function create_nodes(p::Int, a::T=0.0, b::T=1.0)::Tuple{Vector{T},Vector{T}} where {T}
    if p < 3
        throw(ArgumentError("Polynomial degree p must be at least 3."))
    end
    x, w = gausslegendre(p - 2)
    x = vcat(T(-1), x, T(1))
    x .= (b - a) .* (x .+ one(T)) ./ 2 .+ a
    w_endpoint = zero(T)
    w = vcat(w_endpoint, w, w_endpoint)
    w .= (b - a) .* w ./ 2
    return x, w
end

"""
    gauss_legendre_with_boundary_nd(degree::Int, dim::Int, a::Vector{T}, b::Vector{T})::Tuple{Vector{Vector{T}}, Vector{Vector{T}}} where T

Generate Gauss-Legendre points with boundary nodes for n-dimensional problems, preserving domain scaling.

# Arguments
- `degree::Int`: The degree of the Gauss-Legendre polynomial.
- `dim::Int`: The number of dimensions.
- `a::Vector{T}`: Lower bounds of the intervals for each dimension.
- `b::Vector{T}`: Upper bounds of the intervals for each dimension.

# Returns
- `Tuple{Vector{Vector{T}}, Vector{Vector{T}}}`: A tuple containing the points and weights for each dimension.

# Example
```julia
points, weights = gauss_legendre_with_boundary_nd(5, 3, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
println("Points: ", points)
println("Weights: ", weights)
```
"""
function gauss_legendre_with_boundary_nd(degree::Int, dim::Int, a::Vector{T}, b::Vector{T}) where {T}
    if length(a) != dim || length(b) != dim
        throw(ArgumentError("Dimensions of a and b must match the specified dimension dim."))
    end

    X = Vector{Vector{T}}(undef, dim)
    W = Vector{Vector{T}}(undef, dim)

    for i in 1:dim
        X[i], W[i] = create_nodes(degree, a[i], b[i])
    end

    return X, W
end

