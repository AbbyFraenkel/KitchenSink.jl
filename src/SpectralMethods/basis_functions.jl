"""
    barycentric_weights(nodes::AbstractVector{T})::Vector{T} where {T}

Compute the barycentric weights for a set of nodes.

# Arguments
- `nodes::AbstractVector{T}`: The nodes to compute weights for.

# Returns
- `Vector{T}`: The barycentric weights for the nodes.

# Example
```julia
nodes, _ = create_nodes(5)
weights = barycentric_weights(nodes)
println("Barycentric Weights: ", weights)
```
"""
function barycentric_weights(nodes::AbstractVector{T})::Vector{T} where {T}
    n = length(nodes)
    if n < 3
        throw(ArgumentError("The length of nodes must be at least 3."))
    end

    w = ones(T, n)
    for j in 1:n
        for k in 1:n
            if j != k
                w[j] /= (nodes[j] - nodes[k])
            end
        end
    end
    return w
end

"""
    Lagrange_polynomials(nodes::AbstractVector{T})::Matrix{Float64}

Compute the Lagrange polynomials for a given set of nodes.

# Arguments
- `nodes::AbstractVector{T}`: The nodes to compute the polynomials for.

# Returns
- `Matrix{Float64}`: The Lagrange polynomials evaluated at the nodes.

# Example
```julia
nodes, _ = create_nodes(5)
L = Lagrange_polynomials(nodes)
println("Lagrange polynomials: ", L)
```
"""
function Lagrange_polynomials(nodes::AbstractVector{T}) where {T}
    n = length(nodes)
    if n < 3
        throw(ArgumentError("The length of nodes must be at least 3."))
    end

    # Lagrange polynomials create the identity matrix at Gauss points
    return sparse(I, n, n)
end

"""
    barycentric_interpolation(nodes::AbstractVector{T}, values::AbstractVector{T}, x::T)::T where {T}

Perform barycentric interpolation for a given set of nodes and values at a point x.

# Arguments
- `nodes::AbstractVector{T}`: The nodes.
- `values::AbstractVector{T}`: The values at the nodes.
- `x::T`: The point to interpolate at.

# Returns
- `T`: The interpolated value at x.

# Example
```julia
nodes, _ = create_nodes(3)
values = [1.0, 2.0, 3.0]
x = 0.5
interpolated_value = barycentric_interpolation(nodes, values, x)
println("Interpolated value at ", x, ": ", interpolated_value)
```
"""
function barycentric_interpolation(nodes::AbstractVector{T}, values::AbstractVector{T}, x::T)::T where {T}
    n = length(nodes)
    weights = barycentric_weights(nodes)
    numerator = zero(T)
    denominator = zero(T)

    for j in 1:n
        if x == nodes[j]
            return values[j]
        end
        term = weights[j] / (x - nodes[j])
        numerator += term * values[j]
        denominator += term
    end

    return numerator / denominator
end

"""
    interpolate_nd(nodes::Vector{Vector{T}}, values::AbstractArray{T}, point::Vector{T})::T where {T<:AbstractFloat}

Perform n-dimensional barycentric interpolation for a given set of nodes and values at a point.

# Arguments
- `nodes::Vector{Vector{T}}`: The nodes in each dimension.
- `values::AbstractArray{T}`: The values at the nodes.
- `point::Vector{T}`: The point to interpolate at.

# Returns
- `T`: The interpolated value at the point.

# Example
```julia
nodes = gauss_legendre_with_boundary_nd(5, 2, [0.0, 0.0], [1.0, 1.0])
values = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
point = [0.5, -0.5]
interpolated_value = interpolate_nd(nodes, values, point)
println("Interpolated value at ", point, ": ", interpolated_value)
```
"""
function interpolate_nd(nodes::Vector{Vector{T}}, values::AbstractArray{T}, point::Vector{T})::T where {T<:AbstractFloat}

    weights = [barycentric_weights(n) for n in nodes]

    function interpolate_nd_recursive(nodes::Vector{Vector{T}}, weights::Vector{Vector{T}},
        values::AbstractArray{T}, point::Vector{T})::T where {T}
        if length(nodes) == 1
            return barycentric_interpolation(nodes[1], values, point[1])
        else
            interpolated_values = [interpolate_nd_recursive(nodes[2:end], weights[2:end],
                values[:, i], point[2:end]) for i in 1:length(nodes[1])]
            return barycentric_interpolation(nodes[1], interpolated_values, point[1])
        end
    end

    return interpolate_nd_recursive(nodes, weights, values, point)
end
