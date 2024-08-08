"""
    derivative_matrix(x::AbstractVector{T})::Matrix{T} where {T<:AbstractFloat}

Compute the first derivative matrix for a given set of nodes, optimized for symmetric nodes like Legendre nodes.

# Arguments
- `x::AbstractVector{T}`: The nodes.

# Returns
- `Matrix{T}`: The first derivative matrix.

# Example
```julia
nodes = [-1.0, 0.0, 1.0]
D = derivative_matrix(nodes)
println("First derivative matrix: ", D)
```
"""
function derivative_matrix(x::AbstractVector{T})::Matrix{T} where {T<:AbstractFloat}
    n = length(x)
    if n < 3
        throw(ArgumentError("The length of x must be at least 3."))
    end
    D = spzeros(T, n, n)
    w = barycentric_weights(x)

    # Precompute differences and their reciprocals
    diff_recip = spzeros(T, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                diff_recip[i, j] = inv(x[i] - x[j])
            end
        end
    end

    # Process all nodes, leveraging sparsity and precomputed values
    for i in 1:n
        for j in 1:n
            if i != j
                D[i, j] = w[j] * diff_recip[i, j] / w[i]
            end
        end
        D[i, i] = -sum(D[i, k] for k in 1:n if k != i)
    end

    return D
end

"""
    kth_derivative_matrix(x::AbstractVector{T}, k_max::Int)::Vector{Matrix{T}} where {T<:AbstractFloat}

Compute the k-th derivative matrices up to a maximum derivative order for symmetric nodes like Legendre nodes.

# Arguments
- `x::AbstractVector{T}`: The nodes.
- `k_max::Int`: The maximum derivative order.

# Returns
- `Vector{Matrix{T}}`: A vector of derivative matrices.

# Example
```julia
nodes = [-1.0, 0.0, 1.0]
Dks = kth_derivative_matrix(nodes, 3)
for (k, D) in enumerate(Dks)
    println("Derivative matrix for k=", k, ": ", D)
end
```
"""
function kth_derivative_matrix(x::AbstractVector{T}, k_max::Int)::Vector{Matrix{T}} where {T<:AbstractFloat}
    n = length(x)
    if k_max <= 0
        throw(ArgumentError("k_max must be positive."))
    end
    if n < 3
        throw(ArgumentError("The length of x must be at least 3."))
    end

    Dks = Vector{Matrix{T}}(undef, k_max)
    weights = barycentric_weights(x)
    D_prev = derivative_matrix(x)
    Dks[1] = D_prev

    # Precompute differences and their reciprocals
    diff_recip = zeros(T, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                diff_recip[i, j] = 1 / (x[i] - x[j])
            end
        end
    end

    for k in 2:k_max
        D_k = zeros(T, n, n)
        for i in 1:n
            for j in 1:n
                if i != j
                    D_k[i, j] = k * diff_recip[i, j] * (weights[j] / weights[i] * D_prev[i, i] - D_prev[i, j])
                end
            end
            D_k[i, i] = -sum(D_k[i, j] for j in 1:n if j != i)
        end
        D_prev = D_k
        Dks[k] = D_k
    end

    return Dks
end


"""
    enforce_ck_continuity(D::Matrix{T}, k::Int)::Matrix{T} where {T<:AbstractFloat}

Enforce C(k) continuity on the given derivative matrix at the boundaries.

# Arguments
- `D::Matrix{T}`: The derivative matrix.
- `k::Int`: The order of continuity to enforce.

# Returns
- `Matrix{T}`: The derivative matrix with enforced C(k) continuity.
"""
function enforce_ck_continuity(D::Matrix{T}, k::Int)::Matrix{T} where {T<:AbstractFloat}
    n = size(D, 1)
    for i in 1:k
        D[i, :] .= 0
        D[:, i] .= 0
        D[i, i] = 1
        D[n-i+1, :] .= 0
        D[:, n-i+1] .= 0
        D[n-i+1, n-i+1] = 1
    end
    return D
end

"""
    derivative_matrix_nd(nodes::Vector{Vector{T}}, dim::Int, k::Int=1)::Vector{Matrix{T}} where {T}

Compute the first derivative matrices for each dimension, optimized for symmetric nodes, and enforce C(k) continuity.

# Arguments
- `nodes::Vector{Vector{T}}`: The nodes in each dimension.
- `dim::Int`: The number of dimensions.
- `k::Int`: The order of continuity to enforce.

# Returns
- `Vector{Matrix{T}}`: A vector of first derivative matrices with enforced C(k) continuity.
"""
function derivative_matrix_nd(nodes::Vector{Vector{T}}, dim::Int, k::Int=1)::Vector{Matrix{T}} where {T<:AbstractFloat}
    # Check for valid input
    if dim <= 0
        throw(ArgumentError("Dimension must be a positive integer"))
    end
    if isempty(nodes)
        throw(ArgumentError("Nodes cannot be empty"))
    end

    D_matrices = Vector{Matrix{T}}(undef, dim)
    for i in 1:dim
        D_matrices[i] = enforce_ck_continuity(derivative_matrix(nodes[i]), k)
    end
    return D_matrices
end
