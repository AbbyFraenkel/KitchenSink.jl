
@doc """
    smooth(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, x::Vector{Float64}, iterations::Int)::Vector{Float64}

Perform Jacobi smoothing iterations.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `iterations::Int`: Number of smoothing iterations.

# Returns
- `x::Vector{Float64}`: Smoothed solution.
"""
function smooth(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iterations::Int,
)::Vector{Float64}
    D_vec = diagm(A)  # Extract the diagonal elements as a vector
    for _ = 1:iterations
        x = (b .- A * x .+ D_vec .* x) ./ D_vec
    end
    return x
end

@doc """
    gauss_seidel(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, x::Vector{Float64}, iterations::Int)::Vector{Float64}

Perform Gauss-Seidel smoothing iterations.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `iterations::Int`: Number of smoothing iterations.

# Returns
- `x::Vector{Float64}`: Smoothed solution.
"""
function gauss_seidel(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iterations::Int,
)::Vector{Float64}
    n = length(b)
    for _ = 1:iterations
        for i = 1:n
            sigma = sum(A[i, j] * x[j] for j = 1:n if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        end
    end
    return x
end

@doc """
    sor(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, x::Vector{Float64}, omega::Float64, iterations::Int)::Vector{Float64}

Perform Successive Over-Relaxation (SOR) smoothing iterations.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `omega::Float64`: Relaxation parameter.
- `iterations::Int`: Number of smoothing iterations.

# Returns
- `x::Vector{Float64}`: Smoothed solution.
"""
function sor(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    omega::Float64,
    iterations::Int,
)::Vector{Float64}
    @assert 0 < omega < 2 "Relaxation parameter ω must be in (0, 2)"
    n = length(b)
    for _ = 1:iterations
        for i = 1:n
            sigma = sum(A[i, j] * x[j] for j = 1:n if j != i)
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        end
    end
    return x
end

@doc """
    restrict(residual_fine::Array{Float64, N}) where N::Array{Float64, N}

Restrict the residual to a coarser grid.

# Arguments
- `residual_fine::Array{Float64, N}`: Residual on the fine grid.

# Returns
- `residual_coarse::Array{Float64, N}`: Residual on the coarse grid.
"""
function restrict(residual_fine::Array{Float64,N}) where {N}
    size_fine = size(residual_fine)
    size_coarse = div.(size_fine .+ 1, 2)
    residual_coarse = zeros(Float64, size_coarse)

    _, w_fine = gausslegendre(size_fine[1])
    w_fine = w_fine / sum(w_fine)

    weights = ntuple(d -> w_fine, N)

    for idx in CartesianIndices(residual_coarse)
        fine_indices = CartesianIndices(ntuple(d -> 2*idx[d]-1:2*idx[d], N)...)
        sum_weights = 0.0
        value_sum = 0.0

        for fi in fine_indices
            weight = prod(weights[d][fi[d]] for d = 1:N)
            value_sum += residual_fine[fi] * weight
            sum_weights += weight
        end

        residual_coarse[idx] = value_sum / sum_weights
    end

    return residual_coarse
end

@doc """
    prolong(correction_coarse::Array{Float64, N}) where N::Array{Float64, N}

Prolong the correction to a finer grid.

# Arguments
- `correction_coarse::Array{Float64, N}`: Correction on the coarse grid.

# Returns
- `correction_fine::Array{Float64, N}`: Correction on the fine grid.
"""
function prolong(correction_coarse::Array{Float64,N}) where {N}
    size_coarse = size(correction_coarse)
    size_fine = 2 .* (size_coarse .- 1) .+ 1
    correction_fine = zeros(Float64, size_fine)

    for idx in CartesianIndices(correction_coarse)
        fine_indices = CartesianIndices(ntuple(d -> 2*idx[d]-1:2*idx[d], N)...)
        correction_fine[fine_indices...] .= correction_coarse[idx]
        for d = 1:N
            neighbor_idx = copy(fine_indices)
            neighbor_idx[d] += 1
            if neighbor_idx[d] <= size_fine[d]
                correction_fine[neighbor_idx...] .=
                    0.5 * (correction_coarse[idx] + correction_coarse[neighbor_idx])
            end
        end
    end

    return correction_fine
end

@doc """
    v_cycle(level::Int, levels::Int, mesh::MultiLayerMesh, operators::Vector{SparseMatrixCSC{Float64}}, b::Vector{Float64}, x::Vector{Float64}, smoothing_iterations::Int, logger::ConsoleLogger)::Vector{Float64}

Perform a V-cycle multigrid iteration.

# Arguments
- `level::Int`: Current grid level.
- `levels::Int`: Total number of grid levels.
- `mesh::MultiLayerMesh`: Multi-layer mesh with hierarchical grids.
- `operators::Vector{SparseMatrixCSC{Float64}}`: Operators for each grid level.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `smoothing_iterations::Int`: Number of smoothing iterations.
- `logger::ConsoleLogger`: Logger for output.

# Returns
- `x::Vector{Float64}`: Solution vector after V-cycle.
"""
function v_cycle(
    level::Int,
    levels::Int,
    mesh::MultiLayerMesh,
    operators::Vector{SparseMatrixCSC{Float64}},
    b::Vector{Float64},
    x::Vector{Float64},
    smoothing_iterations::Int,
    logger::ConsoleLogger,
)::Vector{Float64}
    if level == levels
        x = operators[level] \ b
    else
        x = smooth(operators[level], b, x, smoothing_iterations)
        residual = b - operators[level] * x
        residual_coarse = restrict(residual)
        correction_coarse = zeros(length(residual_coarse))
        correction_coarse = v_cycle(
            level + 1,
            levels,
            mesh,
            operators,
            residual_coarse,
            correction_coarse,
            smoothing_iterations,
            logger,
        )
        correction_fine = prolong(correction_coarse)
        x .+= correction_fine
        x = smooth(operators[level], b, x, smoothing_iterations)
    end
    @info logger "V-cycle at level $level completed"
    return x
end
