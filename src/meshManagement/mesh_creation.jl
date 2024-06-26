@doc """
    ensure_higher_order_continuity!(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, num_elements::Int, num_basis::Int, continuity_order::Int=2)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}

Ensure higher order continuity in the system by modifying the system matrix and the right-hand side vector.

# Arguments
- `A::SparseMatrixCSC{Float64,Int}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `num_elements::Int`: Number of elements in the mesh.
- `num_basis::Int`: Number of basis functions per element.
- `continuity_order::Int=2`: Desired order of continuity between elements (default is 2).

# Returns
- `Tuple{SparseMatrixCSC{Float64,Int}, Vector{Float64}}`: Modified system matrix and right-hand side vector to enforce continuity.

# Description
This function enforces higher order continuity between adjacent elements in the mesh by modifying the system matrix `A` and the right-hand side vector `b`. The continuity order is specified by `continuity_order`, which defaults to 2. The function iterates over the elements and adjusts the matrix and vector entries to ensure the desired continuity.
"""
function EnsureHigherOrderContinuity!(
    A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    num_elements::Int,
    num_basis::Int,
    continuity_order::Int = 2,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    for e = 1:(num_elements-1)
        for k = 1:continuity_order
            i1 = e * num_basis - (num_basis - k)
            i2 = e * num_basis + k

            A[i1, :] .-= A[i2, :]
            A[:, i1] .-= A[:, i2]
            b[i1] -= b[i2]
        end
    end
    return A, b
end


@inline function check_degrees_of_freedom(
    A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    num_dofs::Int,
)::Nothing
    @assert size(A, 1) == num_dofs "System matrix row size does not match number of degrees of freedom"
    @assert size(A, 2) == num_dofs "System matrix column size does not match number of degrees of freedom"
    @assert length(b) == num_dofs "Right-hand side vector size does not match number of degrees of freedom"
end

function MeshAssembly(
    ndims::Int,
    domain::Vector{Float64},
    num_elements::Int,
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    lambda::Float64,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    num_basis = length(basis_functions)
    num_dofs = num_elements^ndims * num_basis
    A = spzeros(Float64, num_dofs, num_dofs)
    b = zeros(Float64, num_dofs)

    Threads.@threads for el in CartesianIndices(ntuple(_ -> num_elements, ndims))
        for i = 1:num_basis
            for j = 1:num_basis
                A[
                    linear_index(el, i, num_elements, num_basis),
                    linear_index(el, j, num_elements, num_basis),
                ] += integrate_basis(
                    ndims,
                    domain,
                    basis_functions[i],
                    basis_functions[j],
                    collocation_points,
                    weights,
                )
            end
            b[linear_index(el, i, num_elements, num_basis)] += integrate_forcing(
                ndims,
                domain,
                basis_functions[i],
                collocation_points,
                weights,
            )
        end
    end

    A .+= lambda * I

    return A, b
end
