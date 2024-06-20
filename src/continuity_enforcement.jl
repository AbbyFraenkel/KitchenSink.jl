module ContinuityEnforcement

export ensure_higher_order_continuity!

using LinearAlgebra
using SparseArrays
using ..Types

"""
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
function ensure_higher_order_continuity!(
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

end # module
