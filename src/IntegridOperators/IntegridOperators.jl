# src/IntegridOperators/IntegridOperators.jl
module IntegridOperators

using LinearAlgebra, SparseArrays
using ..KSTypes, ..SpectralMethods

export prolongate, restrict, compute_local_prolongation, compute_local_restriction

"""
    prolongate(coarse_solution::AbstractVector{T}, coarse_mesh::KSMesh, fine_mesh::KSMesh)::AbstractVector{T} where {T}

Prolongate a solution from a coarse mesh to a fine mesh.

# Arguments
- `coarse_solution::AbstractVector{T}`: The solution on the coarse mesh.
- `coarse_mesh::KSMesh`: The coarse mesh.
- `fine_mesh::KSMesh`: The fine mesh.

# Returns
- `AbstractVector{T}`: The prolongated solution on the fine mesh.
"""
function prolongate(coarse_solution::AbstractVector{T}, coarse_mesh::KSMesh, fine_mesh::KSMesh)::AbstractVector{T} where {T}
    fine_solution = zeros(T, sum(length(elem.basis_functions) for elem in fine_mesh.elements))

    for fine_element in fine_mesh.elements
        coarse_element = fine_element.parent
        if coarse_element === nothing
            continue
        end

        local_prolongation = compute_local_prolongation(coarse_element, fine_element)
        coarse_indices = get_active_indices(coarse_element)
        fine_indices = get_active_indices(fine_element)

        fine_solution[fine_indices] .+= local_prolongation * coarse_solution[coarse_indices]
    end

    return fine_solution
end

"""
    restrict(fine_solution::AbstractVector{T}, fine_mesh::KSMesh, coarse_mesh::KSMesh)::AbstractVector{T} where {T}

Restrict a solution from a fine mesh to a coarse mesh.

# Arguments
- `fine_solution::AbstractVector{T}`: The solution on the fine mesh.
- `fine_mesh::KSMesh`: The fine mesh.
- `coarse_mesh::KSMesh`: The coarse mesh.

# Returns
- `AbstractVector{T}`: The restricted solution on the coarse mesh.
"""
function restrict(fine_solution::AbstractVector{T}, fine_mesh::KSMesh, coarse_mesh::KSMesh)::AbstractVector{T} where {T}
    coarse_solution = zeros(T, sum(length(elem.basis_functions) for elem in coarse_mesh.elements))

    for coarse_element in coarse_mesh.elements
        for fine_element in coarse_element.children
            local_restriction = compute_local_restriction(coarse_element, fine_element)

            coarse_indices = get_active_indices(coarse_element)
            fine_indices = get_active_indices(fine_element)

            coarse_solution[coarse_indices] .+= local_restriction * fine_solution[fine_indices]
        end
    end

    return coarse_solution
end

"""
    compute_local_prolongation(coarse_element::KSElement, fine_element::KSElement)::Matrix{T} where {T}

Compute the local prolongation matrix for a fine element from a coarse element.

# Arguments
- `coarse_element::KSElement`: The coarse element.
- `fine_element::KSElement`: The fine element.

# Returns
- `Matrix{T}`: The local prolongation matrix.
"""
function compute_local_prolongation(coarse_element::KSElement, fine_element::KSElement)
    coarse_basis = coarse_element.basis_functions
    fine_basis = fine_element.basis_functions

    prolongation = zeros(T, length(fine_basis), length(coarse_basis))

    for (i, fine_bf) in enumerate(fine_basis)
        for (j, coarse_bf) in enumerate(coarse_basis)
            prolongation[i, j] = SpectralMethods.inner_product(fine_bf, coarse_bf, fine_element)
        end
    end

    return prolongation
end

"""
    compute_local_restriction(coarse_element::KSElement, fine_element::KSElement)::Matrix{T} where {T}

Compute the local restriction matrix for a fine element to a coarse element.

# Arguments
- `coarse_element::KSElement`: The coarse element.
- `fine_element::KSElement`: The fine element.

# Returns
- `Matrix{T}`: The local restriction matrix.
"""
function compute_local_restriction(coarse_element::KSElement, fine_element::KSElement)
    return compute_local_prolongation(coarse_element, fine_element)'
end

end # module IntegridOperators
