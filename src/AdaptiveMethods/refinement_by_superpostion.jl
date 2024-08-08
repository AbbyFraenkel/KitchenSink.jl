"""
    hp_refine_superposition(element::KSElement, error::T, smoothness::T, threshold::T)::Vector{KSElement} where {T}

Perform hp-refinement with superposition on a given element based on error and smoothness indicators.

# Arguments
- `element::KSElement`: The element to be refined.
- `error::T`: The error indicator for the element.
- `smoothness::T`: The smoothness indicator for the element.
- `threshold::T`: The threshold for deciding between h- and p-refinement.

# Returns
- `Vector{KSElement}`: A vector of new elements resulting from the refinement.
"""
function hp_refine_superposition(element::KSElement, error::T, smoothness::T, threshold::T)::Vector{KSElement} where {T}
    if error > threshold
        if smoothness > log(element.polynomial_degree) / element.polynomial_degree
            return [p_refine(element)]
        else
            return h_refine(element)
        end
    else
        return [element]  # No refinement needed
    end
end

"""
    superconvergence_refinement(element::KSElement, solution::AbstractVector{T}, problem::AbstractKSProblem)::Vector{KSElement} where {T}

Perform refinement based on superconvergence principles.

# Arguments
- `element::KSElement`: The element to be refined.
- `solution::AbstractVector{T}`: The current solution vector.
- `problem::AbstractKSProblem`: The problem being solved.

# Returns
- `Vector{KSElement}`: A vector of new elements resulting from the refinement.
"""
function superconvergence_refinement(element::KSElement, solution::AbstractVector{T}, problem::AbstractKSProblem)::Vector{KSElement} where {T}
    error, smoothness = compute_error_indicator(element, solution, problem)
    if smoothness > log(element.polynomial_degree) / element.polynomial_degree
        return [p_refine(element)]
    else
        return h_refine(element)
    end
end

"""
    adapt_mesh_superposition(mesh::KSMesh, solution::AbstractVector{T}, problem::AbstractKSProblem, threshold::T)::KSMesh where {T}

Adapt the mesh using superposition and superconvergence techniques.

# Arguments
- `mesh::KSMesh`: The current mesh.
- `solution::AbstractVector{T}`: The current solution vector.
- `problem::AbstractKSProblem`: The problem being solved.
- `threshold::T`: The error threshold for refinement.

# Returns
- `KSMesh`: The adapted mesh.
"""
function adapt_mesh_superposition(mesh::KSMesh, solution::AbstractVector{T}, problem::AbstractKSProblem, threshold::T)::KSMesh where {T}
    new_elements = Vector{KSElement}()

    for element in mesh.elements
        refined_elements = superconvergence_refinement(element, solution, problem)
        append!(new_elements, refined_elements)
    end

    new_mesh = KSMesh(
        new_elements,
        Vector{BitArray}(undef, length(new_elements)),
        Vector{Dict{Int,Int}}(undef, length(new_elements)),
        unique(vcat([el.basis_functions for el in new_elements]...)),
        sum([el.error_estimate for el in new_elements]),
        mesh.dimensions
    )

    Preprocessing.update_tensor_product_masks!(new_mesh)

    return new_mesh
end
