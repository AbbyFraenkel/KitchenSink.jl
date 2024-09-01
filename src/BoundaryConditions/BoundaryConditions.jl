module BoundaryConditions

using LinearAlgebra
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods

export apply_boundary_condition!, is_boundary_point, compute_normal_vector, gradient_operator

"""
    apply_boundary_condition!(A::AbstractMatrix, b::AbstractVector, mesh::KSMesh, bc::AbstractKSBoundaryCondition)

Apply a boundary condition to the system matrix and right-hand side vector.
"""
function apply_boundary_condition!(A::AbstractMatrix, b::AbstractVector, mesh::KSMesh, bc::AbstractKSBoundaryCondition)
    for (i, element) in enumerate(mesh.elements)
        std_elem = SpectralMethods.get_or_create_standard_element(element.polynomial_degree)
        transformed_elem = SpectralMethods.transform_standard_element(std_elem, element.physical_domain, element.level)
        for (j, point) in enumerate(element.points)
            if is_boundary_point(point, mesh)
                global_index = mesh.location_matrices[i][j]
                apply_bc_at_point!(A, b, global_index, point, bc, element, transformed_elem)
            end
        end
    end
end

"""
    apply_bc_at_point!(A::AbstractMatrix, b::AbstractVector, index::Int, point::AbstractVector{T},
                       bc::AbstractKSBoundaryCondition, element::KSElement,
                       std_elem::StandardElement) where {T<:Real}

Apply a boundary condition at a specific point.
"""
function apply_bc_at_point!(A::AbstractMatrix, b::AbstractVector, index::Int, point::AbstractVector{T},
                            bc::AbstractKSBoundaryCondition, element::KSElement,
                            std_elem::StandardElement) where {T<:Real}
    if bc isa KSDirichletBC
        A[index, :] .= 0.0
        A[index, index] = 1.0
        b[index] = bc.value(point)
    elseif bc.isa(KSNeumannBC)
        normal = compute_normal_vector(point, element)
        grad_op = gradient_operator(point, element, std_elem)
        @inbounds for j in axes(A, 2)
            A[index, j] = dot(normal, grad_op[j, :])
        end
        b[index] = bc.flux(point)
    elseif bc.isa(KSRobinBC)
        normal = compute_normal_vector(point, element)
        grad_op = gradient_operator(point, element, std_elem)
        A[index, index] += bc.value(point)
        @inbounds for j in axes(A, 2)
            A[index, j] += bc.flux(point) * dot(normal, grad_op[j, :])
        end
        b[index] = bc.source(point)
    else
        error("Unsupported boundary condition type")
    end
end

"""
    is_boundary_point(point::AbstractVector{T}, mesh::KSMesh) -> Bool where {T<:Real}

Check if a point is on the boundary of the mesh.
"""
function is_boundary_point(point::AbstractVector{T}, mesh::KSMesh) where {T<:Real}
    # Check if the point's coordinates match any boundary point's coordinates
    return any(p -> all(isapprox.(p, point, rtol = 1e-10)), mesh.boundary_points)
end

"""
    compute_normal_vector(point::AbstractVector{T}, element::KSElement) -> AbstractVector{T} where {T<:Real}

Compute the outward normal vector at a boundary point.
"""
function compute_normal_vector(point::AbstractVector{T}, element::KSElement) where {T<:Real}
    dims = length(point)
    normal = zeros(T, dims)

    for i in 1:dims
        if isapprox(point[i], minimum(getindex.(element.points, i)), rtol = 1e-10)
            normal[i] = -1.0
        elseif isapprox(point[i], maximum(getindex.(element.points, i)), rtol = 1e-10)
            normal[i] = 1.0
        end
    end

    return normalize(normal)
end

"""
    gradient_operator(point::AbstractVector{T}, element::KSElement, std_elem::StandardElement) -> Matrix{T} where {T<:Real}

Compute the gradient operator at a point using the element's differentiation matrices.
"""
function gradient_operator(point::AbstractVector{T}, element::KSElement, std_elem::StandardElement) where {T<:Real}
    # Ensure differentiation matrices are available and correctly scaled
    if isnothing(element.differentiation_matrices)
        error("Differentiation matrices are not defined for this element.")
    end

    # Use the precomputed and scaled differentiation matrices from the StandardElement
    return hcat(std_elem.differentiation_matrices...)
end

end # module BoundaryConditions
