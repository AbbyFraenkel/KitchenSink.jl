"""
    h_refine(element::KSElement)::Vector{KSElement}

Perform h-refinement on a given element by splitting it into smaller elements.

# Arguments
- `element::KSElement`: The element to be refined.

# Returns
- `Vector{KSElement}`: A vector of new elements resulting from the refinement.
"""

function h_refine(element::KSTypes.KSElement)::Vector{KSTypes.KSElement}
    new_elements = Vector{KSTypes.KSElement}()
    mid_points = [(node1.coordinates .+ node2.coordinates) ./ 2 for node1 in element.points for node2 in element.points if node1 != node2]
    unique_mid_points = unique(mid_points)

    for i in 1:2^element.dimension
        sub_element_nodes = select_subelement_nodes(element.points, unique_mid_points, i)
        new_points, new_weights = KSTypes.SpectralMethods.gauss_legendre_with_boundary_nd(element.polynomial_degree, element.dimension)
        new_element = KSTypes.KSElement(
            element.id * 2^element.dimension + i,
            sub_element_nodes,
            element.basis_functions,
            Vector{Union{Nothing,KSTypes.KSElement}}(nothing, 2^element.dimension),
            element,
            Vector{KSTypes.KSElement}(),
            element.level + 1,
            element.polynomial_degree,
            0.0,
            KSTypes.KSPoints(new_points, new_weights)
        )
        push!(new_elements, new_element)
    end

    return new_elements
end
"""
    p_refine(element::KSElement)::KSElement

Perform p-refinement on a given element by increasing its polynomial degree.

# Arguments
- `element::KSElement`: The element to be refined.

# Returns
- `KSElement`: The element with increased polynomial degree.
"""
function p_refine(element::KSTypes.KSElement)::KSTypes.KSElement
    new_degree = element.polynomial_degree + 1
    new_points, new_weights = KSTypes.SpectralMethods.gauss_legendre_with_boundary_nd(new_degree, element.dimension)

    return KSTypes.KSElement(
        element.id,
        element.points,
        element.basis_functions,
        element.neighbors,
        element.parent,
        element.children,
        element.level,
        new_degree,
        element.error_estimate,
        KSTypes.KSPoints(new_points, new_weights)
    )
end
"""
    hp_refine(element::KSElement, error::T, smoothness::T, threshold::T)::Vector{KSElement} where {T}

Perform hp-refinement on a given element based on error and smoothness indicators.

# Arguments
- `element::KSElement`: The element to be refined.
- `error::T`: The error indicator for the element.
- `smoothness::T`: The smoothness indicator for the element.
- `threshold::T`: The threshold for deciding between h- and p-refinement.

# Returns
- `Vector{KSElement}`: A vector of new elements resulting from the refinement.
"""
function hp_refine(element::KSElement, error::T, smoothness::T, threshold::T)::Vector{KSElement} where {T<:Real}
    if error > threshold
        if smoothness > log(element.polynomial_degree) / element.polynomial_degree
            return [p_refine(element)]
        else
            return h_refine(element)
        end
    else
        return [element]
    end
end


"""
    select_subelement_nodes(nodes::Vector{KSPoint}, index::Int, dim::Int)::Vector{KSPoint}

Select the nodes for a subelement based on the index.

# Arguments
- `nodes::Vector{KSPoint}`: The nodes of the parent element.
- `index::Int`: The index of the subelement.
- `dim::Int`: The dimension of the element.

# Returns
- `Vector{KSPoint}`: The nodes for the subelement.
"""
function select_subelement_nodes(nodes::Vector{KSPoint}, mid_points::Vector{Vector{T}}, index::Int)::Vector{KSPoint} where {T<:Real}
    selected_nodes = Vector{KSPoint}()
    for node in nodes
        if all(node.coordinates .>= mid_points[i] for i in 1:length(mid_points) if (index - 1) & (1 << (i - 1)) != 0)
            push!(selected_nodes, node)
        end
    end
    return selected_nodes
end
