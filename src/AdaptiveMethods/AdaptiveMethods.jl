module AdaptiveMethods

using LinearAlgebra, SparseArrays, StaticArrays
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..ErrorEstimation, ..Preprocessing

export h_refine, p_refine, hp_refine, select_subelement_nodes
export hp_refine_superposition, superconvergence_refinement, adapt_mesh_superposition
export estimate_error_and_smoothness, compute_error_indicator

"""
    h_refine(element::KSElement{T,N}) -> Vector{KSElement{T,N}}

Perform h-refinement on a given element by splitting it into smaller elements.

# Arguments
- `element::KSElement{T,N}`: The element to be refined.

# Returns
- `Vector{KSElement{T,N}}`: A vector of new elements resulting from the refinement.
"""
function h_refine(element::KSElement{T,N}) where {T <: Number, N}
    new_elements = KSElement{T,N}[]
    dimension = N
    sub_element_size = CommonMethods.element_size(element) ./ 2

    mid_points = [(node1 .+ node2) ./ 2 for node1 in element.points for node2 in element.points if node1 != node2]
    unique_mid_points = unique(mid_points)

    for i in 1:(2^dimension)
        sub_element_nodes = select_subelement_nodes(element.points, unique_mid_points, i)
        new_points, new_weights = SpectralMethods.gauss_legendre_with_boundary_nd(
            element.polynomial_degree,
            dimension,
            [minimum(getindex.(sub_element_nodes, j)) for j in 1:dimension],
            [maximum(getindex.(sub_element_nodes, j)) for j in 1:dimension]
        )
        new_collocation_points = new_points

        diff_matrices = [SpectralMethods.derivative_matrix(getindex.(new_collocation_points, j)) for j in 1:dimension]

        new_element = KSElement{T,N}(
            id = element.id * 2^dimension + i,
            level = element.level + 1,
            polynomial_degree = element.polynomial_degree,
            parent = element.id,
            children = nothing,
            neighbors = nothing,
            error_estimate = 0.0,
            legendre_decay_rate = element.legendre_decay_rate
        )
        push!(new_elements, new_element)
    end

    return new_elements
end

"""
    p_refine(element::KSElement{T,N}) -> KSElement{T,N}

Perform p-refinement on a given element by increasing its polynomial degree.

# Arguments
- `element::KSElement{T,N}`: The element to be refined.

# Returns
- `KSElement{T,N}`: The element with increased polynomial degree.
"""
function p_refine(element::KSElement{T,N}) where {T <: Number, N}
    new_degree = element.polynomial_degree .+ 1
    dimension = N
    new_points, new_weights = SpectralMethods.gauss_legendre_with_boundary_nd(
        new_degree,
        dimension,
        [minimum(getindex.(element.points, j)) for j in 1:dimension],
        [maximum(getindex.(element.points, j)) for j in 1:dimension]
    )
    new_collocation_points = new_points

    new_diff_matrices = [SpectralMethods.derivative_matrix(getindex.(new_collocation_points, j)) for j in 1:dimension]
    new_basis_functions = CommonMethods.create_basis_functions(new_degree, dimension)

    return KSElement{T,N}(
        id = element.id,
        level = element.level,
        polynomial_degree = new_degree,
        parent = element.parent,
        children = element.children,
        neighbors = element.neighbors,
        error_estimate = element.error_estimate,
        legendre_decay_rate = element.legendre_decay_rate
    )
end

"""
    hp_refine(element::KSElement{T,N}, error::T, smoothness::T, threshold::T) -> Vector{KSElement{T,N}}

Perform hp-refinement on a given element based on error and smoothness indicators.

# Arguments
- `element::KSElement{T,N}`: The element to be refined.
- `error::T`: The error indicator for the element.
- `smoothness::T`: The smoothness indicator for the element.
- `threshold::T`: The threshold for deciding between h- and p-refinement.

# Returns
- `Vector{KSElement{T,N}}`: A vector of new elements resulting from the refinement.
"""
function hp_refine(element::KSElement{T,N}, error::T, smoothness::T, threshold::T) where {T <: Number, N}
    if error > threshold
        if smoothness > log(element.polynomial_degree[1]) / element.polynomial_degree[1]
            return [p_refine(element)]
        else
            return h_refine(element)
        end
    else
        return [element]
    end
end

"""
    select_subelement_nodes(nodes::Vector{SVector{N,T}}, mid_points::Vector{SVector{N,T}}, index::Int) -> Vector{SVector{N,T}}

Select the nodes for a subelement based on the index.

# Arguments
- `nodes::Vector{SVector{N,T}}`: The nodes of the parent element.
- `mid_points::Vector{SVector{N,T}}`: The mid-points of the parent element.
- `index::Int`: The index of the subelement.

# Returns
- `Vector{SVector{N,T}}`: The nodes for the subelement.
"""
function select_subelement_nodes(nodes::Vector{SVector{N,T}}, mid_points::Vector{SVector{N,T}}, index::Int) where {T <: Number, N}
    selected_nodes = SVector{N,T}[]
    for node in nodes
        if all(node[i] >= mid_points[i] for i in 1:length(mid_points) if (index - 1) & (1 << (i - 1)) != 0)
            push!(selected_nodes, node)
        end
    end
    return selected_nodes
end

"""
    hp_refine_superposition(element::KSElement{T,N}, error::T, smoothness::T, threshold::T) -> Vector{KSElement{T,N}}

Perform hp-refinement with superposition on a given element based on error and smoothness indicators.

# Arguments
- `element::KSElement{T,N}`: The element to be refined.
- `error::T`: The error indicator for the element.
- `smoothness::T`: The smoothness indicator for the element.
- `threshold::T`: The threshold for deciding between h- and p-refinement.

# Returns
- `Vector{KSElement{T,N}}`: A vector of new elements resulting from the refinement.
"""
function hp_refine_superposition(element::KSElement{T,N}, error::T, smoothness::T, threshold::T) where {T <: Number, N}
    if error > threshold
        if smoothness > log(element.polynomial_degree[1]) / element.polynomial_degree[1]
            return [p_refine(element)]
        else
            return h_refine(element)
        end
    else
        return [element]  # No refinement needed
    end
end

"""
    superconvergence_refinement(element::KSElement{T,N}, solution::Vector{T}, problem::AbstractKSProblem) -> Vector{KSElement{T,N}}

Perform refinement based on superconvergence principles.

# Arguments
- `element::KSElement{T,N}`: The element to be refined.
- `solution::Vector{T}`: The current solution vector.
- `problem::AbstractKSProblem`: The problem being solved.

# Returns
- `Vector{KSElement{T,N}}`: A vector of new elements resulting from the refinement.
"""
function superconvergence_refinement(element::KSElement{T,N}, solution::Vector{T}, problem::AbstractKSProblem) where {T <: Number, N}
    error, smoothness = estimate_error_and_smoothness(element, solution, problem)
    if smoothness > log(element.polynomial_degree[1]) / element.polynomial_degree[1]
        return [p_refine(element)]
    else
        return h_refine(element)
    end
end

"""
    adapt_mesh_superposition(mesh::KSMesh{T,N}, solution::Vector{T}, problem::AbstractKSProblem, threshold::T) -> KSMesh{T,N}

Adapt the mesh using superposition and superconvergence techniques.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh.
- `solution::Vector{T}`: The current solution vector.
- `problem::AbstractKSProblem`: The problem being solved.
- `threshold::T`: The error threshold for refinement.

# Returns
- `KSMesh{T,N}`: The adapted mesh.
"""
function adapt_mesh_superposition(mesh::KSMesh{T,N}, solution::Vector{T}, problem::AbstractKSProblem, threshold::T) where {T <: Number, N}
    new_elements = KSElement{T,N}[]

    for element in mesh.elements
        refined_elements = superconvergence_refinement(element, solution, problem)
        append!(new_elements, refined_elements)
    end

    new_mesh = KSMesh{T,N}(
        elements = new_elements,
        tensor_product_masks = [trues(tuple(fill(el.polynomial_degree[1] + 1, N)...)) for el in new_elements],
        location_matrices = [Dict{Int, Int}() for _ in 1:length(new_elements)],
        standard_elements = Dict{Tuple{Int, Int}, StandardElement{T, N}}(),  # Assuming standard_elements for new mesh
        global_error_estimate = sum([el.error_estimate for el in new_elements])
    )

    CommonMethods.update_tensor_product_masks!(new_mesh)
    CommonMethods.update_location_matrices!(new_mesh)

    return new_mesh
end

"""
    estimate_error_and_smoothness(element::KSElement{T,N}, solution::Vector{T}, problem::AbstractKSProblem) -> Tuple{T,T}

Estimate the error and smoothness for a given element.

# Arguments
- `element::KSElement{T,N}`: The element to estimate error and smoothness for.
- `solution::Vector{T}`: The current solution vector.
- `problem::AbstractKSProblem`: The problem being solved.

# Returns
- `Tuple{T,T}`: A tuple containing the estimated error and smoothness.
"""
function estimate_error_and_smoothness(element::KSElement{T,N}, solution::Vector{T}, problem::AbstractKSProblem) where {T <: Number, N}
    error = CommonMethods.estimate_error(element, solution, problem)
    smoothness = CommonMethods.estimate_smoothness(element, solution)
    return error, smoothness
end

"""
    compute_error_indicator(element::KSElement{T,N}, solution::Vector{T}, problem::AbstractKSProblem) -> Tuple{T,T}

Compute an error indicator for hp-adaptive refinement.

# Arguments
- `element::KSElement{T,N}`: The element to compute the error indicator for.
- `solution::Vector{T}`: The solution vector.
- `problem::AbstractKSProblem`: The problem being solved.

# Returns
- `Tuple{T,T}`: A tuple containing the error indicator and a smoothness measure.
"""
function compute_error_indicator(element::KSElement{T,N}, solution::Vector{T}, problem::AbstractKSProblem) where {T <: Number, N}
    return estimate_error_and_smoothness(element, solution, problem)
end

end # module AdaptiveMethods
