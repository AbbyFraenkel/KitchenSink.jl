@inline function estimate_residual(
    A::Matrix{Float64},
    x::Vector{Float64},
    b::Vector{Float64},
)::Float64
    residual = b - A * x
    return norm(residual)
end

function compute_residuals(
    elements::Vector{Element},
    solution::Vector{Float64},
    A::Function,
    f::Function,
)::Vector{Float64}
    residuals = Vector{Float64}()
    for el in elements
        push!(residuals, compute_residual(el, solution, A, f))
    end
    return residuals
end

@inline function evaluate_at_superconvergent_points(
    solution::Vector{Float64},
    superconvergent_points::Vector{Float64},
)::Vector{Float64}
    return [solution[pt] for pt in superconvergent_points]
end

function recover_derivatives(values::Vector{Float64}, method::Symbol)::Vector{Float64}
    if method == :superconvergence
        return superconvergence_recovery(values)
    else
        error("Unknown method: $method")
    end
end

function superconvergence_recovery(values::Vector{Float64})::Vector{Float64}
    n = length(values)
    recovered_derivatives = zeros(Float64, n)

    Threads.@threads for i = 2:n-1
        recovered_derivatives[i] = (values[i+1] - values[i-1]) / 2
    end

    recovered_derivatives[1] = values[2] - values[1]
    recovered_derivatives[n] = values[n] - values[n-1]

    return recovered_derivatives
end


@inline function compute_smoothness_indicator(coefficients::Vector{Float64})::Float64
    decay_rate = norm(coefficients[end]) / norm(coefficients[1])
    return decay_rate
end


@inline function determine_refinement_strategy(
    smoothness_indicator::Float64,
    threshold::Float64,
)::Symbol
    if smoothness_indicator < threshold
        return :h_refinement
    else
        return :p_refinement
    end
    end

"""
    altRefine(mesh::Mesh)

Alternative refinement strategy for the given mesh.
"""
function altRefine(mesh::Mesh)
    refined_mesh = deepcopy(mesh)
    for element in mesh.elements
        refine_element!(element)
    end
    return refined_mesh
end

"""
    adaptiveRefinement(mesh::Mesh, criteria::Function)

Perform adaptive refinement on the mesh based on the given criteria.
"""
function adaptiveRefinement(mesh::Mesh, criteria::Function)
    refined_mesh = deepcopy(mesh)
    for element in mesh.elements
        if criteria(element)
            refine_element!(element)
        end
    end
    return refined_mesh
end

"""
    refineMesh(mesh::Mesh, levels::Int)

Refine the mesh to the specified number of levels.
"""
function refineMesh(mesh::Mesh, levels::Int)
    refined_mesh = deepcopy(mesh)
    for i in 1:levels
        for element in refined_mesh.elements
            refine_element!(element)
        end
    end
    return refined_mesh
end

"""
    refine_element!(element::Element{T}) where T<:AbstractFloat

Refine the given element.

# Arguments
- `element::Element`: The element to refine.

# Returns
- `element::Element`: The refined element.
"""

"""
    refine_element!(element::Element)

Refine the given element.

# Arguments
- `element::Element`: The element to refine.

# Returns
- `element::Element`: The refined element.
"""
function refine_element!(element::Element)
    # Split each dimension into 2
    new_nodes = Vector{Node{Float64}}()
    new_weights = Vector{Vector{Float64}}()

    for i in 1:length(element.nodes)
        mid_points = (element.nodes[i][1:end-1] + element.nodes[i][2:end]) / 2
        new_nodes[i] = [element.nodes[i]; mid_points]
        new_nodes[i] = sort(new_nodes[i])

        mid_weights = (element.weights[i][1:end-1] + element.weights[i][2:end]) / 2
        new_weights[i] = [element.weights[i]; mid_weights]
        new_weights[i] = sort(new_weights[i])
    end

    element.nodes = new_nodes
    element.weights = new_weights
    element.tensor_data = zeros(Float64, size(element.tensor_data) .+ 1)
    element.solution = zeros(Float64, size(element.solution) .+ 1)
    element.residuals = zeros(Float64, size(element.residuals) .+ 1)
    element.refined = true
    element.connectivity = create_initial_connectivity(length(new_nodes[1]), length(new_nodes))
    element.level += 1

    return element
end


"""
Calculate the sensitivity of the polynomial order for the given solution.

# Arguments
- `solution::AbstractVector`: The approximate solution vector.
- `D1::AbstractMatrix`: The first derivative matrix.
- `k::Int`: The current polynomial order.
- `k_max::Int`: The maximum polynomial order.

# Returns
- `change_in_smoothness::Real`: The change in smoothness measure between the current and next polynomial order.

# Notes
- This function computes the smoothness measure at the current and next polynomial order.
- If the current polynomial order is the maximum, the sensitivity is zero.
- The smoothness measure is calculated using the `calculate_smoothness` function.
"""
function polynomial_order_sensitivity(solution, D1, k, k_max)
    # solution: Approximate solution vector
    # Dks: Array of derivative matrices
    # k: Current polynomial order
    xkplus, wkplus = ShiftedAdaptedGaussLegendre(k + 1)
    D1plus = EO_matrix_derivative(xkplus)

    # Compute the smoothness measure at the current polynomial order
    smoothness_current = calculate_smoothness(D1 * solution)

    # Compute the smoothness measure at the next polynomial order, if possible
    if k < k_max
        smoothness_next = calculate_smoothness(D1plus * solution)
    else
        # If the current polynomial order is the maximum, the sensitivity is zero
        return 0.0
    end

    # Compute the change in the smoothness measure
    change_in_smoothness = smoothness_next - smoothness_current

    return change_in_smoothness
end


"""
Calculate the curvature of the solution based on the second derivative norm.

# Arguments
- `solution::AbstractVector`: The solution vector.
- `D2::AbstractMatrix`: The second derivative matrix.
- `curvature_threshold::Real`: The threshold for curvature.

# Returns
- `curvature_norm::Real`: The norm of the curvature vector.

# Notes
- The curvature of the solution is determined by the second derivative norm.
- If the curvature norm is small, the solution is considered smooth.
- This function is used to determine if the solution is smooth enough for further refinement.
"""
function calculate_curvature(solution, D2, curvature_threshold)
    curvature = D2 * solution
    curvature_norm = norm(curvature)
    return curvature_norm
end

"""
Calculate the smoothness of the solution based on the gradient norm.

# Arguments
- `solution::AbstractVector`: The solution vector.
- `D1::AbstractMatrix`: The first derivative matrix.

# Returns
- `gradient_norm::Real`: The norm of the gradient vector.

# Notes
- The smoothness of the solution is determined by the gradient norm.
- If the gradient norm is small, the solution is considered smooth.
"""
function calculate_smoothness(solution, D1)
    # Compute the gradient of the solution
    gradient = D1 * solution
    # Compute the norm of the gradient
    gradient_norm = norm(gradient)
    # Return the gradient norm
    return gradient_norm
end


