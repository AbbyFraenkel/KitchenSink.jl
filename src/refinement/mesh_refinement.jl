
function goal_oriented_refinement(
    elements::Vector{Element{Float64}},
    goal_quantity::Function,
    tolerance::Float64,
)::Vector{Element{Float64}}
    refined_elements = Element{Float64}[]
    for el in elements
        error = estimate_error(el, goal_quantity)
        if error > tolerance
            append!(refined_elements, refine_element(el))
        else
            push!(refined_elements, el)
        end
    end
    return refined_elements
end


function refine_element(el::Element{Float64})::Vector{Element{Float64}}
    new_elements = [
        Element{Float64}(
            el.nodes,
            el.basis_functions,
            el.active_basis_indices,
            el.removable_basis_indices,
            el.parent,
            el.level + 1,
            el.tensor_product_masks,
            el.location_matrix,
            el.error_estimate,
            el.boundary_information,
        ) for _ in 1:2
    ]
    return new_elements
end


function refine_mesh(
    mesh::Mesh,
    error_estimates::Vector{Float64},
    threshold::Float64,
)::Nothing
    for (i, el) in enumerate(mesh.elements)
        if error_estimates[i] > threshold
            if mesh.is_leaf[i]
                split_element!(mesh, i)
            end
        end
    end
end

function coarsen_mesh(
    mesh::Mesh,
    error_estimates::Vector{Float64},
    threshold::Float64,
)::Nothing
    for (i, el) in enumerate(mesh.elements)
        if error_estimates[i] < threshold / 2
            if !mesh.is_leaf[i]
                merge_elements!(mesh, i)
            end
        end
    end
end

function split_element!(mesh::Mesh, el_idx::Int)::Nothing
    el = mesh.elements[el_idx]
    new_elements = []
    for i = 1:2
        new_nodes = deepcopy(el.nodes)
        new_basis_functions = deepcopy(el.basis_functions)
        new_active_indices = deepcopy(el.active_basis_indices)
        push!(
            new_elements,
            Element(
                new_nodes,
                new_basis_functions,
                new_active_indices,
                el_idx,
                el.level + 1,
            ),
        )
    end
    mesh.elements[el_idx].is_leaf = false
    append!(mesh.elements, new_elements)
    update_neighbors!(mesh, el_idx)
end

function merge_elements!(mesh::Mesh, el_idx::Int)::Nothing
    el = mesh.elements[el_idx]
    children = find_children(mesh, el_idx)
    for child in children
        mesh.elements[child].is_leaf = false
    end
    mesh.elements[el_idx].is_leaf = true
    mesh.elements[el_idx].active_basis_indices = [
        el.active_basis_indices
        [mesh.elements[child].active_basis_indices for child in children]...
    ]
    update_neighbors!(mesh, el_idx)
end

function update_neighbors!(mesh::Mesh, el_idx::Int)::Nothing
    el = mesh.elements[el_idx]
    for d = 1:ndims(mesh.neighbors)
        for j = 1:size(mesh.neighbors, 3)
            if mesh.neighbors[el_idx, d, j] != -1
                neighbor_idx = mesh.neighbors[el_idx, d, j]
                neighbor = mesh.elements[neighbor_idx]
                if neighbor.level == el.level
                    mesh.neighbors[el_idx, d, j] = length(mesh.elements) - 2 + j
                    mesh.neighbors[length(mesh.elements)-2+j, d, 3-j] = el_idx
                end
            end
        end
    end
end

function find_children(mesh::Mesh, el_idx::Int)::Vector{Int}
    return findall(el -> el.parent == el_idx, mesh.elements)
end



function refine_by_superposition(
    domain::Vector{Float64},
    num_elements::Int,
    solution::Vector{Float64},
    p_old::Int,
    p_new::Int,
)::Vector{Float64}
    old_basis_functions, _, _ = hierarchical_basis_functions(length(domain), p_old)
    new_basis_functions, _, _ = hierarchical_basis_functions(length(domain), p_new)
    basis_functions = vcat(old_basis_functions, new_basis_functions)
    collocation_points, weights = legendre(p_new + 1)
    A, b = assemble_system(
        length(domain),
        domain,
        num_elements,
        basis_functions,
        collocation_points,
        weights,
    )
    boundary_conditions = []
    A, b = apply_boundary_conditions!(A, b, boundary_conditions)
    A, b = ensure_higher_order_continuity!(A, b, num_elements, length(new_basis_functions))
    new_solution = A \ b
    refined_solution = solution + new_solution[end-length(new_basis_functions)+1:end]
    return refined_solution
end




@doc """
    refine_grid(layer::GridLayer, marked_elements::Vector{Int})::GridLayer

Refine the grid at the given level.

# Arguments
- `layer::GridLayer`: Grid layer at the current level.
- `marked_elements::Vector{Int}`: Indices of elements to be refined.

# Returns
- `refined_layer::GridLayer`: Refined grid layer.
"""
function refine_grid(layer::GridLayer, marked_elements::Vector{Int})::GridLayer
    refined_nodes = Vector{Vector{Float64}}()
    refined_elements = Vector{Element}()
    for i = 1:(length(layer.elements)-1)
        push!(refined_nodes, layer.elements[i].nodes)
        push!(refined_elements, layer.elements[i])
        if i in marked_elements
            midpoint = (layer.elements[i].nodes .+ layer.elements[i+1].nodes) ./ 2
            push!(refined_nodes, midpoint)
            new_element = deepcopy(layer.elements[i])
            new_element.nodes = midpoint
            push!(refined_elements, new_element)
        end
    end
    push!(refined_nodes, layer.elements[end].nodes)
    push!(refined_elements, layer.elements[end])

    new_layer = deepcopy(layer)
    new_layer.nodes = refined_nodes
    new_layer.elements = refined_elements
    new_layer.levels .= maximum(new_layer.levels) + 1

    return new_layer
end
