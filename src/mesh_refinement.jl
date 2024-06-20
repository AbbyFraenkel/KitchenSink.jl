module MeshRefinement

export refine_mesh, coarsen_mesh

using ..Types
using ..SpectralMethods

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

end # module
