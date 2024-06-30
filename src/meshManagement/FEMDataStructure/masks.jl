"""
    createMask(grid::AbstractArray)

Create a mask for the given grid.
"""
function createMask(grid::AbstractArray)
    mask = trues(size(grid))
    return mask
end

"""
    applyMask(data::AbstractArray, mask::AbstractArray)

Apply the given mask to the data.
"""
function applyMask(data::AbstractArray, mask::AbstractArray)
    masked_data = data .* mask
    return masked_data
end

function initializeGlobalIndices(M::MaskList, mesh_level::Int)
    G = Vector{LocationMatrix}()
    nids = 0

    for Mi in M.masks
        Gi = LocationMatrix(zeros(Int, sizes(Mi)))
        for j in productIndices(sizes(Mi))
            Gi.indices[j] = Mi[j] ? nids : -1
            nids += 1
        end
        push!(G, Gi)
    end

    return G, nids, mesh_level
end

function removeUnassignedIndices(G::Vector{LocationMatrix}, nids::Int, mesh_level::Int)
    exists = zeros(Int, nids)
    map = fill(-1, nids)

    for Gi in G
        for j in productIndices(sizes(Gi))
            if Gi.indices[j] != -1
                exists[Gi.indices[j]] = 1
            end
        end
    end

    nnew = 0
    for i in 1:nids
        map[i] = exists[i] ? nnew : -1
        nnew += 1
    end

    for Gi in G
        for j in productIndices(sizes(Gi))
            if Gi.indices[j] != -1
                Gi.indices[j] = map[Gi.indices[j]]
            end
        end
    end

    mesh_level += 1  # Update mesh level
end

function create_tensor_product_masks(elements::Vector{Element})
    masks = Dict{Int,Array{Bool}}()
    for element in elements
        if element.is_leaf
            mask_size = tuple((p + 1 for p in element.polynomial_degrees)...)
            mask = trues(mask_size)
            for bf in element.basis_functions
                if !bf.active
                    mask[CartesianIndex(bf.id)] = false
                end
            end
            masks[element.id] = mask
        else
            masks[element.id] = Bool[]
        end
    end
    return masks
end

function compute_tensor_product_masks(element::Element)
    dims = ndims(element.nodes)
    masks = Vector{Matrix{Bool}}(undef, dims)
    for i in 1:dims
        masks[i] = trues(length(element.nodes[i]), length(element.nodes[i]))
    end
    return masks
end



function restore_interface_compatibility!(masks, elements)
    for element in elements
        for neighbor_id in element.neighbors
            if neighbor_id != -1 && !isempty(masks[neighbor_id])
                mask = masks[element.id]
                neighbor_mask = masks[neighbor_id]
                masks[element.id] = mask .| neighbor_mask
            end
        end
    end
end


function deactivate_internal_boundary_functions!(masks, elements)
    for element in elements
        if !element.is_leaf
            for neighbor_id in element.neighbors
                if neighbor_id != -1 && elements[neighbor_id].level < element.level
                    masks[element.id][neighbor_id] = false
                end
            end
        end
    end
end

function create_location_matrices(masks, elements)
    location_matrices = Dict{Int,Array{Int}}()
    global_index = 1
    for element in elements
        mask = masks[element.id]
        location_matrix = fill(-1, size(mask))
        for idx in CartesianIndices(mask)
            if mask[idx]
                location_matrix[idx] = global_index
                global_index += 1
            end
        end
        location_matrices[element.id] = location_matrix
    end
    return location_matrices
end
