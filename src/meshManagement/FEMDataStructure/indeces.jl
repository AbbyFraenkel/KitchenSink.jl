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
