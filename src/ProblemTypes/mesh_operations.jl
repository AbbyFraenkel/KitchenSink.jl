using LinearAlgebra
using ..KSTypes, ..SpectralMethods, ..CoordinateSystems
using ..CacheManagement, ..NumericUtilities, ..Transforms, ..BoundaryConditions

const LEVEL_CACHE = CacheManager{Any}(100)

function update_mesh_connectivity!(mesh::KSMesh{T,N}) where {T,N}
    computational_nodes = map_nodes_to_computational_domain(mesh)
    update_cell_neighbors!(mesh)
    update_boundary_cells!(mesh)
    node_cell_map = create_node_cell_map(mesh)

    if !isnothing(mesh.transformation_data)
        update_node_map_with_transform!(mesh, computational_nodes, node_cell_map)
    else
        update_node_map!(mesh, computational_nodes, node_cell_map)
    end
    return mesh
end

function map_nodes_to_computational_domain(mesh::KSMesh{T,N}) where {T,N}
    comp_nodes = Dict{Int,Vector{T}}()
    for cell in mesh.cells
        !cell.is_fictitious || continue
        std_cell = SpectralMethods.get_or_create_standard_cell(cell.p, cell.level)
        for (local_idx, global_idx) in cell.node_map
            coords = ntuple(d -> std_cell.nodes_with_boundary[d][local_idx[d]], N)
            comp_nodes[global_idx] = collect(coords)
        end
    end
    return comp_nodes
end

function update_cell_neighbors!(mesh::KSMesh{T,N}) where {T,N}
    level_groups = group_cells_by_level(mesh)
    for (level, cells) in level_groups
        process_level_neighbors!(mesh, cells)
    end
end

function group_cells_by_level(mesh::KSMesh{T,N}) where {T,N}
    groups = Dict{Int,Vector{Int}}()
    for (i, cell) in enumerate(mesh.cells)
        !cell.is_fictitious || continue
        push!(get!(groups, cell.level, Int[]), i)
    end
    return groups
end

function process_level_neighbors!(mesh::KSMesh{T,N}, cell_indices::Vector{Int}) where {T,N}
    for i in cell_indices
        cell = mesh.cells[i]
        initialize_cell_neighbors!(cell, N)
        for j in cell_indices
            i == j && continue
            other = mesh.cells[j]
            if are_cells_neighbors(cell, other)
                dir = get_neighbor_direction(cell, other)
                cell.neighbors[dir] = j
                other.neighbors[get_opposite_direction(dir)] = i
            end
        end
    end
end

function initialize_cell_neighbors!(cell::KSCell{T,N}, dim::Int) where {T,N}
    cell.neighbors = Dict{Symbol,Int}()
    for d in 1:dim
        cell.neighbors[Symbol("dim$(d)_pos")] = -1
        cell.neighbors[Symbol("dim$(d)_neg")] = -1
    end
end

function are_cells_neighbors(cell1::KSCell{T,N}, cell2::KSCell{T,N}) where {T,N}
    shared_nodes = get_shared_nodes(cell1, cell2)
    return !isempty(shared_nodes)
end

function get_shared_nodes(cell1::KSCell{T,N}, cell2::KSCell{T,N}) where {T,N}
    std_cell1 = SpectralMethods.get_or_create_standard_cell(cell1.p, cell1.level)
    std_cell2 = SpectralMethods.get_or_create_standard_cell(cell2.p, cell2.level)
    shared = Vector{NTuple{N,Int}}()

    for (local1, global1) in cell1.node_map
        coords1 = ntuple(d -> std_cell1.nodes_with_boundary[d][local1[d]], N)
        for (local2, global2) in cell2.node_map
            coords2 = ntuple(d -> std_cell2.nodes_with_boundary[d][local2[d]], N)
            if are_coords_equal(coords1, coords2, T)
                push!(shared, local1)
                break
            end
        end
    end
    return shared
end

function are_coords_equal(c1::NTuple{N,T}, c2::NTuple{N,T}, ::Type{T}) where {N,T}
    return all(i -> abs(c1[i] - c2[i]) < sqrt(eps(T)), 1:N)
end

function get_neighbor_direction(cell1::KSCell{T,N}, cell2::KSCell{T,N}) where {T,N}
    std_cell1 = SpectralMethods.get_or_create_standard_cell(cell1.p, cell1.level)
    std_cell2 = SpectralMethods.get_or_create_standard_cell(cell2.p, cell2.level)

    center1 = get_cell_center(cell1, std_cell1)
    center2 = get_cell_center(cell2, std_cell2)

    diff = center2 .- center1
    abs_diff = abs.(diff)
    max_val, dim = findmax(abs_diff)

    side = diff[dim] > zero(T) ? :pos : :neg
    return Symbol("dim$(dim)_$(side)")
end

function get_cell_center(cell::KSCell{T,N}, std_cell::StandardKSCell{T,N}) where {T,N}
    return ntuple(d -> mean(std_cell.nodes_with_boundary[d]), N)
end

function update_boundary_cells!(mesh::KSMesh{T,N}) where {T,N}
    empty!(mesh.boundary_cells)
    boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)

    for cell_idx in mesh.physical_cells
        cell = mesh.cells[cell_idx]
        std_cell = SpectralMethods.get_or_create_standard_cell(cell.p, cell.level)

        for dim in 1:N
            process_dimension_boundary!(mesh, cell, cell_idx, dim, std_cell)
        end
    end
end

function process_dimension_boundary!(mesh::KSMesh{T,N}, cell::KSCell{T,N},
                                   cell_idx::Int, dim::Int,
                                   std_cell::StandardKSCell{T,N}) where {T,N}
    nodes = std_cell.nodes_with_boundary[dim]

    # Check negative boundary
    if is_on_boundary(nodes[1])
        add_to_boundary_cells!(mesh, cell_idx, dim, :neg)
    end

    # Check positive boundary
    if is_on_boundary(nodes[end])
        add_to_boundary_cells!(mesh, cell_idx, dim, :pos)
    end
end

function is_on_boundary(coord::T) where T
    return abs(coord + one(T)) < sqrt(eps(T)) || abs(coord - one(T)) < sqrt(eps(T))
end

function add_to_boundary_cells!(mesh::KSMesh{T,N}, cell_idx::Int,
                              dim::Int, side::Symbol) where {T,N}
    dir_sym = Symbol("dim$(dim)_$(side)")
    if !haskey(mesh.boundary_cells, dir_sym)
        mesh.boundary_cells[dir_sym] = Int[]
    end
    push!(mesh.boundary_cells[dir_sym], cell_idx)
end

function create_node_cell_map(mesh::KSMesh{T,N}) where {T,N}
    node_cell_map = Dict{Int,Vector{Int}}()
    for (cell_idx, cell) in enumerate(mesh.cells)
        !cell.is_fictitious || continue
        for global_idx in values(cell.node_map)
            if !haskey(node_cell_map, global_idx)
                node_cell_map[global_idx] = Int[]
            end
            push!(node_cell_map[global_idx], cell_idx)
        end
    end
    return node_cell_map
end

function update_node_map_with_transform!(mesh::KSMesh{T,N},
                                       computational_nodes::Dict{Int,Vector{T}},
                                       node_cell_map::Dict{Int,Vector{Int}}) where {T,N}
    physical_nodes = Dict{Int,Vector{T}}()
    for (global_idx, comp_coords) in computational_nodes
        physical_coords = Transforms.apply_transform(mesh.transformation_data, comp_coords)
        physical_nodes[global_idx] = physical_coords
    end
    update_node_map!(mesh, physical_nodes, node_cell_map)
end

function update_node_map!(mesh::KSMesh{T,N}, nodes::Dict{Int,Vector{T}},
                         node_cell_map::Dict{Int,Vector{Int}}) where {T,N}
    for cell in mesh.cells
        !cell.is_fictitious || continue
        new_map = Dict{NTuple{N,Int},Int}()
        for (local_idx, global_idx) in cell.node_map
            if haskey(nodes, global_idx)
                new_map[local_idx] = global_idx
            end
        end
        cell.node_map = new_map
    end
end

function get_opposite_direction(dir::Symbol)
    str = string(dir)
    m = match(r"dim(\d+)_(pos|neg)", str)
    isnothing(m) && throw(ArgumentError("Invalid direction format: $dir"))

    dim = m.captures[1]
    side = m.captures[2] == "pos" ? "neg" : "pos"
    return Symbol("dim$(dim)_$(side)")
end
