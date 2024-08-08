
"""
    create_mesh_hierarchy(base_mesh::AbstractKSMesh, max_levels::Int) -> Vector{AbstractKSMesh}

Create a hierarchy of meshes starting from a base mesh.

# Arguments
- `base_mesh::AbstractKSMesh`: The initial mesh
- `max_levels::Int`: The maximum number of levels in the hierarchy

# Returns
- `Vector{AbstractKSMesh}`: A vector of meshes representing the hierarchy

# Example
```julia
hierarchy = create_mesh_hierarchy(base_mesh, 3)
```
"""
function create_mesh_hierarchy(base_mesh::AbstractKSMesh, max_levels::Int)
    hierarchy = [base_mesh]
    for _ in 2:max_levels
        push!(hierarchy, refine_mesh_uniformly(hierarchy[end]))
    end
    return hierarchy
end

"""
    coarsen_mesh(mesh::AbstractKSMesh) -> AbstractKSMesh

Create a coarser version of the given mesh.

# Arguments
- `mesh::AbstractKSMesh`: The mesh to coarsen

# Returns
- `AbstractKSMesh`: The coarsened mesh

# Example
```julia
coarse_mesh = coarsen_mesh(fine_mesh)
```
"""

function coarsen_mesh(mesh::AbstractKSMesh)
    coarse_elements = [merge_elements(mesh.elements[i], mesh.elements[i+1]) for i in 1:2:length(mesh.elements)-1]
    return typeof(mesh)(coarse_elements, mesh.boundaries, mesh.coordinate_system)
end

"""
    merge_elements(el1::AbstractKSElement, el2::AbstractKSElement) -> AbstractKSElement

Merge two elements into a single, coarser element.

# Arguments
- `el1::AbstractKSElement`: The first element to merge
- `el2::AbstractKSElement`: The second element to merge

# Returns
- `AbstractKSElement`: The merged element

# Example
```julia
merged_element = merge_elements(mesh.elements[1], mesh.elements[2])
```
"""
function merge_elements(el1::AbstractKSElement, el2::AbstractKSElement)
    new_vertices = unique(vcat(el1.vertices, el2.vertices))
    new_basis_functions = vcat(el1.basis_functions, el2.basis_functions)
    new_level = min(el1.level, el2.level) - 1
    return typeof(el1)(el1.id, new_vertices, new_basis_functions, [], nothing, [el1, el2], new_level, el1.polynomial_degree)
end

"""
    refine_mesh_uniformly(mesh::AbstractKSMesh) -> AbstractKSMesh

Refine all elements in the given mesh uniformly.

# Arguments
- `mesh::AbstractKSMesh`: The mesh to refine

# Returns
- `AbstractKSMesh`: The refined mesh

# Example
```julia
refined_mesh = refine_mesh_uniformly(mesh)
```
"""
function refine_mesh_uniformly(mesh::AbstractKSMesh)
    new_elements = vcat([AdaptiveMethods.h_refine(el) for el in mesh.elements]...)
    return typeof(mesh)(new_elements, mesh.boundaries, mesh.coordinate_system)
end
"""
    refine_mesh_hierarchy(hierarchy::Vector{<:AbstractKSMesh}, level::Int, marked_elements::AbstractVector{Int}) -> Vector{AbstractKSMesh}

Refine specific elements in a mesh hierarchy at a given level.

# Arguments
- `hierarchy::Vector{<:AbstractKSMesh}`: The current mesh hierarchy
- `level::Int`: The level at which to perform refinement
- `marked_elements::AbstractVector{Int}`: Indices of elements to be refined

# Returns
- `Vector{AbstractKSMesh}`: The updated mesh hierarchy

# Example
```julia
new_hierarchy = refine_mesh_hierarchy(hierarchy, 2, [1, 3, 5])
```
"""
function refine_mesh_hierarchy(hierarchy::Vector{<:AbstractKSMesh}, level::Int, marked_elements::AbstractVector{Int})
    if level < 1 || level > length(hierarchy)
        throw(ArgumentError("Invalid refinement level"))
    end

    refined_mesh = refine_marked_elements(hierarchy[level], marked_elements)
    new_hierarchy = copy(hierarchy)
    new_hierarchy[level] = refined_mesh

    for l in (level+1):length(new_hierarchy)
        new_hierarchy[l] = adjust_finer_level(new_hierarchy[l-1], new_hierarchy[l])
    end

    return new_hierarchy
end

"""
    refine_marked_elements(mesh::AbstractKSMesh, marked_elements::AbstractVector{Int}) -> AbstractKSMesh

Refine specific elements in a mesh.

# Arguments
- `mesh::AbstractKSMesh`: The mesh to refine
- `marked_elements::AbstractVector{Int}`: Indices of elements to be refined

# Returns
- `AbstractKSMesh`: The refined mesh

# Example
```julia
refined_mesh = refine_marked_elements(mesh, [1, 3, 5])
```
"""


function refine_marked_elements(mesh::AbstractKSMesh, marked_elements::AbstractVector{Int})
    new_elements = []
    for (i, el) in enumerate(mesh.elements)
        if i in marked_elements
            append!(new_elements, AdaptiveMethods.h_refine(el))
        else
            push!(new_elements, el)
        end
    end
    return typeof(mesh)(new_elements, mesh.boundaries, mesh.coordinate_system)
end

"""
    adjust_finer_level(coarse_mesh::AbstractKSMesh, fine_mesh::AbstractKSMesh) -> AbstractKSMesh

Adjust a finer level mesh based on changes in the coarser level.

# Arguments
- `coarse_mesh::AbstractKSMesh`: The refined coarser level mesh
- `fine_mesh::AbstractKSMesh`: The current finer level mesh

# Returns
- `AbstractKSMesh`: The adjusted finer level mesh

# Example
```julia
adjusted_fine_mesh = adjust_finer_level(coarse_mesh, fine_mesh)
```
"""

function adjust_finer_level(coarse_mesh::AbstractKSMesh, fine_mesh::AbstractKSMesh)
    new_elements = []
    for fine_el in fine_mesh.elements
        if fine_el.parent in coarse_mesh.elements
            push!(new_elements, fine_el)
        else
            append!(new_elements, AdaptiveMethods.h_refine(fine_el.parent))
        end
    end
    return typeof(fine_mesh)(new_elements, fine_mesh.boundaries, fine_mesh.coordinate_system)
end
