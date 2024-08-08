"""
    update_tensor_product_masks!(mesh::KSMesh{T,N}) where {T<:Real,N}

Update the tensor product masks for the given mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh.

# Examples
```julia
update_tensor_product_masks!(mesh)
```
"""
function update_tensor_product_masks!(mesh::KSMesh{T,N}) where {T<:Real,N}
    isempty(mesh.elements) && throw(ArgumentError("Mesh must contain elements"))

    # Initialize masks
    for (i, element) in enumerate(mesh.elements)
        mesh.tensor_product_masks[i] = trues(fill(element.polynomial_degree + 1, N)...)
    end

    # Restore interface compatibility
    for _ in 1:(N-1)
        for element in mesh.elements
            for (neighbor_idx, neighbor) in enumerate(element.neighbors)
                if !isnothing(neighbor) && neighbor.level == element.level
                    direction = div(neighbor_idx - 1, 2) + 1
                    side = mod(neighbor_idx - 1, 2)
                    mesh.tensor_product_masks[element.id] = restore_interface_compatibility!(
                        mesh.tensor_product_masks[element.id],
                        mesh.tensor_product_masks[neighbor.id],
                        direction,
                        side
                    )
                end
            end
        end
    end

    # Deactivate internal boundary functions
    for element in mesh.elements
        for (neighbor_idx, neighbor) in enumerate(element.neighbors)
            if !isnothing(neighbor) && neighbor.level < element.level
                direction = div(neighbor_idx - 1, 2) + 1
                side = mod(neighbor_idx - 1, 2)
                mesh.tensor_product_masks[element.id] = deactivate_internal_boundary!(
                    mesh.tensor_product_masks[element.id],
                    direction,
                    side
                )
            end
        end
    end

    # Restore interface compatibility again
    for _ in 1:(N-2)
        for element in mesh.elements
            for (neighbor_idx, neighbor) in enumerate(element.neighbors)
                if !isnothing(neighbor) && neighbor.level == element.level
                    direction = div(neighbor_idx - 1, 2) + 1
                    side = mod(neighbor_idx - 1, 2)
                    mesh.tensor_product_masks[element.id] = restore_interface_compatibility!(
                        mesh.tensor_product_masks[element.id],
                        mesh.tensor_product_masks[neighbor.id],
                        direction,
                        side,
                        :and
                    )
                end
            end
        end
    end

    # Update location matrices
    update_location_matrices!(mesh)

    nothing
end

"""
    restore_interface_compatibility!(mask1::BitArray, mask2::BitArray, direction::Int, side::Int, op::Symbol=:or)

Restore interface compatibility between two tensor product masks.

# Arguments
- `mask1::BitArray`: The first tensor product mask.
- `mask2::BitArray`: The second tensor product mask.
- `direction::Int`: The direction of the interface.
- `side::Int`: The side of the interface.
- `op::Symbol`: The operation to use (:or or :and).

# Returns
- `BitArray`: The updated tensor product mask.

# Examples
```julia
result = restore_interface_compatibility!(mask1, mask2, 1, 1)
```
"""
function restore_interface_compatibility!(mask1::BitArray, mask2::BitArray, direction::Int, side::Int, op::Symbol=:or)
    size(mask1) == size(mask2) || throw(ArgumentError("Masks must have the same size"))
    1 <= direction <= ndims(mask1) || throw(ArgumentError("Invalid direction"))
    side in (0, 1) || throw(ArgumentError("Side must be 0 or 1"))
    op in (:or, :and) || throw(ArgumentError("Operation must be :or or :and"))

    slice1 = selectdim(mask1, direction, side == 0 ? 1 : size(mask1, direction))
    slice2 = selectdim(mask2, direction, side == 0 ? size(mask2, direction) : 1)

    if op == :or
        slice1 .|= slice2
        slice2 .|= slice1
    else
        slice1 .&= slice2
        slice2 .&= slice1
    end

    mask1
end

"""
    deactivate_internal_boundary!(mask::BitArray, direction::Int, side::Int)

Deactivate the internal boundary for the given tensor product mask.

# Arguments
- `mask::BitArray`: The tensor product mask.
- `direction::Int`: The direction of the boundary.
- `side::Int`: The side of the boundary.

# Returns
- `BitArray`: The updated tensor product mask.

# Examples
```julia
result = deactivate_internal_boundary!(mask, 1, 0)
```
"""
function deactivate_internal_boundary!(mask::BitArray, direction::Int, side::Int)
    1 <= direction <= ndims(mask) || throw(ArgumentError("Invalid direction"))
    side in (0, 1) || throw(ArgumentError("Side must be 0 or 1"))

    slice = selectdim(mask, direction, side == 0 ? 1 : size(mask, direction))
    slice .= false
    mask
end

"""
    update_location_matrices!(mesh::KSMesh{T,N}) where {T<:Real,N}

Update the location matrices for the given mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh.

# Examples
```julia
update_location_matrices!(mesh)
```
"""
function update_location_matrices!(mesh::KSMesh{T,N}) where {T<:Real,N}
    isempty(mesh.elements) && throw(ArgumentError("Mesh must contain elements"))

    global_index = 1
    for (i, element) in enumerate(mesh.elements)
        local_to_global = Dict{Int,Int}()
        for idx in CartesianIndices(mesh.tensor_product_masks[i])
            if mesh.tensor_product_masks[i][idx]
                local_to_global[linear_index(idx, size(mesh.tensor_product_masks[i]))] = global_index
                global_index += 1
            end
        end
        mesh.location_matrices[i] = local_to_global
    end

    nothing
end

"""
    linear_index(idx::CartesianIndex, sz::NTuple

{N,Int}) where {N}

Compute the linear index for the given Cartesian index and size.

# Arguments
- `idx::CartesianIndex`: The Cartesian index.
- `sz::NTuple{N,Int}`: The size of the array.

# Returns
- `Int`: The linear index.

# Examples
```julia
index = linear_index(CartesianIndex(2, 2), (3, 3))
```
"""
function linear_index(idx::CartesianIndex, sz::NTuple{N,Int}) where {N}
    all(1 .<= idx.I .<= sz) || throw(ArgumentError("Index out of bounds"))
    sum((idx.I .- 1) .* cumprod([1; collect(sz[1:end-1])]))
end

"""
    update_tensor_product_masks_with_trunk!(mesh::KSMesh{T,N}) where {T<:Real,N}

Update the tensor product masks with trunk for the given mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh.

# Examples
```julia
update_tensor_product_masks_with_trunk!(mesh)
```
"""
function update_tensor_product_masks_with_trunk!(mesh::KSMesh{T,N}) where {T<:Real,N}
    isempty(mesh.elements) && throw(ArgumentError("Mesh must contain elements"))

    for element in mesh.elements
        mask_size = fill(element.polynomial_degree + 1, N)
        mask = falses(mask_size...)
        for idx in CartesianIndices(mask)
            if all(i -> i <= element.polynomial_degree + 1, Tuple(idx))
                mask[idx] = true
            end
        end
        mesh.tensor_product_masks[element.id] = mask
    end

    # Apply compatibility and boundary deactivation
    update_tensor_product_masks!(mesh)

    nothing
end



"""
    apply_boundary_conditions!(u::AbstractArray{T}, problem::AbstractKSProblem{T}, element::KSElement{T}, mask::BitArray{N}) where {T<:Real,N}

Apply the boundary conditions for the given problem and element.

# Arguments
- `u::AbstractArray{T}`: The current solution.
- `problem::AbstractKSProblem{T}`: The problem to be solved.
- `element::KSElement{T}`: The current element.
- `mask::BitArray{N}`: The tensor product mask.

# Examples
```julia
apply_boundary_conditions!(u, problem, element, mask)
```
"""
function apply_boundary_conditions!(u::AbstractArray{T}, problem::AbstractKSProblem{T}, element::KSElement{T}, mask::BitArray{N}) where {T<:Real,N}
    for idx in CartesianIndices(u)
        if mask[idx] && is_boundary_node(element.points[idx], element)
            u[idx] = problem.boundary_conditions(element.points[idx].coordinates)
        end
    end
    nothing
end

"""
    enforce_continuity!(solution::Vector{AbstractArray{T,N}}, mesh::KSMesh{T,N}) where {T<:Real,N}

Enforce continuity for the given solution and mesh.

# Arguments
- `solution::Vector{AbstractArray{T,N}}`: The current solution.
- `mesh::KSMesh{T,N}`: The current mesh.

# Examples
```julia
enforce_continuity!(solution, mesh)
```
"""
function enforce_continuity!(solution::Vector{AbstractArray{T,N}}, mesh::KSMesh{T,N}) where {T<:Real,N}
    for (i, element) in enumerate(mesh.elements)
        for (neighbor_idx, neighbor) in enumerate(element.neighbors)
            if !isnothing(neighbor) && neighbor.level == element.level
                direction = div(neighbor_idx - 1, 2) + 1
                side = mod(neighbor_idx - 1, 2)
                restore_interface_compatibility!(
                    mesh.tensor_product_masks[i],
                    mesh.tensor_product_masks[neighbor.id],
                    direction,
                    side,
                    :and
                )
            end
        end
    end

    for (i, element) in enumerate(mesh.elements)
        solution[i] .*= mesh.tensor_product_masks[i]
    end
    nothing
end


"""
    is_boundary_node(point::KSPoint{T}, element::KSElement{T}) where {T<:Real}

Check if the given point is a boundary node.

# Arguments
- `point::KSPoint{T}`: The point to check.
- `element::KSElement{T}`: The element containing the point.

# Returns
- `Bool`: `true` if the point is a boundary node, `false` otherwise.

# Examples
```julia
is_boundary_node(element.points[1], element)
```
"""
function is_boundary_node(point::KSPoint{T}, element::KSElement{T}) where {T<:Real}
    return any(p -> p.coordinates == point.coordinates, element.points[[1, end]])
end
