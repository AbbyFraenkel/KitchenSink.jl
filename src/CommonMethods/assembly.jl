"""
    assemble_system_matrix(mesh::KSMesh{T,N}) where {T<:Real,N} -> SparseMatrixCSC{T,Int}

Assemble the global system matrix for the given mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The mesh to assemble the matrix for

# Returns
- `SparseMatrixCSC{T,Int}`: The assembled system matrix

# Example
```julia
A = assemble_system_matrix(mesh)
```
"""
function assemble_system_matrix(mesh::KSMesh{T,N}) where {T<:Real,N}
    n = sum(length(el.basis_functions) for el in mesh.elements)
    A = spzeros(T, n, n)
    for element in mesh.elements
        A_local = assemble_local_system_matrix(element)
        indices = get_active_indices(element)
        A[indices, indices] += A_local
    end
    return A
end

"""
    assemble_rhs_vector(mesh::KSMesh{T,N}, f::Function) where {T<:Real,N} -> Vector{T}

Assemble the global right-hand side vector for the given mesh and source function.

# Arguments
- `mesh::KSMesh{T,N}`: The mesh to assemble the vector for
- `f::Function`: The source function

# Returns
- `Vector{T}`: The assembled right-hand side vector

# Example
```julia
b = assemble_rhs_vector(mesh, x -> sin(π * x[1]))
```
"""
function assemble_rhs_vector(mesh::KSMesh{T,N}, f::Function) where {T<:Real,N}
    n = sum(length(el.basis_functions) for el in mesh.elements)
    b = zeros(T, n)
    for element in mesh.elements
        b_local = [f(x.coordinates) for x in element.points]
        indices = get_active_indices(element)
        b[indices] += b_local
    end
    return b
end

"""
    assemble_local_system_matrix(element::KSElement{T}) where {T<:Real} -> Matrix{T}

Assemble the local system matrix for a given element.

# Arguments
- `element::KSElement{T}`: The element to assemble the matrix for

# Returns
- `Matrix{T}`: The assembled local system matrix

# Example
```julia
A_local = assemble_local_system_matrix(mesh.elements[1])
```
"""
function assemble_local_system_matrix(element::KSElement{T}) where {T<:Real}
    n = length(element.basis_functions)
    A_local = zeros(T, n, n)
    for i in 1:n, j in 1:n
        A_local[i, j] = inner_product(element.basis_functions[i], element.basis_functions[j], element)
    end
    return A_local
end

"""
    element_size(element::KSElement{T}) where {T<:Real} -> T

Compute the size (e.g., diameter) of a given element.

# Arguments
- `element::KSElement{T}`: The element to compute the size for

# Returns
- `T`: The size of the element

# Example
```julia
size = element_size(mesh.elements[1])
```
"""
function element_size(element::KSElement{T}) where {T<:Real}
    return maximum([norm(v1.coordinates .- v2.coordinates) for v1 in element.points for v2 in element.points])
end

"""
    estimate_smoothness(element::KSElement{T}, solution::AbstractVector{T}) where {T<:Real} -> T

Estimate the smoothness of the solution on a given element.

# Arguments
- `element::KSElement{T}`: The element to estimate smoothness for
- `solution::AbstractVector{T}`: The solution vector

# Returns
- `T`: The smoothness indicator

# Example
```julia
smoothness = estimate_smoothness(mesh.elements[1], solution)
```
"""
function estimate_smoothness(element::KSElement{T}, solution::AbstractVector{T}) where {T<:Real}
    coeffs = compute_expansion_coefficients(element, solution)
    return -log(norm(coeffs[end-1:end])) / element.polynomial_degree
end

"""
    get_active_indices(element::KSElement{T}) where {T<:Real} -> Vector{Int}

Get the active indices for a given element in the global solution vector.

# Arguments
- `element::KSElement{T}`: The element to get indices for

# Returns
- `Vector{Int}`: The active indices

# Example
```julia
indices = get_active_indices(mesh.elements[1])
```
"""
function get_active_indices(mesh::KSMesh)
    active_indices = Int[]
    for (i, element) in enumerate(mesh.elements)
        if mesh.tensor_product_masks[i]
            push!(active_indices, i)
        end
    end
    return active_indices
end

"""
    inner_product(f::KSBasisFunction, g::KSBasisFunction, element::KSElement{T}) where {T<:Real} -> T

Compute the inner product of two basis functions over a given element.

# Arguments
- `f::KSBasisFunction`: The first basis function
- `g::KSBasisFunction`: The second basis function
- `element::KSElement{T}`: The element to integrate over

# Returns
- `T`: The inner product

# Example
```julia
prod = inner_product(element.basis_functions[1], element.basis_functions[2], element)
```
"""
function inner_product(f::KSBasisFunction, g::KSBasisFunction, element::KSElement{T}) where {T<:Real}
    return sum(f.function_handle(x.coordinates) * g.function_handle(x.coordinates) * x.weight for x in element.points)
end

"""
    total_dofs(mesh::KSMesh{T,N}) where {T<:Real,N} -> Int

Calculate the total number of degrees of freedom in the mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The mesh to calculate the total DOFs for.

# Returns
- `Int`: The total number of degrees of freedom in the mesh.

# Example
```julia
total = total_dofs(mesh)
```
"""
function total_dofs(mesh::KSMesh{T,N}) where {T<:Real,N}
    return sum(element_dofs(el) for el in mesh.elements)
end

"""
    element_dofs(element::KSElement{T}) where {T<:Real} -> Int

Calculate the number of degrees of freedom in a single element.

# Arguments
- `element::KSElement{T}`: The element to calculate the DOFs for.

# Returns
- `Int`: The number of degrees of freedom in the element.

# Example
```julia
dofs = element_dofs(mesh.elements[1])
```
"""
function element_dofs(element::KSElement{T}) where {T<:Real}
    return length(element.basis_functions)
end
