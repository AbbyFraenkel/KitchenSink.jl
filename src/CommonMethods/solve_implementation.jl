"""
    solve_equation(problem::KSProblem{T,N,C}, mesh::KSMesh{T,N}) where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}

Solve the equation for the given problem and mesh.

# Arguments
- `problem::KSProblem{T,N,C}`: The problem to be solved.
- `mesh::KSMesh{T,N}`: The current mesh.

# Returns
- `Vector{AbstractArray{T,N}}`: The solution.

# Examples
```julia
solution = solve_equation(problem, mesh)
```
"""
function solve_equation(problem::KSProblem{T,N,C}, mesh::KSMesh{T,N}) where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}
    solution = initialize_solution(mesh)

    for _ in 1:problem.max_iterations
        new_solution = similar(solution)

        for (i, element) in enumerate(mesh.elements)
            element_solution = apply_differential_operator(solution[i], element, problem)
            apply_boundary_conditions!(element_solution, problem, element, mesh.tensor_product_masks[i])
            new_solution[i] = element_solution
        end

        enforce_continuity!(new_solution, mesh)

        if converged(solution, new_solution, problem.tolerance)
            return new_solution
        end

        solution = new_solution
    end

    throw(ErrorException("Solution did not converge"))
end

"""
    apply_boundary_conditions!(solution::AbstractArray{T,N}, problem::KSProblem{T,N,C}, element::KSElement{T}, mask::AbstractArray{Bool,N}) where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}

Apply the boundary conditions for the given problem and element.

# Arguments
- `solution::AbstractArray{T,N}`: The current solution.
- `problem::KSProblem{T,N,C}`: The problem being solved.
- `element::KSElement{T}`: The current element.
- `mask::AbstractArray{Bool,N}`: The tensor product mask.

# Examples
```julia
apply_boundary_conditions!(solution, problem, element, mask)
```
"""
function apply_boundary_conditions!(solution::AbstractArray{T,N}, problem::KSProblem{T,N,C}, element::KSElement{T}, mask::AbstractArray{Bool,N}) where {T<:Real,N,C<:AbstractKSCoordinateSystem{N}}
    for idx in CartesianIndices(solution)
        if mask[idx] && is_boundary_node(element.points[idx], element)
            solution[idx] = problem.boundary_conditions(element.points[idx].coordinates)
        end
    end
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
is_boundary = is_boundary_node(element.points[1], element)
```
"""
function is_boundary_node(point::KSPoint{T}, element::KSElement{T}) where {T<:Real}
    return any(p -> p.coordinates == point.coordinates, element.points[[1, end]])
end

"""
    solve_multi_level(hierarchy::Vector{KSMesh{T,N}}, f::Function, num_cycles::Int) where {T<:Real,N}

Solve a problem using a multi-level method.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy.
- `f::Function`: The right-hand side function.
- `num_cycles::Int`: Number of multi-level cycles to perform.

# Returns
- `Vector{T}`: The solution on the finest mesh.

# Examples
```julia
solution = solve_multi_level(mesh_hierarchy, f, 3)
```
"""

function solve_multi_level(hierarchy::Vector{KSMesh{T,N}}, f::Function, num_cycles::Int) where {T<:Real,N}
    finest_mesh = hierarchy[end]
    solution = zeros(T, total_dofs(finest_mesh))

    for _ in 1:num_cycles
        solution = v_cycle(hierarchy, f, solution, length(hierarchy))
    end

    return solution
end
"""
    converged(old_solution::Vector{AbstractArray{T,N}}, new_solution::Vector{AbstractArray{T,N}}, tolerance::T) where {T<:Real,N}

Check if the solution has converged by comparing the maximum difference between old and new solutions.

# Arguments
- `old_solution::Vector{AbstractArray{T,N}}`: The old solution.
- `new_solution::Vector{AbstractArray{T,N}}`: The new solution.
- `tolerance::T`: The convergence tolerance.

# Returns
- `Bool`: `true` if the solution has converged, `false` otherwise.

# Examples
```julia
has_converged = converged(old_solution, new_solution, 1e-6)
```
"""

function converged(old_solution::Vector{AbstractArray{T,N}}, new_solution::Vector{AbstractArray{T,N}}, tolerance::T) where {T<:Real,N}
    if length(old_solution) != length(new_solution)
        throw(ArgumentError("Old and new solutions must have the same length"))
    end

    max_diff = maximum(maximum(abs.(old - new)) for (old, new) in zip(old_solution, new_solution))
    return max_diff < tolerance
end

"""
    initialize_solution(mesh::KSMesh{T,N}) where {T<:Real,N}

Initialize the solution for the given mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh.

# Returns
- `Vector{Array{T,N}}`: The initialized solution.

# Examples
```julia
initial_solution = initialize_solution(mesh)
```
"""

function initialize_solution(mesh::KSMesh{T,N}) where {T<:Real,N}
    return [zeros(T, size(element.points)) for element in mesh.elements]
end


"""
    solve_coarse_problem(coarse_mesh::KSMesh{T,N}, f::Function) where {T<:Real,N}

Solve the coarse problem exactly.

# Arguments
- `coarse_mesh::KSMesh{T,N}`: The coarse mesh
- `f::Function`: The right-hand side function

# Returns
- `Vector{T}`: The solution on the coarse mesh

# Example
```julia
u_coarse = solve_coarse_problem(hierarchy[1], f)
```
"""
function solve_coarse_problem(coarse_mesh::KSMesh{T,N}, f::Function) where {T<:Real,N}
    A = assemble_system_matrix(coarse_mesh)
    b = assemble_rhs_vector(coarse_mesh, f)
    return A \ b
end

"""
    smooth(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N}

Perform a smoothing step.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector

# Returns
- `Vector{T}`: The smoothed solution vector

# Example
```julia
u_smoothed = smooth(mesh, f, u)
```
"""
function smooth(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N}
    num_iterations = 3  # Number of smoothing iterations
    for _ in 1:num_iterations
        u = gauss_seidel_iteration(mesh, f, u)
    end
    return u
end

"""
    gauss_seidel_iteration(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N}

Perform a single Gauss-Seidel iteration.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector

# Returns
- `Vector{T}`: The updated solution vector after one iteration

# Example
```julia
u_new = gauss_seidel_iteration(mesh, f, u)
```
"""


function gauss_seidel_iteration(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N}
    A = assemble_system_matrix(mesh)
    b = assemble_rhs_vector(mesh, f)
    n = length(u)
    for i in 1:n
        u[i] = (b[i] - sum(A[i, j] * u[j] for j in 1:i-1) - sum(A[i, j] * u[j] for j in i+1:n)) / A[i, i]
    end
    return u
end
