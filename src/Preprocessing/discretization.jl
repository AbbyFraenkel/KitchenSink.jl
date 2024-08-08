"""
    preprocess_mesh(problem::AbstractKSProblem, domain::NTuple{N,Tuple{T,T}}, coord_system::AbstractKSCoordinateSystem{N},
    num_elements::NTuple{N,Int}, polynomial_degree::Int, max_levels::Int, tolerance::T) where {T<:Real,N}

Preprocess the mesh for the given problem.

# Arguments
- `problem::AbstractKSProblem`: The problem to be solved.
- `domain::NTuple{N,Tuple{T,T}}`: The domain of the problem.
- `coord_system::AbstractKSCoordinateSystem{N}`: The coordinate system.
- `num_elements::NTuple{N,Int}`: The number of elements in each dimension.
- `polynomial_degree::Int`: The polynomial degree for the elements.
- `max_levels::Int`: The maximum levels of refinement.
- `tolerance::T`: The tolerance for error estimation.

# Returns
- `KSMesh{T,N}`: The preprocessed mesh.

# Examples
```julia
problem, _, _ = create_test_problem(Float64, 2, :cartesian, :pde)
mesh = preprocess_mesh(problem, problem.domain, problem.coordinate_system, (5, 5), 3, 3, 1e-6)
```
"""
function preprocess_mesh(problem::AbstractKSProblem, domain::NTuple{N,Tuple{T,T}}, coord_system::AbstractKSCoordinateSystem{N},
    num_elements::NTuple{N,Int}, polynomial_degree::Int, max_levels::Int, tolerance::T) where {T<:Real,N}
    any(ne -> ne <= 0, num_elements) && throw(ArgumentError("Initial number of elements must be positive in all dimensions"))
    polynomial_degree <= 0 && throw(ArgumentError("Polynomial degree must be positive"))
    tolerance <= 0 && throw(ArgumentError("Tolerance must be positive"))

    mesh = generate_initial_mesh(domain, coord_system, num_elements, polynomial_degree)
    error_estimates = estimate_mesh_error(mesh, problem)

    for _ in 1:max_levels
        maximum(error_estimates) < tolerance && break

        new_elements = Vector{KSElement{T}}()
        for (i, element) in enumerate(mesh.elements)
            if error_estimates[i] >= tolerance
                if element.polynomial_degree < polynomial_degree
                    push!(new_elements, AdaptiveMethods.p_refine(element))
                else
                    append!(new_elements, AdaptiveMethods.hp_refine_superposition(element, error_estimates[i], CommonMethods.estimate_smoothness(element), tolerance))
                end
            else
                push!(new_elements, element)
            end
        end

        mesh = KSMesh{T}(
            new_elements,
            [falses(el.polynomial_degree + 1) for el in new_elements],
            [Dict{Int,Int}() for _ in 1:length(new_elements)],
            mesh.basis_functions,
            zero(T),
            N
        )

        update_tensor_product_masks_with_trunk!(mesh)
        error_estimates = estimate_mesh_error(mesh, problem)
    end

    return mesh
end

"""
    generate_initial_mesh(domain::NTuple{N,Tuple{T,T}}, coord_system::AbstractKSCoordinateSystem{N},
    num_elements::NTuple{N,Int}, polynomial_degree::Int) where {T<:Real,N}

Generate the initial mesh for the given problem.

# Arguments
- `domain::NTuple{N,Tuple{T,T}}`: The domain of the problem.
- `coord_system::AbstractKSCoordinateSystem{N}`: The coordinate system.
- `num_elements::NTuple{N,Int}`: The number of elements in each dimension.
- `polynomial_degree::Int`: The polynomial degree for the elements.

# Returns
- `KSMesh{T,N}`: The initial mesh.

# Examples
```julia
mesh = generate_initial_mesh((0.0, 1.0), KSCartesianCoordinates{2,Float64}(), (5, 5), 3)
```
"""
function generate_initial_mesh(domain, coord_system::AbstractKSCoordinateSystem,
                               num_elements, polynomial_degree::Int)
    nodes = create_nodes(domain, coord_system, num_elements)
    elements = create_elements(nodes, num_elements, polynomial_degree, coord_system)

    mesh = KSMesh(elements,
                  [falses(polynomial_degree + 1) for _ in 1:length(elements)],
                  [Dict{Int, Int}() for _ in 1:length(elements)],
                  Vector{KSBasisFunction}(),
                  0.0,
                  length(domain))

    update_tensor_product_masks!(mesh)
    update_location_matrices!(mesh)

    return mesh
end

"""
    refine_mesh(mesh::KSMesh{T}, error_estimates::Vector{T}, tolerance::T) where {T<:Real}

Refine the mesh based on error estimates.

# Arguments
- `mesh::KSMesh{T}`: The current mesh.
- `error_estimates::Vector{T}`: The error estimates for each element.
- `tolerance::T`: The tolerance for refinement.

# Returns
- `KSMesh{T}`: The refined mesh.

# Examples
```julia
refined_mesh = refine_mesh(mesh, error_estimates, 0.5)
```
"""
function refine_mesh(mesh::KSMesh{T}, error_estimates::Vector{T}, tolerance::T) where {T<:Real}
    length(error_estimates) != length(mesh.elements) && throw(ArgumentError("Error estimates must match the number of elements"))
    tolerance <= 0 && throw(ArgumentError("Tolerance must be positive"))

    new_elements = Vector{KSElement{T}}()
    for (i, element) in enumerate(mesh.elements)
        if error_estimates[i] > tolerance
            append!(new_elements, AdaptiveMethods.h_refine(element))
        else
            push!(new_elements, element)
        end
    end

    KSMesh{T}(
        new_elements,
        [falses(el.polynomial_degree + 1) for el in new_elements],
        [Dict{Int,Int}() for _ in 1:length(new_elements)],
        mesh.basis_functions,
        zero(T),
        mesh.dimensions
    )
end

"""
    estimate_mesh_error(mesh::KSMesh{T, N}, problem::AbstractKSProblem) where {T<:Real, N}

Estimate the error for each element in the mesh.

# Arguments
- `mesh::KSMesh{T, N}`: The current mesh.
- `problem::AbstractKSProblem`: The problem to be solved.

# Returns
- `Vector{T}`: The error estimates for each element.

# Examples
```julia
error_estimates = estimate_mesh_error(mesh, problem)
```
"""
function estimate_mesh_error(mesh::KSMesh{T, N}, problem::AbstractKSProblem) where {T<:Real, N}
    isempty(mesh.elements) && throw(ArgumentError("Mesh must contain elements"))

    error_estimates = Vector{T}(undef, length(mesh.elements))
    Threads.@threads for i in eachindex(mesh.elements)
        error_estimates[i] = CommonMethods.estimate_error(mesh.elements[i], problem)
    end
    error_estimates
end

"""
    create_OCFE_discretization(mesh::KSMesh{T, N}, problem::AbstractKSProblem, max_derivative_order::Int) where {T<:Real, N}

Create an OCFE (Orthogonal Collocation on Finite Elements) discretization for the given problem.

# Arguments
- `mesh::KSMesh{T, N}`: The current mesh.
- `problem::AbstractKSProblem`: The problem to be solved.
- `max_derivative_order::Int`: The maximum derivative order.

# Returns
- `SparseMatrixCSC{T, Int}`: The system matrix.

# Examples
```julia
A = create_OCFE_discretization(mesh, problem, 2)
```
"""
function create_OCFE_discretization(mesh::KSMesh{T, N}, problem::AbstractKSProblem, max_derivative_order::Int) where {T<:Real, N}
    N < 1 && throw(ArgumentError("Mesh must be at least one-dimensional"))

    num_nodes = sum(length(el.points) for el in mesh.elements)
    A = spzeros(T, num_nodes, num_nodes)

    global_index = 1
    for element in mesh.elements
        local_A = CommonMethods.assemble_local_system_matrix(element, problem, max_derivative_order)
        num_local_nodes = length(element.points)
        local_indices = global_index:(global_index + num_local_nodes - 1)
        A[local_indices, local_indices] .+= local_A
        global_index += num_local_nodes
    end

    return A
end

"""
    apply_boundary_condition!(A::SparseMatrixCSC{T, Int}, b::AbstractVector{T}, global_idx::Int, bc::AbstractKSBoundaryCondition, x::AbstractVector{T}) where T<:Real

Apply the boundary condition to the system matrix and right-hand side vector.

# Arguments
- `A::SparseMatrixCSC{T, Int}`: The system matrix.
- `b::AbstractVector{T}`: The right-hand side vector.
- `global_idx::Int`: The global index of the node.
- `bc::AbstractKSBoundaryCondition`: The boundary condition.
- `x::AbstractVector{T}`: The coordinates of the node.

# Examples
```julia
apply_boundary_condition!(A, b, 1, bc, [0.0])
```
"""
function apply_boundary_condition!(A::SparseMatrixCSC{T, Int}, b::AbstractVector{T}, global_idx::Int, bc::AbstractKSBoundaryCondition, x::AbstractVector{T}) where T<:Real
    if bc isa KSDirichletBC && bc.boundary(x)
        A[global_idx, :] .= zero(T)
        A[global_idx, global_idx] = one(T)
        b[global_idx] = bc.value(x)
    elseif bc isa KSNeumannBC && bc.boundary(x)
        b[global_idx] += bc.flux(x)
    elseif bc isa KSRobinBC && bc.boundary(x)
        A[global_idx, global_idx] += bc.a(x) + bc.b(x) * A[global_idx, global_idx]
        b[global_idx] += bc.c(x)
    end
end
