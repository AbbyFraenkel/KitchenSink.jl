module ProblemTypes

using ..KSTypes, ..SpectralMethods, ..Preprocessing, ..CoordinateSystems, ..AdaptiveMethods, ..MultiLevelMethods
using LinearAlgebra, SparseArrays

export assemble_problem, apply_boundary_conditions!, create_system_matrix_and_vector,
    create_pde_system_matrix_and_vector, create_ode_system_matrix_and_vector,
    create_dae_system_matrix_and_vector, create_bvdae_system_matrix_and_vector,
    create_ide_system_matrix_and_vector, create_pide_system_matrix_and_vector,
    create_moving_boundary_pde_system_matrix_and_vector, create_coupled_system_matrix_and_vector,
    apply_boundary_condition!, is_boundary_point, get_global_index, quadrature_matrix,
    find_element_containing_point, is_on_boundary

"""
    assemble_problem(problem::AbstractKSProblem, options::KSSolverOptions)

Assembles the problem by generating the initial mesh, creating a mesh hierarchy, and solving the system using adaptive refinement and multi-level methods.

Arguments:
- problem: AbstractKSProblem - The problem definition.
- options: KSSolverOptions - Solver options including tolerance, max levels, etc.

Returns:
- A: Matrix - Assembled system matrix.
- b: Vector - Right-hand side vector.
- mesh_hierarchy: Vector{KSMesh} - Hierarchical mesh structure.
"""
function assemble_problem(problem::AbstractKSProblem, options::KSSolverOptions)
    base_mesh = generate_initial_mesh(problem.domain, problem.coordinate_system, options.initial_elements, options.initial_degree)
    mesh_hierarchy = create_mesh_hierarchy(base_mesh, options.max_levels)

    for level in eachindex(mesh_hierarchy)
        current_mesh = mesh_hierarchy[level]
        update_tensor_product_masks_with_trunk!(current_mesh)
        update_location_matrices!(current_mesh)

        A, b = create_system_matrix_and_vector(problem, current_mesh)
        apply_boundary_conditions!(A, b, current_mesh, problem.boundary_conditions)

        u = A \ b  # Initial solution, could be replaced with a more sophisticated solver

        error_indicators = [compute_error_indicator(element, u[get_active_indices(element)], problem) for element in current_mesh.elements]
        max_error, _ = findmax(first.(error_indicators))

        if max_error < options.tolerance
            break
        end

        if level < length(mesh_hierarchy)
            marked_elements = findall(e -> e[1] > options.tolerance, error_indicators)
            mesh_hierarchy = refine_mesh_hierarchy(mesh_hierarchy, level, marked_elements)

            current_mesh = adapt_mesh_superposition(current_mesh, u, problem, options.tolerance)

            for (i, element) in enumerate(current_mesh.elements)
                error, smoothness = error_indicators[i]
                if error > options.tolerance
                    if smoothness > options.smoothness_threshold
                        current_mesh.elements[i] = p_refine(element)
                    else
                        current_mesh.elements[i] = hp_refine_superposition(element, error, smoothness, options.tolerance)[1]
                    end
                end
            end
        end
    end

    finest_mesh = mesh_hierarchy[end]
    A, b = create_system_matrix_and_vector(problem, finest_mesh)
    apply_boundary_conditions!(A, b, finest_mesh, problem.boundary_conditions)

    return A, b, mesh_hierarchy
end

"""
    apply_boundary_condition!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, global_idx::Int, boundary_conditions::AbstractKSBoundaryCondition, x::Float64)

Applies boundary conditions to the OCFE discretization matrix.

# Arguments
- `A::SparseMatrixCSC{Float64, Int}`: The OCFE discretization matrix.
- `b::Vector{Float64}`: The right-hand side vector.
- `global_idx::Int`: The global index of the point in the mesh.
- `boundary_conditions::AbstractKSBoundaryCondition`: The boundary conditions.
- `x::Float64`: The spatial coordinate.
"""
function apply_boundary_condition!(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, global_idx::Int, boundary_conditions::AbstractKSBoundaryCondition, x::Float64)
    if isa(boundary_conditions, KSDirichletBC)
        A[global_idx, :] .= 0.0
        A[global_idx, global_idx] = 1.0
        b[global_idx] = boundary_conditions.value(x)
    elseif isa(boundary_conditions, KSNeumannBC)
        b[global_idx] += boundary_conditions.flux(x)
    elseif isa(boundary_conditions, KSRobinBC)
        A[global_idx, global_idx] += boundary_conditions.a(x) + boundary_conditions.b(x)
        b[global_idx] += boundary_conditions.c(x)
    else
        throw(ArgumentError("Unsupported boundary condition type"))
    end
end

"""
    create_system_matrix_and_vector(problem::AbstractKSProblem, mesh::KSMesh)

Creates the system matrix and vector for different types of problems.

Arguments:
- problem: AbstractKSProblem - The problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_system_matrix_and_vector(problem::AbstractKSProblem, mesh::KSMesh)
    if isa(problem, KSPDEProblem)
        return create_pde_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSODEProblem)
        return create_ode_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSDAEProblem)
        return create_dae_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSBVDAEProblem)
        return create_bvdae_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSIDEProblem)
        return create_ide_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSPIDEProblem)
        return create_pide_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSMovingBoundaryPDEProblem)
        return create_moving_boundary_pde_system_matrix_and_vector(problem, mesh)
    elseif isa(problem, KSCoupledProblem)
        return create_coupled_system_matrix_and_vector(problem, mesh)
    else
        error("Unsupported problem type")
    end
end

"""
    create_pde_system_matrix_and_vector(problem::KSPDEProblem, mesh::KSMesh)

Creates the system matrix and vector for a PDE problem.

Arguments:
- problem: KSPDEProblem - The PDE problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_pde_system_matrix_and_vector(problem::KSPDEProblem, mesh::KSMesh)
    A = create_OCFE_discretization(mesh, problem.pde, problem.boundary_conditions)
    b = zeros(size(A, 1))
    return A, b
end

"""
    create_ode_system_matrix_and_vector(problem::KSODEProblem, mesh::KSMesh)

Creates the system matrix and vector for an ODE problem.

Arguments:
- problem: KSODEProblem - The ODE problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_ode_system_matrix_and_vector(problem::KSODEProblem, mesh::KSMesh)
    t_nodes = [node.coordinates[1] for node in mesh.elements]
    D = SpectralMethods.derivative_matrix(t_nodes)
    A = I - problem.tspan[2] * D
    b = [problem.initial_conditions; zeros(length(t_nodes) - 1)]
    return A, b
end

"""
    create_dae_system_matrix_and_vector(problem::KSDAEProblem, mesh::KSMesh)

Creates the system matrix and vector for a DAE problem.

Arguments:
- problem: KSDAEProblem - The DAE problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_dae_system_matrix_and_vector(problem::KSDAEProblem, mesh::KSMesh)
    t_nodes = [node.coordinates[1] for node in mesh.elements]
    D = SpectralMethods.derivative_matrix(t_nodes)
    A_diff = I - problem.tspan[2] * D
    A_alg = zeros(length(t_nodes), length(t_nodes))
    A = [A_diff zeros(size(A_diff)); zeros(size(A_alg)) A_alg]
    b = [problem.initial_conditions; zeros(length(t_nodes) - 1)]
    return A, b
end

"""
    create_bvdae_system_matrix_and_vector(problem::KSBVDAEProblem, mesh::KSMesh)

Creates the system matrix and vector for a Boundary Value DAE problem.

Arguments:
- problem: KSBVDAEProblem - The Boundary Value DAE problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_bvdae_system_matrix_and_vector(problem::KSBVDAEProblem, mesh::KSMesh)
    x_nodes = [node.coordinates[1] for node in mesh.elements]
    D = SpectralMethods.derivative_matrix(x_nodes)
    A_diff = D
    A_alg = zeros(length(x_nodes), length(x_nodes))
    A = [A_diff zeros(size(A_diff)); zeros(size(A_alg)) A_alg]
    b = zeros(2 * length(x_nodes))
    return A, b
end

"""
    create_ide_system_matrix_and_vector(problem::KSIDEProblem, mesh::KSMesh)

Creates the system matrix and vector for an Integral-Differential Equation problem.

Arguments:
- problem: KSIDEProblem - The Integral-Differential Equation problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_ide_system_matrix_and_vector(problem::KSIDEProblem, mesh::KSMesh)
    x_nodes = [node.coordinates[1] for node in mesh.elements]
    D = SpectralMethods.derivative_matrix(x_nodes)
    Q = quadrature_matrix(mesh)
    A = D + Q * problem.K.(x_nodes, x_nodes')
    b = [problem.initial_conditions; zeros(length(x_nodes) - 1)]
    return A, b
end

"""
    create_pide_system_matrix_and_vector(problem::KSPIDEProblem, mesh::KSMesh)

Creates the system matrix and vector for a Partial Integral-Differential Equation problem.

Arguments:
- problem: KSPIDEProblem - The Partial Integral-Differential Equation problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_pide_system_matrix_and_vector(problem::KSPIDEProblem, mesh::KSMesh)
    A_pde = create_OCFE_discretization(mesh, problem.pide, problem.boundary_conditions)
    Q = quadrature_matrix(mesh)
    A_integral = Q * problem.K.(mesh.elements, mesh.elements')
    A = A_pde + A_integral
    b = zeros(size(A, 1))
    return A, b
end

"""
    create_moving_boundary_pde_system_matrix_and_vector(problem::KSMovingBoundaryPDEProblem, mesh::KSMesh)

Creates the system matrix and vector for a Moving Boundary PDE problem.

Arguments:
- problem: KSMovingBoundaryPDEProblem - The Moving Boundary PDE problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_moving_boundary_pde_system_matrix_and_vector(problem::KSMovingBoundaryPDEProblem, mesh::KSMesh)
    A = create_OCFE_discretization(mesh, problem.pde, problem.boundary_conditions)
    b = zeros(size(A, 1))
    return A, b
end

"""
    create_coupled_system_matrix_and_vector(problem::KSCoupledProblem, mesh::KSMesh)

Creates the system matrix and vector for a coupled problem.

Arguments:
- problem: KSCoupledProblem - The coupled problem definition.
- mesh: KSMesh - The mesh structure.

Returns:
- A: Matrix - System matrix.
- b: Vector - Right-hand side vector.
"""
function create_coupled_system_matrix_and_vector(problem::KSCoupledProblem, mesh::KSMesh)
    sub_systems = [create_system_matrix_and_vector(sub_prob, mesh) for sub_prob in problem.problems]
    A = BlockDiagonal([sys[1] for sys in sub_systems])
    b = vcat([sys[2] for sys in sub_systems]...)

    for (i, j) in problem.coupling_terms
        A[i, j] = problem.coupling_terms[i, j](mesh)
    end

    return A, b
end

"""
    apply_boundary_conditions!(A::AbstractMatrix, b::AbstractVector, mesh::KSMesh, boundary_conditions)

Applies the boundary conditions to the system matrix and vector.

Arguments:
- A: AbstractMatrix - System matrix.
- b: AbstractVector - Right-hand side vector.
- mesh: KSMesh - The mesh structure.
- boundary_conditions: Function - Boundary conditions function.
"""
function apply_boundary_conditions!(A::AbstractMatrix, b::AbstractVector, mesh::KSMesh, boundary_conditions)
    for element in mesh.elements
        for (i, point) in enumerate(element.points)
            if is_boundary_point(point, mesh)
                global_index = get_global_index(mesh, element, i)
                for bc in boundary_conditions
                    apply_boundary_condition!(A, b, global_index, bc, point.coordinates)
                end
            end
        end
    end
end

"""
    is_boundary_point(point::KSPoint, mesh::KSMesh)

Determines if a point is a boundary point in the mesh.

Arguments:
- point: KSPoint - The point to check.
- mesh: KSMesh - The mesh structure.

Returns:
- Bool - True if the point is a boundary point, otherwise false.
"""
function is_boundary_point(point::KSPoint, mesh::KSMesh)
    element = find_element_containing_point(mesh, point)
    if isnothing(element)
        return false
    end

    local_index = findfirst(p -> p == point, element.points)
    if isnothing(local_index)
        return false
    end

    mask = mesh.tensor_product_masks[element.id]
    return any(i -> mask[i] && is_on_boundary(local_index, i, size(mask)), CartesianIndices(mask))
end

"""
    get_global_index(mesh::KSMesh, element::KSElement, local_index::Int)

Gets the global index of a point in the mesh.

Arguments:
- mesh: KSMesh - The mesh structure.
- element: KSElement - The element containing the point.
- local_index: Int - The local index of the point in the element.

Returns:
- Int - The global index of the point in the mesh.
"""
function get_global_index(mesh::KSMesh, element::KSElement, local_index::Int)
    return mesh.location_matrices[element.id][local_index]
end

"""
    quadrature_matrix(mesh::KSMesh)

Creates the quadrature matrix for the mesh.

Arguments:
- mesh: KSMesh - The mesh structure.

Returns:
- SparseMatrixCSC - The quadrature matrix.
"""
function quadrature_matrix(mesh::KSMesh)
    n_total = sum(length(element.points) for element in mesh.elements)
    Q = spzeros(n_total, n_total)

    for element in mesh.elements
        for (i, point) in enumerate(element.points)
            if !isnothing(point.weight)
                global_i = get_global_index(mesh, element, i)
                Q[global_i, global_i] = point.weight
            end
        end
    end

    return Q
end

"""
    find_element_containing_point(mesh::KSMesh, point::KSPoint)

Finds the element containing a given point.

Arguments:
- mesh: KSMesh - The mesh structure.
- point: KSPoint - The point to find.

Returns:
- Union{Nothing, KSElement} - The element containing the point, or nothing if not found.
"""
function find_element_containing_point(mesh::KSMesh, point::KSPoint)
    for element in mesh.elements
        if point in element.points
            return element
        end
    end
    return nothing
end

"""
    is_on_boundary(local_index::Int, mask_index::CartesianIndex, mask_size::Tuple)

Determines if a point is on the boundary of an element.

Arguments:
- local_index: Int - The local index of the point in the element.
- mask_index: CartesianIndex - The index of the tensor product mask.
- mask_size: Tuple - The size of the tensor product mask.

Returns:
- Bool - True if the point is on the boundary, otherwise false.
"""
function is_on_boundary(local_index::Int, mask_index::CartesianIndex, mask_size::Tuple)
    for dim in eachindex(mask_size)
        if mask_index[dim] == 1 || mask_index[dim] == mask_size[dim]
            return true
        end
    end
    return false
end

end # module ProblemTypes



