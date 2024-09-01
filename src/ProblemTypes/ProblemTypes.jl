module ProblemTypes

using LinearAlgebra, SparseArrays, StaticArrays
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..ErrorEstimation
using ..AdaptiveMethods, ..IntergridOperators, ..MultiLevelMethods
using ..Preconditioners, ..LinearSolvers, ..Preprocessing

export solve_problem, estimate_errors, refine_mesh
export assemble_problem, apply_boundary_conditions!, create_system_matrix_and_vector

export create_bvdae_system_matrix_and_vector, create_ide_system_matrix_and_vector,
       create_pide_system_matrix_and_vector, create_moving_boundary_pde_system_matrix_and_vector,
       create_coupled_system_matrix_and_vector, apply_boundary_condition!, is_boundary_point,
       get_global_index, quadrature_matrix, is_on_boundary

"""
    solve_problem(problem::AbstractKSProblem, mesh::AbstractKSMesh, solver_options)

Solves the given problem using the specified mesh and solver options.

# Arguments
- `problem::AbstractKSProblem`: The problem to be solved.
- `mesh::AbstractKSMesh`: The mesh on which the problem is defined.
- `solver_options`: The solver options to use.

# Returns
- `u`: The solution vector.
- `mesh_hierarchy`: The hierarchy of meshes used during the solution process.
"""
function solve_problem(problem::AbstractKSProblem, mesh::AbstractKSMesh, solver_options)
    A, b, mesh_hierarchy = assemble_problem(problem, solver_options)
    u = LinearSolvers.solve_linear_system(A, b, solver_options.linear_solver)
    return u, mesh_hierarchy
end

"""
    estimate_errors(problem::AbstractKSProblem, solution, mesh::AbstractKSMesh)

Estimates the errors in the solution for each element of the mesh.

# Arguments
- `problem::AbstractKSProblem`: The problem definition.
- `solution`: The solution vector.
- `mesh::AbstractKSMesh`: The mesh used for the problem.

# Returns
- `error_estimates`: A vector of error estimates for each element.
"""
function estimate_errors(problem::AbstractKSProblem, solution, mesh::AbstractKSMesh)
    error_estimates = [ErrorEstimation.estimate_error(problem, solution, element) for element in mesh.elements]
    return error_estimates
end

"""
    refine_mesh(problem::AbstractKSProblem, mesh::AbstractKSMesh, error_estimates)

Refines the mesh based on the provided error estimates.

# Arguments
- `problem::AbstractKSProblem`: The problem definition.
- `mesh::AbstractKSMesh`: The current mesh.
- `error_estimates`: A vector of error estimates for each element.

# Returns
- `refined_mesh`: The refined mesh.
"""
function refine_mesh(problem::AbstractKSProblem, mesh::AbstractKSMesh, error_estimates)
    refined_mesh = Preprocessing.refine_mesh(mesh, error_estimates, problem.tolerance)
    return refined_mesh
end

"""
    assemble_problem(problem::AbstractKSProblem, options::KSSolverOptions)

Assembles the system matrix and vector for the given problem and solver options.

# Arguments
- `problem::AbstractKSProblem`: The problem definition.
- `options::KSSolverOptions`: The solver options.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
- `mesh_hierarchy`: The hierarchy of meshes used during assembly.
"""
function assemble_problem(problem::AbstractKSProblem, options::KSSolverOptions)
    base_mesh = Preprocessing.generate_initial_mesh(problem.domain, problem.coordinate_system, options.initial_elements, options.initial_degree)
    mesh_hierarchy = MultiLevelMethods.create_mesh_hierarchy(base_mesh, options.max_levels)

    for level in eachindex(mesh_hierarchy)
        current_mesh = mesh_hierarchy[level]
        Preprocessing.update_tensor_product_masks_with_trunk!(current_mesh)
        Preprocessing.update_location_matrices!(current_mesh)

        A, b = create_system_matrix_and_vector(problem, current_mesh)
        apply_boundary_conditions!(A, b, current_mesh, problem.boundary_conditions)

        u = LinearSolvers.solve_linear_system(A, b, options.linear_solver)

        error_indicators = [ErrorEstimation.compute_error_indicator(element, u[Preprocessing.get_active_indices(element)], problem) for element in current_mesh.elements]
        max_error, _ = findmax(first.(error_indicators))

        if max_error < options.tolerance
            break
        end

        if level < length(mesh_hierarchy)
            marked_elements = findall(e -> e[1] > options.tolerance, error_indicators)
            mesh_hierarchy = MultiLevelMethods.refine_mesh_hierarchy(mesh_hierarchy, level, marked_elements)
        end
    end

    finest_mesh = mesh_hierarchy[end]
    A, b = create_system_matrix_and_vector(problem, finest_mesh)
    apply_boundary_conditions!(A, b, finest_mesh, problem.boundary_conditions)

    return A, b, mesh_hierarchy
end

"""
    apply_boundary_conditions!(A::AbstractMatrix, b::AbstractVector, mesh::AbstractKSMesh, boundary_conditions::Function)

Applies the boundary conditions to the system matrix and vector.

# Arguments
- `A::AbstractMatrix`: The system matrix.
- `b::AbstractVector`: The right-hand side vector.
- `mesh::AbstractKSMesh`: The mesh structure.
- `boundary_conditions::Function`: The boundary conditions function.
"""
function apply_boundary_conditions!(A::AbstractMatrix, b::AbstractVector, mesh::AbstractKSMesh, boundary_conditions::Function)
    for element in mesh.elements
        for (i, point) in enumerate(element.collocation_points)
            if is_boundary_point(point, mesh)
                global_index = get_global_index(mesh, element, i)
                apply_boundary_condition!(A, b, global_index, boundary_conditions, point)
            end
        end
    end
end

"""
    apply_boundary_condition!(A::AbstractMatrix, b::AbstractVector, global_index::Int, boundary_conditions::Function, x::AbstractVector)

Applies a boundary condition at a specific point in the system matrix and vector.

# Arguments
- `A::AbstractMatrix`: The system matrix.
- `b::AbstractVector`: The right-hand side vector.
- `global_index::Int`: The global index of the point.
- `boundary_conditions::Function`: The boundary conditions function.
- `x::AbstractVector`: The coordinates of the point.
"""
function apply_boundary_condition!(A::AbstractMatrix, b::AbstractVector, global_index::Int, boundary_conditions::Function, x::SVector)
    A[global_index, :] .= 0
    A[global_index, global_index] = 1
    b[global_index] = boundary_conditions(x)
end

"""
    create_system_matrix_and_vector(problem::AbstractKSProblem, mesh::KSMesh)

Creates the system matrix and vector for different types of problems.

# Arguments
- `problem::AbstractKSProblem`: The problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_system_matrix_and_vector(problem::AbstractKSProblem, mesh::KSMesh)
    if problem isa KSPDEProblem
        return create_pde_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSODEProblem
        return create_ode_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSDAEProblem
        return create_dae_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSBVDAEProblem
        return create_bvdae_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSIDEProblem
        return create_ide_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSPIDEProblem
        return create_pide_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSMovingBoundaryPDEProblem
        return create_moving_boundary_pde_system_matrix_and_vector(problem, mesh)
    elseif problem isa KSCoupledProblem
        return create_coupled_system_matrix_and_vector(problem, mesh)
    else
        error("Unsupported problem type")
    end
end

"""
    create_pde_system_matrix_and_vector(problem::KSPDEProblem, mesh::KSMesh)

Creates the system matrix and vector for a PDE problem.

# Arguments
- `problem::KSPDEProblem`: The PDE problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_pde_system_matrix_and_vector(problem::KSPDEProblem, mesh::KSMesh)
    A, b = create_ocfe_discretization(mesh, problem)
    return A, b
end

"""
    create_ode_system_matrix_and_vector(problem::KSODEProblem, mesh::KSMesh)

Creates the system matrix and vector for an ODE problem.

# Arguments
- `problem::KSODEProblem`: The ODE problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_ode_system_matrix_and_vector(problem::KSODEProblem, mesh::KSMesh)
    t_nodes = [point[1] for element in mesh.elements for point in element.collocation_points]
    D = SpectralMethods.derivative_matrix(t_nodes)
    A = I - (problem.tspan[2] - problem.tspan[1]) * D
    b = [problem.initial_conditions; zeros(length(t_nodes) - 1)]
    return A, b
end

"""
    create_dae_system_matrix_and_vector(problem::KSDAEProblem, mesh::KSMesh)

Creates the system matrix and vector for a DAE problem.

# Arguments
- `problem::KSDAEProblem`: The DAE problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_dae_system_matrix_and_vector(problem::KSDAEProblem, mesh::KSMesh)
    t_nodes = [point[1] for element in mesh.elements for point in element.collocation_points]
    D = SpectralMethods.derivative_matrix(t_nodes)
    A_diff = I - (problem.tspan[2] - problem.tspan[1]) * D
    A_alg = zeros(length(t_nodes), length(t_nodes))
    A = [A_diff zeros(size(A_diff)); zeros(size(A_alg)) A_alg]
    b = [problem.initial_conditions; zeros(length(t_nodes) - 1)]
    return A, b
end

"""
    create_bvdae_system_matrix_and_vector(problem::KSBVDAEProblem, mesh::KSMesh)

Creates the system matrix and vector for a Boundary Value DAE problem.

# Arguments
- `problem::KSBVDAEProblem`: The Boundary Value DAE problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_bvdae_system_matrix_and_vector(problem::KSBVDAEProblem, mesh::KSMesh)
    x_nodes = [point[1] for element in mesh.elements for point in element.collocation_points]
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

# Arguments
- `problem::KSIDEProblem`: The Integral-Differential Equation problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_ide_system_matrix_and_vector(problem::KSIDEProblem, mesh::KSMesh)
    x_nodes = [point[1] for element in mesh.elements for point in element.collocation_points]
    D = SpectralMethods.derivative_matrix(x_nodes)
    Q = quadrature_matrix(mesh)
    A = D + Q * problem.K.(x_nodes, x_nodes')
    b = [problem.initial_conditions; zeros(length(x_nodes) - 1)]
    return A, b
end

"""
    create_pide_system_matrix_and_vector(problem::KSPIDEProblem, mesh::KSMesh)

Creates the system matrix and vector for a Partial Integral-Differential Equation problem.

# Arguments
- `problem::KSPIDEProblem`: The Partial Integral-Differential Equation problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_pide_system_matrix_and_vector(problem::KSPIDEProblem, mesh::KSMesh)
    A_pde, _ = create_ocfe_discretization(mesh, problem)
    Q = quadrature_matrix(mesh)
    A_integral = Q * problem.K.(mesh.elements, mesh.elements')
    A = A_pde + A_integral
    b = zeros(size(A, 1))
    return A, b
end

"""
    create_moving_boundary_pde_system_matrix_and_vector(problem::KSMovingBoundaryPDEProblem, mesh::KSMesh)

Creates the system matrix and vector for a Moving Boundary PDE problem.

# Arguments
- `problem::KSMovingBoundaryPDEProblem`: The Moving Boundary PDE problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_moving_boundary_pde_system_matrix_and_vector(problem::KSMovingBoundaryPDEProblem, mesh::KSMesh)
    A, b = create_ocfe_discretization(mesh, problem)
    return A, b
end

"""
    create_coupled_system_matrix_and_vector(problem::KSCoupledProblem, mesh::KSMesh)

Creates the system matrix and vector for a coupled problem.

# Arguments
- `problem::KSCoupledProblem`: The coupled problem definition.
- `mesh::KSMesh`: The mesh structure.

# Returns
- `A`: The system matrix.
- `b`: The right-hand side vector.
"""
function create_coupled_system_matrix_and_vector(problem::KSCoupledProblem, mesh::KSMesh)
    sub_systems = [create_system_matrix_and_vector(sub_prob, mesh) for sub_prob in problem.subproblems]
    A = BlockDiagonal([sys[1] for sys in sub_systems])
    b = vcat([sys[2] for sys in sub_systems]...)

    for (i, j) in keys(problem.coupling_terms)
        A[i, j] = problem.coupling_terms[i, j](mesh)
    end

    return A, b
end

"""
    block_indices(block::Int, block_size::Int)

Compute the indices for a block in a block matrix.

# Arguments
- `block::Int`: The block number.
- `block_size::Int`: The size of the block.

# Returns
- `UnitRange{Int}`: The range of indices for the block.
"""
function block_indices(block::Int, block_size::Int)
    start_idx = (block - 1) * block_size + 1
    end_idx = block * block_size
    return start_idx:end_idx
end

"""
    apply_boundary_conditions!(A::AbstractArray, b::AbstractArray, mesh::KSMesh, boundary_conditions)

Applies the boundary conditions to the system matrix and vector.

# Arguments
- `A::AbstractArray`: The system matrix.
- `b::AbstractArray`: The right-hand side vector.
- `mesh::KSMesh`: The mesh structure.
- `boundary_conditions`: Function - Boundary conditions function.
"""
function apply_boundary_conditions!(A::AbstractArray, b::AbstractArray, mesh::KSMesh, boundary_conditions)
    for element in mesh.elements
        for (i, point) in enumerate(element.collocation_points)
            if is_boundary_point(point, mesh)
                global_index = get_global_index(mesh, element, i)
                apply_boundary_condition!(A, b, global_index, boundary_conditions, point)
            end
        end
    end
end

"""
    is_boundary_point(point::SVector{N,T}, mesh::KSMesh{T,N}) where {T<:Real,N}

Determines if a point is a boundary point in the mesh.

# Arguments
- `point::SVector{N,T}`: The point to check.
- `mesh::KSMesh{T,N}`: The mesh structure.

# Returns
- `Bool`: True if the point is a boundary point, otherwise false.
"""
function is_boundary_point(point::SVector{N,T}, mesh::KSMesh{T,N}) where {T<:Real,N}
    for element in mesh.elements
        if any(x -> all(isapprox.(x, point)), element.collocation_points)
            return true
        end
    end
    return false
end

"""
    get_global_index(mesh::KSMesh, element::KSElement, local_index::Int)

Gets the global index of a point in the mesh.

# Arguments
- `mesh::KSMesh`: The mesh structure.
- `element::KSElement`: The element containing the point.
- `local_index::Int`: The local index of the point in the element.

# Returns
- `Int`: The global index of the point in the mesh.
"""
function get_global_index(mesh::KSMesh, element::KSElement, local_index::Int)
    return mesh.location_matrices[element.id][local_index]
end

"""
    quadrature_matrix(mesh::KSMesh)

Creates the quadrature matrix for the mesh.

# Arguments
- `mesh::KSMesh`: The mesh structure.

# Returns
- `SparseMatrixCSC`: The quadrature matrix.
"""
function quadrature_matrix(mesh::KSMesh)
    n_total = total_dofs(mesh)
    Q = spzeros(n_total, n_total)

    for element in mesh.elements
        std_elem = mesh.standard_elements[(element.level, element.polynomial_degree)]
        weights = std_elem.collocation_weights

        for (i, weight) in enumerate(weights)
            global_i = get_global_index(mesh, element, i)
            Q[global_i, global_i] = weight
        end
    end

    return Q
end

"""
    is_on_boundary(local_index::Int, mask_index::CartesianIndex, mask_size::Tuple)

Determines if a point is on the boundary of an element.

# Arguments
- `local_index::Int`: The local index of the point in the element.
- `mask_index::CartesianIndex`: The index of the tensor product mask.
- `mask_size::Tuple`: The size of the tensor product mask.

# Returns
- `Bool`: True if the point is on the boundary, otherwise false.
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
