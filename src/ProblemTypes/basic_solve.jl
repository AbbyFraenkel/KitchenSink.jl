"""
Core solution functionality for ProblemTypes module.
Handles solution process for all problem types.
"""

"""
    solve_problem(problem::AbstractKSProblem, mesh::KSMesh{T,N}) where {T,N}

Master solution function that handles the entire solution process.
"""
function solve_problem(problem::AbstractKSProblem, mesh::KSMesh{T,N}) where {T,N}
    # Validate problem setup
    if !validate_problem_setup(problem, mesh, mesh.coordinate_system)
        throw(ArgumentError("Invalid problem/mesh configuration"))
    end

    try
        # Create system using problem's coordinate system
        A, b = create_system_matrix_and_vector(problem, mesh, mesh.coordinate_system)

        # Generate cache key using DOF information
        cache_key = (
            typeof(problem),
            get_total_problem_dof(problem, mesh),
            has_temporal_dof(problem) ? get_temporal_dof(problem) : nothing,
            get_mesh_dof(mesh),
            get_physical_mesh_dof(mesh)
        )

        # Get cached solution or solve
        solution = CacheManagement.get_or_create_cached_item(SOLVER_CACHE, cache_key) do
            solve_stabilized_system(A, b, problem, mesh)
        end

        # Transform and verify solution
        return transform_and_verify(solution, problem, mesh)
    catch e
        @error "Problem solution failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    solve_stabilized_system(A::AbstractMatrix{T}, b::AbstractVector{T},
                          problem::AbstractKSProblem, mesh::KSMesh{T,N}) where {T,N}

Solves the linear system with stability checks and conditioning improvements.
"""
function solve_stabilized_system(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    problem::AbstractKSProblem,
    mesh::KSMesh{T,N}) where {T,N}

    # Check system size consistency with DOFs
    expected_size = get_total_problem_dof(problem, mesh)
    if has_temporal_dof(problem)
        expected_size *= get_temporal_dof(problem)
    end

    if size(A, 1) != expected_size || size(A, 2) != expected_size
        throw(DimensionMismatch("Matrix size does not match problem DOFs"))
    end

    # Estimate condition number
    κ = NumericUtilities.estimate_condition_number(A)

    # Select appropriate preconditioner
    P = Preconditioners.select_preconditioner(A)

    # Solve system
    solution = LinearSolvers.solve_linear_system(A, b, P)

    # Verify solution dimensions
    if length(solution) != expected_size
        throw(DimensionMismatch("Solution size does not match problem DOFs"))
    end

    return solution
end

"""
    transform_and_verify(solution::Vector{T}, problem::AbstractKSProblem,
                        mesh::KSMesh{T,N}) where {T,N}

Transform solution to physical space if needed and verify properties.
"""
function transform_and_verify(
    solution::Vector{T},
    problem::AbstractKSProblem,
    mesh::KSMesh{T,N}) where {T,N}

    # Verify solution properties including DOF consistency
    if !verify_solution_properties(solution, problem)
        throw(ArgumentError("Solution fails basic property checks"))
    end

    # Verify boundary conditions if present
    if hasfield(typeof(problem), :boundary_conditions)
        if !BoundaryConditions.verify_boundary_conditions(solution, problem, mesh)
            @warn "Solution may not satisfy boundary conditions"
        end
    end

    # Handle temporal DOFs
    if has_temporal_dof(problem)
        solution = reshape_temporal_solution(solution, problem, mesh)
    end

    # Transform if needed
    if needs_transform(problem, mesh)
        return transform_to_physical_space(solution, problem.coordinate_system, mesh)
    end

    return solution
end

"""
    reshape_temporal_solution(solution::Vector{T}, problem::AbstractKSProblem,
                            mesh::KSMesh{T,N}) where {T,N}

Reshape solution to include temporal dimension if needed.
"""
function reshape_temporal_solution(
    solution::Vector{T},
    problem::AbstractKSProblem,
    mesh::KSMesh{T,N}) where {T,N}

    if !has_temporal_dof(problem)
        return solution
    end

    base_dofs = get_total_problem_dof(problem, mesh)
    temporal_dofs = get_temporal_dof(problem)

    if length(solution) != base_dofs * temporal_dofs
        throw(DimensionMismatch("Solution size inconsistent with temporal DOFs"))
    end

    # Reshape to include temporal dimension
    return reshape(solution, base_dofs, temporal_dofs)
end

"""
    transform_to_physical_space(solution::Union{Vector{T}, Matrix{T}},
                              coord_sys::AbstractKSCoordinateSystem,
                              mesh::KSMesh{T,N}) where {T,N}

Transform solution from computational to physical space.
"""
function transform_to_physical_space(
    solution::Union{Vector{T}, Matrix{T}},
    coord_sys::AbstractKSCoordinateSystem,
    mesh::KSMesh{T,N}) where {T,N}

    transform = Transforms.get_coordinate_transform(coord_sys)
    return Transforms.transform_solution(solution, transform, mesh)
end

"""Determine if solution needs transformation to physical space."""
function needs_transform(problem::AbstractKSProblem, mesh::KSMesh)
    # Check if mesh has transformation data
    has_transform = !isnothing(mesh.transformation_data)

    # Check if problem coordinate system requires transformation
    needs_coord_transform = !isa(problem.coordinate_system, KSCartesianCoordinates)

    return has_transform || needs_coord_transform
end

"""
    get_expected_system_size(problem::AbstractKSProblem, mesh::KSMesh)

Get expected size of the assembled system.
"""
function get_expected_system_size(problem::AbstractKSProblem, mesh::KSMesh)
    base_size = get_total_problem_dof(problem, mesh)
    if has_temporal_dof(problem)
        return base_size * get_temporal_dof(problem)
    end
    return base_size
end

"""
    validate_dof_consistency(problem::AbstractKSProblem, mesh::KSMesh)

Validate consistency between problem and mesh degrees of freedom.
"""
function validate_dof_consistency(problem::AbstractKSProblem, mesh::KSMesh)
    if isempty(mesh.cells)
        return false
    end

    # Get first non-fictitious cell
    first_valid = findfirst(c -> !c.is_fictitious, mesh.cells)
    if isnothing(first_valid)
        return false
    end

    # Check DOF consistency
    base_size = get_total_problem_dof(problem, mesh)
    actual_size = get_mesh_dof(mesh)

    if has_temporal_dof(problem)
        base_size *= get_temporal_dof(problem)
    end

    return base_size == actual_size
end
"""
    get_cell_operators(std_cell::StandardKSCell{T,N}) where {T,N}

Get differential operators for a standard cell.

Returns a dict mapping operator types to their matrices.
"""
function get_cell_operators(std_cell::StandardKSCell{T,N}) where {T,N}
    operators = Dict{Symbol,Any}()

    # Basic differential operators
    operators[:D] = std_cell.differentiation_matrix_with_boundary
    operators[:Q] = std_cell.quadrature_matrix

    # Higher order derivatives if available
    if !isempty(std_cell.higher_order_diff_matrices_with_boundary)
        operators[:D2] = std_cell.higher_order_diff_matrices_with_boundary[1]  # Second derivative
        for (i, D) in enumerate(std_cell.higher_order_diff_matrices_with_boundary[2:end])
            operators[Symbol("D$(i+2)")] = D
        end
    end

    return operators
end

