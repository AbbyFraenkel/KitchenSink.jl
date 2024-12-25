module MultiLevelMethods

using LinearAlgebra, SparseArrays
import AlgebraicMultigrid: ruge_stuben, solve
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..ErrorEstimation
using ..AdaptiveMethods, ..IntergridOperators

export create_mesh_hierarchy, coarsen_mesh, merge_elements, refine_mesh_uniformly
export refine_mesh_hierarchy, refine_marked_elements, adjust_finer_level
export v_cycle, w_cycle, full_multigrid, geometric_multigrid, algebraic_multigrid

"""
	create_mesh_hierarchy(base_mesh::KSMesh{T,N}, max_levels::Int) where {T<:Real,N} -> Vector{KSMesh{T,N}}

Create a hierarchy of meshes starting from a base mesh.

# Arguments
- `base_mesh::KSMesh{T,N}`: The initial mesh
- `max_levels::Int`: The maximum number of levels in the hierarchy

# Returns
- `Vector{KSMesh{T,N}}`: A vector of meshes representing the hierarchy
"""
function create_mesh_hierarchy(
        base_mesh::KSMesh{T, N}, max_levels::Int) where {T <: Real, N}
    max_levels > 0 || throw(ArgumentError("max_levels must be positive"))
    hierarchy = [base_mesh]
    for _ in 2:max_levels
        refined_mesh = refine_mesh_uniformly(hierarchy[end])
        push!(hierarchy, refined_mesh)
    end
    return hierarchy
end

"""
	coarsen_mesh(mesh::KSMesh{T,N}) where {T<:Real,N} -> KSMesh{T,N}

Create a coarser version of the given mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The mesh to coarsen

# Returns
- `KSMesh{T,N}`: The coarsened mesh
"""
function coarsen_mesh(mesh::KSMesh{T, N}) where {T <: Real, N}
    length(mesh.cells) > 1 ||
        throw(ArgumentError("Mesh must have more than one element to coarsen"))
    coarse_elements = [merge_elements(mesh.cells[i], mesh.cells[i + 1])
                       for i in 1:2:(length(mesh.cells) - 1)]
    return KSMesh(
        elements = coarse_elements,
        tensor_product_masks = mesh.tensor_product_masks,
        location_matrices = mesh.location_matrices,
        standard_elements = Dict{Tuple{Int, Int}, StandardKSCell{T, N}}(),  # Initialize empty standard element cache
        global_error_estimate = mesh.global_error_estimate
    )
end

"""
	merge_elements(el1::KSCell{T,N}, el2::KSCell{T,N}) where {T<:Real,N} -> KSCell{T,N}

Merge two elements into a single, coarser element.

# Arguments
- `el1::KSCell{T,N}`: The first element to merge
- `el2::KSCell{T,N}`: The second element to merge

# Returns
- `KSCell{T,N}`: The merged element
"""
function merge_elements(el1::KSCell{T, N}, el2::KSCell{T, N}) where {T <: Real, N}
    new_points = unique(vcat(el1.points, el2.points))
    new_basis_functions = unique(vcat(el1.basis_functions, el2.basis_functions))
    new_level = min(el1.level, el2.level) - 1
    new_collocation_points = unique(vcat(el1.collocation_points, el2.collocation_points))

    # Retrieve or create the StandardKSCell for the merged element
    new_degree = max(el1.polynomial_degree, el2.polynomial_degree)
    std_elem = SpectralMethods.get_or_create_standard_cell(new_degree)
    transformed_std_elem = SpectralMethods.transform_standard_element(
        std_elem,
        (el1.domain..., el2.domain...),
        new_level
    )

    return KSCell(
        id = el1.id,
        level = new_level,
        polynomial_degree = new_degree,
        points = new_points,
        basis_functions = new_basis_functions,
        neighbors = nothing,
        parent = nothing,
        children = [el1, el2],
        error_estimate = max(el1.error_estimate, el2.error_estimate),
        collocation_points = transformed_std_elem.collocation_points,
        differentiation_matrices = transformed_std_elem.differentiation_matrices
    )
end

"""
	refine_mesh_uniformly(mesh::KSMesh{T,N}) where {T<:Real,N} -> KSMesh{T,N}

Refine all elements in the given mesh uniformly.

# Arguments
- `mesh::KSMesh{T,N}`: The mesh to refine

# Returns
- `KSMesh{T,N}`: The refined mesh
"""
function refine_mesh_uniformly(mesh::KSMesh{T, N}) where {T <: Real, N}
    new_elements = vcat([AdaptiveMethods.h_refine(el) for el in mesh.cells]...)
    new_standard_elements = Dict{Tuple{Int, Int}, StandardKSCell{T, N}}()

    # Ensure StandardKSCell cache is updated for the refined elements
    for el in new_elements
        key = (el.level, el.polynomial_degree)
        if !haskey(new_standard_elements, key)
            std_elem = SpectralMethods.get_or_create_standard_cell(el.polynomial_degree)
            new_standard_elements[key] = std_elem
        end
    end

    return KSMesh(
        elements = new_elements,
        tensor_product_masks = mesh.tensor_product_masks,
        location_matrices = mesh.location_matrices,
        standard_elements = new_standard_elements,
        global_error_estimate = mesh.global_error_estimate
    )
end

"""
	refine_mesh_hierarchy(hierarchy::Vector{KSMesh{T,N}}, level::Int, marked_elements::AbstractVector{Int}) where {T<:Real,N} -> Vector{KSMesh{T,N}}

Refine specific elements in a mesh hierarchy at a given level.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The current mesh hierarchy
- `level::Int`: The level at which to perform refinement
- `marked_elements::AbstractVector{Int}`: Indices of elements to be refined

# Returns
- `Vector{KSMesh{T,N}}`: The updated mesh hierarchy
"""
function refine_mesh_hierarchy(
        hierarchy::Vector{KSMesh{T, N}},
        level::Int,
        marked_elements::AbstractVector{Int}
) where {T <: Real, N}
    1 <= level <= length(hierarchy) || throw(ArgumentError("Invalid refinement level"))
    isempty(marked_elements) && return hierarchy

    refined_mesh = refine_marked_elements(hierarchy[level], marked_elements)
    new_hierarchy = copy(hierarchy)
    new_hierarchy[level] = refined_mesh

    for l in (level + 1):length(new_hierarchy)
        new_hierarchy[l] = adjust_finer_level(new_hierarchy[l - 1], new_hierarchy[l])
    end

    return new_hierarchy
end

"""
	refine_marked_elements(mesh::KSMesh{T,N}, marked_elements::AbstractVector{Int}) where {T<:Real,N} -> KSMesh{T,N}

Refine specific elements in a mesh.

# Arguments
- `mesh::KSMesh{T,N}`: The mesh to refine
- `marked_elements::AbstractVector{Int}`: Indices of elements to be refined

# Returns
- `KSMesh{T,N}`: The refined mesh
"""
function refine_marked_elements(
        mesh::KSMesh{T, N},
        marked_elements::AbstractVector{Int}
) where {T <: Real, N}
    new_elements = Vector{KSCell{T, N}}()
    new_standard_elements = Dict{Tuple{Int, Int}, StandardKSCell{T, N}}()

    for (i, el) in enumerate(mesh.cells)
        if i in marked_elements
            refined_elements = AdaptiveMethods.h_refine(el)
            append!(new_elements, refined_elements)

            # Update StandardKSCell cache for refined elements
            for sub_el in refined_elements
                key = (sub_el.level, sub_el.polynomial_degree)
                if !haskey(new_standard_elements, key)
                    std_elem = SpectralMethods.get_or_create_standard_cell(
                        sub_el.polynomial_degree,
                    )
                    new_standard_elements[key] = std_elem
                end
            end
        else
            push!(new_elements, el)
        end
    end

    return KSMesh(
        elements = new_elements,
        tensor_product_masks = mesh.tensor_product_masks,
        location_matrices = mesh.location_matrices,
        standard_elements = new_standard_elements,
        global_error_estimate = mesh.global_error_estimate
    )
end

"""
	adjust_finer_level(coarse_mesh::KSMesh{T,N}, fine_mesh::KSMesh{T,N}) where {T<:Real,N} -> KSMesh{T,N}

Adjust a finer level mesh based on changes in the coarser level.

# Arguments
- `coarse_mesh::KSMesh{T,N}`: The refined coarser level mesh
- `fine_mesh::KSMesh{T,N}`: The current finer level mesh

# Returns
- `KSMesh{T,N}`: The adjusted finer level mesh
"""
function adjust_finer_level(
        coarse_mesh::KSMesh{T, N},
        fine_mesh::KSMesh{T, N}
) where {T <: Real, N}
    new_elements = Vector{KSCell{T, N}}()

    for fine_el in fine_mesh.cells
        if fine_el.parent in coarse_mesh.cells
            push!(new_elements, fine_el)
        else
            refined_elements = AdaptiveMethods.h_refine(fine_el.parent)
            append!(new_elements, refined_elements)
        end
    end

    return KSMesh(
        elements = new_elements,
        tensor_product_masks = fine_mesh.tensor_product_masks,
        location_matrices = fine_mesh.location_matrices,
        standard_elements = fine_mesh.standard_elements,  # Reuse existing standard elements
        global_error_estimate = fine_mesh.global_error_estimate
    )
end

"""
	v_cycle(hierarchy::Vector{KSMesh{T,N}}, f::Function, u::Vector{T}, level::Int) where {T<:Real,N} -> Vector{T}

Perform a V-cycle for the multi-level method.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector
- `level::Int`: The current level in the hierarchy

# Returns
- `Vector{T}`: The updated solution vector
"""
function v_cycle(
        hierarchy::Vector{KSMesh{T, N}},
        f::Function,
        u::Vector{T},
        level::Int
) where {T <: Real, N}
    if level == 1
        return solve_coarse_problem(hierarchy[1], f)
    else
        u = smooth(hierarchy[level], f, u)
        r = compute_residual(hierarchy[level], f, u)
        r_coarse = IntergridOperators.restrict(r, hierarchy[level], hierarchy[level - 1])
        e_coarse = v_cycle(hierarchy, x -> r_coarse, zeros(length(r_coarse)), level - 1)
        u += IntergridOperators.prolongate(e_coarse, hierarchy[level - 1], hierarchy[level])
        return smooth(hierarchy[level], f, u)
    end
end

"""
	w_cycle(hierarchy::Vector{KSMesh{T,N}}, f::Function, u::Vector{T}, level::Int) where {T<:Real,N} -> Vector{T}

Perform a W-cycle for the multi-level method.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector
- `level::Int`: The current level in the hierarchy

# Returns
- `Vector{T}`: The updated solution vector
"""
function w_cycle(
        hierarchy::Vector{KSMesh{T, N}},
        f::Function,
        u::Vector{T},
        level::Int
) where {T <: Real, N}
    if level == 1
        return solve_coarse_problem(hierarchy[1], f)
    else
        u = smooth(hierarchy[level], f, u)
        r = compute_residual(hierarchy[level], f, u)
        r_coarse = IntergridOperators.restrict(r, hierarchy[level], hierarchy[level - 1])
        e_coarse = w_cycle(hierarchy, x -> r_coarse, zeros(length(r_coarse)), level - 1)
        e_coarse = w_cycle(hierarchy, x -> r_coarse, e_coarse, level - 1)
        u += IntergridOperators.prolongate(e_coarse, hierarchy[level - 1], hierarchy[level])
        return smooth(hierarchy[level], f, u)
    end
end

"""
	full_multigrid(hierarchy::Vector{KSMesh{T,N}}, f::Function) where {T<:Real,N} -> Vector{T}

Perform a full multigrid cycle.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy
- `f::Function`: The right-hand side function

# Returns
- `Vector{T}`: The solution vector
"""
function full_multigrid(hierarchy::Vector{KSMesh{T, N}}, f::Function) where {T <: Real, N}
    u = solve_coarse_problem(hierarchy[1], f)
    for level in 2:length(hierarchy)
        u = IntergridOperators.prolongate(u, hierarchy[level - 1], hierarchy[level])
        u = v_cycle(hierarchy[1:level], f, u, level)
    end
    return u
end

"""
	geometric_multigrid(A::AbstractMatrix, b::AbstractVector, mesh_hierarchy::Vector{KSMesh{T,N}};
						cycle_type::Symbol=:v, max_iterations::Int=100, tolerance::Float64=1e-8) where {T<:Real,N} -> AbstractVector

Solves the linear system `Ax = b` using the geometric multigrid method.

# Arguments
- `A::AbstractMatrix`: The coefficient matrix
- `b::AbstractVector`: The right-hand side vector
- `mesh_hierarchy::Vector{KSMesh{T,N}}`: A vector representing the mesh hierarchy for the multigrid method
- `cycle_type::Symbol`: The type of multigrid cycle to use (`:v`, `:w`, or `:fmg`). Default is `:v`
- `max_iterations::Int`: Maximum number of iterations to perform. Default is 100
- `tolerance::Float64`: Convergence tolerance. Default is `1e-8`

# Returns
- `AbstractVector`: The solution vector `x`
"""
function geometric_multigrid(
        A::AbstractMatrix,
        b::AbstractVector,
        mesh_hierarchy::Vector{KSMesh{T, N}};
        cycle_type::Symbol = :v,
        max_iterations::Int = 100,
        tolerance::Float64 = 1e-8
) where {T <: Real, N}
    x = zeros(T, length(b))
    r = b - A * x

    for iter in 1:max_iterations
        if cycle_type == :v
            x = v_cycle(mesh_hierarchy, t -> b - A * t, x, length(mesh_hierarchy))
        elseif cycle_type == :w
            x = w_cycle(mesh_hierarchy, t -> b - A * t, x, length(mesh_hierarchy))
        elseif cycle_type == :fmg
            x = full_multigrid(mesh_hierarchy, t -> b - A * t)
        else
            error("Unknown cycle type: $cycle_type")
        end

        r = b - A * x
        if norm(r) < tolerance
            @info "Converged in $iter iterations"
            return x
        end
    end

    @warn "Did not converge within $max_iterations iterations"
    return x
end

"""
	algebraic_multigrid(A::AbstractMatrix, b::AbstractVector; max_iterations::Int=100, tolerance::Float64=1e-8) -> AbstractVector

Performs algebraic multigrid method to solve a linear system of equations.

# Arguments
- `A::AbstractMatrix`: The coefficient matrix of the linear system
- `b::AbstractVector`: The right-hand side vector of the linear system
- `max_iterations::Int`: The maximum number of iterations (default: 100)
- `tolerance::Float64`: The tolerance for convergence (default: 1e-8)

# Returns
- `AbstractVector`: The solution vector
"""
function algebraic_multigrid(
        A::AbstractMatrix,
        b::AbstractVector;
        max_iterations::Int = 100,
        tolerance::Float64 = 1e-8
)
    ml = AlgebraicMultigrid.ruge_stuben(A)
    x, convergence_history = AlgebraicMultigrid.solve(
        ml,
        b,
        maxiter = max_iterations,
        tol = tolerance,
        log = true
    )

    if convergence_history.converged
        @info "Converged in $(convergence_history.niter) iterations"
    else
        @warn "Did not converge within $max_iterations iterations"
    end

    return x
end

# Helper functions for the multigrid methods

"""
	solve_coarse_problem(coarse_mesh::KSMesh{T,N}, f::Function) where {T<:Real,N} -> Vector{T}

Solve the coarse problem exactly.

# Arguments
- `coarse_mesh::KSMesh{T,N}`: The coarse mesh
- `f::Function`: The right-hand side function

# Returns
- `Vector{T}`: The solution on the coarse mesh
"""
function solve_coarse_problem(coarse_mesh::KSMesh{T, N}, f::Function) where {T <: Real, N}
    A = CommonMethods.assemble_system_matrix(coarse_mesh)
    b = CommonMethods.assemble_rhs_vector(coarse_mesh, f)
    return A \ b
end

"""
	smooth(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N} -> Vector{T}

Perform a smoothing step.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector

# Returns
- `Vector{T}`: The smoothed solution vector
"""
function smooth(mesh::KSMesh{T, N}, f::Function, u::Vector{T}) where {T <: Real, N}
    num_iterations = 3  # Number of smoothing iterations
    for _ in 1:num_iterations
        u = gauss_seidel_iteration(mesh, f, u)
    end
    return u
end

"""
	gauss_seidel_iteration(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N} -> Vector{T}

Perform a single Gauss-Seidel iteration.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector

# Returns
- `Vector{T}`: The updated solution vector after one iteration
"""
function gauss_seidel_iteration(
        mesh::KSMesh{T, N},
        f::Function,
        u::Vector{T}
) where {T <: Real, N}
    A = CommonMethods.assemble_system_matrix(mesh)
    b = CommonMethods.assemble_rhs_vector(mesh, f)
    n = length(u)
    for i in 1:n
        u[i] = (
            b[i] - sum(A[i, j] * u[j] for j in 1:(i - 1)) -
            sum(A[i, j] * u[j] for j in (i + 1):n)
        ) / A[i, i]
    end
    return u
end

"""
	compute_residual(mesh::KSMesh{T,N}, f::Function, u::Vector{T}) where {T<:Real,N} -> Vector{T}

Compute the residual for the current solution.

# Arguments
- `mesh::KSMesh{T,N}`: The current mesh
- `f::Function`: The right-hand side function
- `u::Vector{T}`: The current solution vector

# Returns
- `Vector{T}`: The residual vector
"""
function compute_residual(
        mesh::KSMesh{T, N}, f::Function, u::Vector{T}) where {T <: Real, N}
    A = CommonMethods.assemble_system_matrix(mesh)
    b = CommonMethods.assemble_rhs_vector(mesh, f)
    return b - A * u
end

"""
	validate_mesh_hierarchy(hierarchy::Vector{KSMesh{T,N}}) where {T<:Real,N}

Validate the mesh hierarchy for consistency.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy to validate

# Throws
- `ArgumentError`: If the hierarchy is invalid
"""
function validate_mesh_hierarchy(hierarchy::Vector{KSMesh{T, N}}) where {T <: Real, N}
    isempty(hierarchy) && throw(ArgumentError("Mesh hierarchy cannot be empty"))

    for i in 2:length(hierarchy)
        if hierarchy[i].elements[1].dimension != hierarchy[i - 1].elements[1].dimension
            throw(ArgumentError("Inconsistent dimensions in mesh hierarchy"))
        end
        if length(hierarchy[i].elements) <= length(hierarchy[i - 1].elements)
            throw(
                ArgumentError(
                "Each level in the hierarchy should have more elements than the previous one",
            ),
            )
        end
    end
end

"""
	compute_mesh_complexity(hierarchy::Vector{KSMesh{T,N}}) where {T<:Real,N} -> Float64

Compute the mesh complexity of the hierarchy.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy

# Returns
- `Float64`: The mesh complexity
"""
function compute_mesh_complexity(hierarchy::Vector{KSMesh{T, N}}) where {T <: Real, N}
    validate_mesh_hierarchy(hierarchy)

    total_dofs = sum(sum(CommonMethods.element_dofs(el) for el in mesh.cells)
    for mesh in hierarchy)
    fine_grid_dofs = sum(CommonMethods.element_dofs(el) for el in hierarchy[end].elements)

    return total_dofs / fine_grid_dofs
end

"""
	estimate_convergence_factor(hierarchy::Vector{KSMesh{T,N}}, f::Function, u_exact::Vector{T}, num_iterations::Int) where {T<:Real,N} -> Float64

Estimate the convergence factor of the multigrid method.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy
- `f::Function`: The right-hand side function
- `u_exact::Vector{T}`: The exact solution
- `num_iterations::Int`: The number of iterations to perform

# Returns
- `Float64`: The estimated convergence factor
"""
function estimate_convergence_factor(
        hierarchy::Vector{KSMesh{T, N}},
        f::Function,
        u_exact::Vector{T},
        num_iterations::Int
) where {T <: Real, N}
    validate_mesh_hierarchy(hierarchy)

    u = zeros(T, length(u_exact))
    errors = zeros(T, num_iterations + 1)

    errors[1] = norm(u_exact - u)

    for i in 1:num_iterations
        u = v_cycle(hierarchy, f, u, length(hierarchy))
        errors[i + 1] = norm(u_exact - u)
    end

    convergence_factors = errors[2:end] ./ errors[1:(end - 1)]
    return geometric_mean(convergence_factors)
end

"""
	geometric_mean(x::AbstractVector{T}) where {T<:Real} -> Float64

Compute the geometric mean of a vector of numbers.

# Arguments
- `x::AbstractVector{T}`: The vector of numbers

# Returns
- `Float64`: The geometric mean
"""
function geometric_mean(x::AbstractVector{T}) where {T <: Real}
    return exp(sum(log, x) / length(x))
end

end # module MultiLevelMethods
