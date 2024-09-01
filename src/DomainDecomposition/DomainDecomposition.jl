module DomainDecomposition

using LinearAlgebra, SparseArrays
using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..ProblemTypes
using ..LinearSolvers, ..Preprocessing

export decompose_domain, create_interface_conditions, schwarz_method
export strang_splitting, solve_decomposed_problem

"""
    Subdomain

Represents a subdomain in the domain decomposition.

Fields:
- `mesh::KSMesh`: The mesh of the subdomain.
- `global_to_local::Dict{Int, Int}`: Mapping from global to local DOF indices.
- `local_to_global::Dict{Int, Int}`: Mapping from local to global DOF indices.
"""
struct Subdomain
    mesh::KSMesh
    global_to_local::Dict{Int, Int}
    local_to_global::Dict{Int, Int}
end

"""
    decompose_domain(mesh::KSMesh, num_subdomains::Int)

Decompose the domain into subdomains for parallel computing.

# Arguments
- `mesh::KSMesh`: The global mesh to be decomposed.
- `num_subdomains::Int`: The number of subdomains to create.

# Returns
- A vector of Subdomain objects representing the subdomains.
"""
function decompose_domain(mesh::KSMesh, num_subdomains::Int)
    # Assume 2D rectangular domain for simplicity
    nx = floor(Int, sqrt(length(mesh.elements)))
    ny = ceil(Int, length(mesh.elements) / nx)

    nx_sub = floor(Int, nx / sqrt(num_subdomains))
    ny_sub = ceil(Int, num_subdomains / (nx / nx_sub))

    subdomains = Subdomain[]

    for i in 1:nx_sub:nx, j in 1:ny_sub:ny
        elements = KSElement[]
        global_to_local = Dict{Int, Int}()
        local_to_global = Dict{Int, Int}()

        for ei in i:min(i+nx_sub-1, nx), ej in j:min(j+ny_sub-1, ny)
            element_idx = (ei-1) * ny + ej
            push!(elements, mesh.elements[element_idx])

            for (local_idx, global_idx) in mesh.location_matrices[element_idx]
                if !haskey(global_to_local, global_idx)
                    new_local_idx = length(global_to_local) + 1
                    global_to_local[global_idx] = new_local_idx
                    local_to_global[new_local_idx] = global_idx
                end
            end
        end

        submesh = KSMesh(
            elements,
            [m for (idx, m) in enumerate(mesh.tensor_product_masks) if mesh.elements[idx] in elements],
            [Dict(k => global_to_local[v] for (k, v) in m) for (idx, m) in enumerate(mesh.location_matrices) if mesh.elements[idx] in elements],
            mesh.basis_functions,
            0.0
        )

        push!(subdomains, Subdomain(submesh, global_to_local, local_to_global))
    end

    return subdomains
end

"""
    create_interface_conditions(subdomains::Vector{Subdomain})

Create interface conditions between subdomains.

# Arguments
- `subdomains::Vector{Subdomain}`: The vector of subdomains.

# Returns
- A vector of tuples (i, j, shared_dofs) representing interface conditions.
"""
function create_interface_conditions(subdomains::Vector{Subdomain})
    interface_conditions = Tuple{Int, Int, Vector{Int}}[]

    for i in 1:length(subdomains)
        for j in (i+1):length(subdomains)
            shared_dofs = intersect(keys(subdomains[i].global_to_local), keys(subdomains[j].global_to_local))
            if !isempty(shared_dofs)
                push!(interface_conditions, (i, j, collect(shared_dofs)))
            end
        end
    end

    return interface_conditions
end

"""
    schwarz_method(problem::AbstractKSProblem, subdomains::Vector{Subdomain}, interface_conditions, max_iterations::Int, tolerance::Real)

Implement the Schwarz alternating method for domain decomposition.

# Arguments
- `problem::AbstractKSProblem`: The problem to be solved.
- `subdomains::Vector{Subdomain}`: The vector of subdomains.
- `interface_conditions`: The interface conditions between subdomains.
- `max_iterations::Int`: Maximum number of Schwarz iterations.
- `tolerance::Real`: Convergence tolerance.

# Returns
- The solution on the global domain.
"""
function schwarz_method(problem::AbstractKSProblem, subdomains::Vector{Subdomain}, interface_conditions, max_iterations::Int, tolerance::Real)
    solutions = [zeros(CommonMethods.total_dofs(subdomain.mesh)) for subdomain in subdomains]
    global_solution = zeros(maximum(maximum(keys(sd.global_to_local)) for sd in subdomains))

    for _ in 1:max_iterations
        old_global_solution = copy(global_solution)

        for (i, subdomain) in enumerate(subdomains)
            # Update boundary conditions based on neighboring subdomains
            update_boundary_conditions!(problem, subdomain, interface_conditions, solutions, subdomains)

            # Solve the local problem on the subdomain
            local_solution, _ = ProblemTypes.solve_problem(problem, subdomain.mesh, KSSolverOptions())

            # Update the global solution
            for (local_idx, global_idx) in subdomain.local_to_global
                global_solution[global_idx] = local_solution[local_idx]
            end

            solutions[i] = local_solution
        end

        # Check for convergence
        if norm(global_solution - old_global_solution) < tolerance
            break
        end
    end

    return global_solution
end

"""
    update_boundary_conditions!(problem::AbstractKSProblem, subdomain::Subdomain, interface_conditions, solutions, subdomains)

Update the boundary conditions for a subdomain based on neighboring solutions.

# Arguments
- `problem::AbstractKSProblem`: The problem being solved.
- `subdomain::Subdomain`: The subdomain to update.
- `interface_conditions`: The interface conditions between subdomains.
- `solutions`: The current solutions for all subdomains.
- `subdomains`: The list of all subdomains.
"""
function update_boundary_conditions!(problem::AbstractKSProblem, subdomain::Subdomain, interface_conditions, solutions, subdomains)
    for (i, j, shared_dofs) in interface_conditions
        if i == findfirst(sd -> sd === subdomain, subdomains)
            for dof in shared_dofs
                local_dof = subdomain.global_to_local[dof]
                neighbor_subdomain = subdomains[j]
                neighbor_local_dof = neighbor_subdomain.global_to_local[dof]
                problem.boundary_conditions[local_dof] = solutions[j][neighbor_local_dof]
            end
        elseif j == findfirst(sd -> sd === subdomain, subdomains)
            for dof in shared_dofs
                local_dof = subdomain.global_to_local[dof]
                neighbor_subdomain = subdomains[i]
                neighbor_local_dof = neighbor_subdomain.global_to_local[dof]
                problem.boundary_conditions[local_dof] = solutions[i][neighbor_local_dof]
            end
        end
    end
end

"""
    strang_splitting(problem::AbstractKSProblem, mesh::KSMesh, dt::Real, num_steps::Int)

Implement Strang splitting for solving a problem with multiple operators.

# Arguments
- `problem::AbstractKSProblem`: The problem to be solved.
- `mesh::KSMesh`: The mesh for discretization.
- `dt::Real`: The time step size.
- `num_steps::Int`: The number of time steps.

# Returns
- The solution at the final time step.
"""
function strang_splitting(problem::AbstractKSProblem, mesh::KSMesh, dt::Real, num_steps::Int)
    # Assume the problem can be split into two operators: A and B
    A, B = split_operators(problem)

    u = get_initial_condition(problem, mesh)

    for _ in 1:num_steps
        # First half-step of operator A
        u = solve_operator(A, u, dt/2, mesh)

        # Full step of operator B
        u = solve_operator(B, u, dt, mesh)

        # Second half-step of operator A
        u = solve_operator(A, u, dt/2, mesh)
    end

    return u
end

"""
    split_operators(problem::AbstractKSProblem)

Split the problem into two operators for Strang splitting.

# Arguments
- `problem::AbstractKSProblem`: The problem to be split.

# Returns
- A tuple of two operators (A, B).
"""
function split_operators(problem::AbstractKSProblem)
    if isa(problem, KSPDEProblem)
        # Example: Split a reaction-diffusion equation into diffusion and reaction parts
        diffusion_op = (u, t) -> problem.pde(u, t) - problem.reaction_term(u, t)
        reaction_op = (u, t) -> problem.reaction_term(u, t)
        return (diffusion_op, reaction_op)
    else
        error("Operator splitting not implemented for $(typeof(problem))")
    end
end

"""
    solve_operator(operator, u::AbstractVector, dt::Real, mesh::KSMesh)

Solve a single operator for a given time step.

# Arguments
- `operator`: The operator to be solved.
- `u::AbstractVector`: The current solution vector.
- `dt::Real`: The time step size.
- `mesh::KSMesh`: The mesh for discretization.

# Returns
- The updated solution vector after applying the operator.
"""
function solve_operator(operator, u::AbstractVector, dt::Real, mesh::KSMesh)
    # Create a temporary problem for this operator
    temp_problem = KSPDEProblem(operator, mesh.boundary_conditions, (0, dt), mesh.domain, u)

    # Solve the temporary problem using existing solvers
    solution, _ = ProblemTypes.solve_problem(temp_problem, mesh, KSSolverOptions(max_iterations=100, tolerance=1e-6, adaptive=false))

    return solution
end

"""
    solve_decomposed_problem(problem::AbstractKSProblem, mesh::KSMesh, num_subdomains::Int, solver_options::KSSolverOptions)

Solve a problem using domain decomposition.

# Arguments
- `problem::AbstractKSProblem`: The problem to be solved.
- `mesh::KSMesh`: The global mesh.
- `num_subdomains::Int`: The number of subdomains to use.
- `solver_options::KSSolverOptions`: Options for the solver.

# Returns
- The solution on the global domain.
"""
function solve_decomposed_problem(problem::AbstractKSProblem, mesh::KSMesh, num_subdomains::Int, solver_options::KSSolverOptions)
    subdomains = decompose_domain(mesh, num_subdomains)
    interface_conditions = create_interface_conditions(subdomains)
    return schwarz_method(problem, subdomains, interface_conditions, solver_options.max_iterations, solver_options.tolerance)
end

end # module DomainDecomposition
