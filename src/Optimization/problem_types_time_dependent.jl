# Time-dependent optimization problems

function create_time_dependent_optimization_matrices(
    problem::KSOptimalControlProblem{T},
    mesh::KSMesh{T,N}) where {T<:Real,N}

    # Get dimensions
    num_time_steps = problem.num_time_steps
    num_states = problem.num_vars
    num_controls = problem.num_controls

    # Total DOFs: states + controls at each time
    total_vars = num_states + num_controls
    num_dof = get_total_dof(mesh) * total_vars * num_time_steps

    # Initialize system
    A = spzeros(T, num_dof, num_dof)
    b = zeros(T, num_dof)

    # Add dynamics constraints
    add_dynamics_constraints!(
        A, b, problem, mesh)

    # Add control constraints
    add_control_constraints!(
        A, b, problem)

    # Add objective terms
    add_objective_terms!(
        A, b, problem, mesh)

    return A, b
end

function add_dynamics_constraints!(
    A::SparseMatrixCSC{T},
    b::Vector{T},
    problem::KSOptimalControlProblem{T},
    mesh::KSMesh{T,N}) where {T<:Real,N}

    num_states = problem.num_vars
    num_controls = problem.num_controls
    dt = problem.dt

    for k in 1:(problem.num_time_steps-1)
        # Get time indices
        tk_idx = get_time_index(k, num_states, num_controls)
        tk1_idx = get_time_index(k+1, num_states, num_controls)

        # Add state transition matrix
        state_range = get_state_range(tk_idx, num_states)
        next_state_range = get_state_range(tk1_idx, num_states)

        # Add dynamics using spectral discretization
        add_state_dynamics!(
            A, state_range, next_state_range,
            problem.state_equations, dt, mesh)

        # Add control input effects
        control_range = get_control_range(tk_idx, num_states, num_controls)
        add_control_effects!(
            A, control_range, next_state_range,
            problem.state_equations, dt)
    end
end

function add_state_dynamics!(
    A::SparseMatrixCSC{T},
    state_range::UnitRange{Int},
    next_state_range::UnitRange{Int},
    state_eqns::Vector{Function},
    dt::T,
    mesh::KSMesh{T,N}) where {T<:Real,N}

    # Get standard cell for spectral discretization
    standard_cell = get_or_create_standard_cell(
        first(mesh.cells).standard_cell_key...)

    # Add spectral approximation of dynamics
    for cell in mesh.cells
        if cell.is_fictitious
            # Get differentiation operators
            diff_ops = get_diff_operators(standard_cell)

            # Assemble local dynamics matrices
            local_A = assemble_dynamics_matrices(
                cell, state_eqns, diff_ops, dt)

            # Add to global system
            add_local_dynamics!(
                A, local_A, state_range, next_state_range,
                cell.node_map)
        end
    end
end

function add_control_effects!(
    A::SparseMatrixCSC{T},
    control_range::UnitRange{Int},
    next_state_range::UnitRange{Int},
    state_eqns::Vector{Function},
    dt::T) where T

    # Add control input matrices
    for (i, eq) in enumerate(state_eqns)
        state_idx = next_state_range[i]

        for j in 1:length(control_range)
            control_idx = control_range[j]

            # Add control effect
            A[state_idx, control_idx] = dt
        end
    end
end

function add_control_constraints!(
    A::SparseMatrixCSC{T},
    b::Vector{T},
    problem::KSOptimalControlProblem{T}) where T

    num_states = problem.num_vars
    num_controls = problem.num_controls

    for k in 1:(problem.num_time_steps-1)
        # Get control indices
        tk_idx = get_time_index(k, num_states, num_controls)
        control_range = get_control_range(tk_idx, num_states, num_controls)

        # Add bound constraints
        for (i, bounds) in enumerate(problem.control_bounds)
            control_idx = control_range[i]

            # Lower bound
            if !isnothing(bounds[1])
                b[control_idx] = bounds[1]
            end

            # Upper bound
            if !isnothing(bounds[2])
                b[control_idx] = bounds[2]
            end
        end
    end
end

function add_objective_terms!(
    A::SparseMatrixCSC{T},
    b::Vector{T},
    problem::KSOptimalControlProblem{T},
    mesh::KSMesh{T,N}) where {T<:Real,N}

    num_states = problem.num_vars
    num_controls = problem.num_controls

    # Add running cost
    for k in 1:(problem.num_time_steps-1)
        tk_idx = get_time_index(k, num_states, num_controls)

        # State cost
        state_range = get_state_range(tk_idx, num_states)
        add_state_cost!(A, state_range, problem.cost_functions)

        # Control cost
        control_range = get_control_range(tk_idx, num_states, num_controls)
        add_control_cost!(A, control_range, problem.cost_functions)
    end

    # Add terminal cost
    final_time_idx = get_time_index(
        problem.num_time_steps, num_states, num_controls)
    final_state_range = get_state_range(final_time_idx, num_states)

    add_terminal_cost!(
        A, final_state_range, problem.terminal_cost)
end

# Utility functions for indexing

function get_time_index(
    k::Int,
    num_states::Int,
    num_controls::Int)
    return (k-1) * (num_states + num_controls) + 1
end

function get_state_range(
    time_idx::Int,
    num_states::Int)
    return time_idx:(time_idx + num_states - 1)
end

function get_control_range(
    time_idx::Int,
    num_states::Int,
    num_controls::Int)
    start_idx = time_idx + num_states
    return start_idx:(start_idx + num_controls - 1)
end

function get_diff_operators(
    standard_cell::StandardKSCell{T,N}) where {T,N}
    return ntuple(N) do dim
        standard_cell.differentiation_matrix_with_boundary[dim]
    end
end
