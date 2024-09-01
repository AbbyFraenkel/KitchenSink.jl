module Optimization

using LinearAlgebra, ForwardDiff, JuMP, Ipopt

    using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..ProblemTypes, ..LinearSolvers


export discretize_and_optimize, solve_optimal_control_problem, create_jump_model

"""
    discretize_and_optimize(problem::AbstractKSProblem, num_elements::NTuple{N,Int}, tspan::Tuple{T,T}, degree::Int, solver_options::KSSolverOptions) where {N, T<:Real}

Discretize the given problem in space and time using a simultaneous approach and solve it as an optimization problem using JuMP.

# Arguments
- `problem::AbstractKSProblem`: The problem to be discretized and optimized.
- `num_elements::NTuple{N,Int}`: Number of elements for spatial discretization in each dimension.
- `tspan::Tuple{T,T}`: Time span for the problem.
- `degree::Int`: Polynomial degree for the basis functions.
- `solver_options::KSSolverOptions`: Solver options for the KitchenSink solver.

# Returns
- `model::JuMP.Model`: The JuMP model representing the discretized problem.
- `solution::Array{T,N+1}`: The optimal solution (N spatial dimensions + 1 time dimension).
"""
function discretize_and_optimize(problem::AbstractKSProblem, num_elements::NTuple{N,Int}, tspan::Tuple{T,T}, degree::Int, solver_options::KSSolverOptions) where {N,T<:Real}
    # Create spatial mesh
    mesh = Preprocessing.create_mesh(problem.domain, num_elements, degree, problem.coordinate_system)

    # Create time discretization
    t_nodes, t_weights = SpectralMethods.create_nodes(solver_options.time_elements, tspan...)

    # Assemble the problem
    A, b, _ = assemble_problem(problem, solver_options)

    # Create JuMP model
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "max_iter" => 1000))

    # Define variables
    @variable(model, u[1:size(A, 1), 1:length(t_nodes)])

    # Set initial conditions
    @constraint(model, [i = 1:size(A, 1)], u[i, 1] == problem.initial_condition(mesh.nodes[i].coordinates))

    # Set PDE constraints using simultaneous approach
    for t in 1:length(t_nodes)-1
        dt = t_nodes[t+1] - t_nodes[t]
        @constraint(model, A * u[:, t+1] == A * u[:, t] + dt * b)
    end

    # Set objective function (example: minimize the L2 norm of the solution)
    @objective(model, Min, sum(sum(u[i, t]^2 for i in 1:size(A, 1)) * t_weights[t] for t in 1:length(t_nodes)))

    # Solve the optimization problem
    optimize!(model)

    # Extract and return the solution
    solution = value.(u)
    return model, solution
end

"""
    solve_optimal_control_problem(problem::KSOptimalControlProblem, solver::AbstractKSSolver)

Solve an optimal control problem using the specified solver.

# Arguments
- problem: KSOptimalControlProblem - The optimal control problem to solve.
- solver: AbstractKSSolver - The solver to use for the optimization.

# Returns
- solution: OptimizationSolution - The solution to the optimization problem.
"""
function solve_optimal_control_problem(problem::KSOptimalControlProblem, solver::AbstractKSSolver)
    model = create_jump_model(problem)
    set_optimizer(model, Ipopt.Optimizer)

    optimize!(model)

    return extract_solution(model, problem)
end

"""
    create_jump_model(problem::KSOptimalControlProblem)

Create a JuMP model for the given optimal control problem.

# Arguments
- problem: KSOptimalControlProblem - The optimal control problem to model.

# Returns
- model: Model - The JuMP model representing the optimization problem.
"""
function create_jump_model(problem::KSOptimalControlProblem)
    model = Model()

    # Define variables
    @variable(model, x[1:length(problem.initial_state), 1:problem.num_time_steps])
    @variable(model, u[1:problem.num_controls, 1:problem.num_time_steps])

    # Set initial conditions
    @constraint(model, [i = 1:length(problem.initial_state)], x[i, 1] == problem.initial_state[i])

    # Set dynamics constraints
    for t in 1:problem.num_time_steps-1
        @constraint(model, x[:, t+1] .== x[:, t] + problem.state_equation(x[:, t], u[:, t]) * problem.dt)
    end

    # Set control bounds
    for i in 1:problem.num_controls
        @constraint(model, u[i, :] .<= problem.control_bounds[i][2])
        @constraint(model, u[i, :] .>= problem.control_bounds[i][1])
    end

    # Set objective function
    @objective(model, Min,
        sum(problem.cost_function(x[:, t], u[:, t]) for t in 1:problem.num_time_steps) +
        problem.terminal_cost(x[:, end])
    )

    return model
end

"""
    extract_solution(model::Model, problem::KSOptimalControlProblem)

Extract the solution from a solved JuMP model.

# Arguments
- model: Model - The solved JuMP model.
- problem: KSOptimalControlProblem - The original optimal control problem.

# Returns
- solution: OptimizationSolution - The extracted solution.
"""
function extract_solution(model::Model, problem::KSOptimalControlProblem)
    x_sol = value.(model[:x])
    u_sol = value.(model[:u])
    obj_value = objective_value(model)

    return OptimizationSolution(x_sol, u_sol, obj_value)
end

end # module Optimization
