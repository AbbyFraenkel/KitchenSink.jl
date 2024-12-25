using ..KitchenSink

# Example 1: Solving a 2D Poisson equation
function poisson_example()
    println("Example 1: 2D Poisson Equation")

    # Define the problem
    domain = ((0.0, 1.0), (0.0, 1.0))
    f(x, y) = 2π^2 * sin(π * x) * sin(π * y)  # Source term
    u_exact(x, y) = sin(π * x) * sin(π * y)   # Exact solution
    bc(x, y) = 0.0  # Dirichlet boundary condition

    problem = KSPDEProblem(
        (D, u, x) -> -sum(D[i] * u for i in 1:2) - f(x...),  # -Δu = f
        bc,
        domain,
        KSCartesianCoordinates{2}((0.0, 0.0))
    )

    # Create solver options
    options = KSSolverOptions(
        max_iterations = 100,
        tolerance = 1e-6,
        adaptive = true,
        max_levels = 5,
        smoothness_threshold = 0.1,
        initial_elements = (4, 4),
        initial_degree = 3
    )

    # Solve the problem
    solution = solve_equation(problem, options)

    # Visualize the solution
    plot_solution(problem.mesh, solution)

    # Compute and print the error
    error = compute_l2_error(problem.mesh, solution, u_exact)
    println("L2 error: ", error)
end

# Example 2: Solving an ODE (Lorenz system)
function lorenz_example()
    println("\nExample 2: Lorenz System ODE")

    function lorenz!(du, u, p, t)
        σ, ρ, β = p
        du[1] = σ * (u[2] - u[1])
        du[2] = u[1] * (ρ - u[3]) - u[2]
        du[3] = u[1] * u[2] - β * u[3]
    end

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    p = [10.0, 28.0, 8 / 3]

    problem = KSODEProblem(lorenz!, u0, tspan, p)

    # Solve the ODE
    solution = solve_equation(problem, KSRKF45())

    # Visualize the solution
    plot_phase_portrait(solution)
end

# Example 3: Optimization problem (optimal control)
function optimal_control_example()
    println("\nExample 3: Optimal Control Problem")

    # Define the optimal control problem
    function dynamics(x, u, t)
        return [x[2], u[1]]
    end

    function cost(x, u, t)
        return 0.5 * (x[1]^2 + x[2]^2 + u[1]^2)
    end

    x0 = [1.0, 0.0]
    tf = 5.0
    umin, umax = -1.0, 1.0

    problem = KSOptimalControlProblem(
        dynamics,
        cost,
        x0,
        tf,
        2,  # state dimension
        1,  # control dimension
        [(umin, umax)]  # control bounds
    )

    # Solve the optimal control problem
    solution = solve_optimal_control_problem(problem)

    # Visualize the results
    plot_optimal_control_solution(solution)
end

# Run the examples
poisson_example()
lorenz_example()
optimal_control_example()
