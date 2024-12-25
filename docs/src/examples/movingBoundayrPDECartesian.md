using LinearAlgebra, SparseArrays, Printf, Plots
using .SpectralMethods
using .ProblemTypes

# 1. 1D Stefan Problem with Melting Ice
function test_stefan_problem()
    # Define the domain and time span
    domain = (0.0, 1.0)
    tspan = (0.0, 0.5)

    # Define PDE dynamics: heat equation u_t = alpha * u_xx
    alpha = 0.01
    pde_function = (u, nodes, t) -> alpha * derivative_matrix!(nodes) * u

    # Define boundary conditions: fixed temperature at the left boundary
    bc_function = (u, x, t) -> 1.0

    # Define initial condition: temperature starts at 0 except for left boundary
    ic_function = x -> zeros(length(x))

    # Define moving boundary function: speed proportional to temperature gradient
    boundary_function = (u, nodes, s) -> -0.5 * derivative_matrix!(nodes)[end, :] * u  # Example scaling factor

    # Solve the moving boundary problem
    results = solve_moving_boundary_pde(
        domain,
        tspan,
        pde_function,
        bc_function,
        ic_function,
        boundary_function
    )

    # Analytical solutions
    analytical_u = x -> erf.(x ./ (2 * sqrt.(alpha * results["time"])))
    analytical_s = t -> 2 * sqrt(alpha * t)

    # Plot results for boundary movement
    plot(
        results["time"],
        results["boundary"],
        xlabel = "Time",
        ylabel = "Boundary Position",
        title = "Stefan Problem: Numerical vs Analytical Boundary Movement"
    )
    plot!(
        results["time"],
        analytical_s.(results["time"]),
        label = "Analytical Boundary",
        linestyle = :dash
    )
    println(
        "Stefan Problem: Max Error in Boundary Position = ",
        maximum(abs.(results["boundary"] - analytical_s.(results["time"])))
    )

    # Plot results for temperature profile
    u_num = results["solution"][end]
    x_vals = linspace(domain[1], results["boundary"][end], length(u_num))
    plot(
        x_vals,
        u_num,
        xlabel = "x",
        ylabel = "Temperature",
        title = "Temperature Profile at Final Time",
        label = "Numerical Solution"
    )
    plot!(x_vals, analytical_u.(x_vals), label = "Analytical Solution", linestyle = :dash)
end

# 2. Semi-Infinite Heat Conduction Problem
function test_semi_infinite_heat_conduction()
    domain = (0.0, 1.0)
    tspan = (0.0, 0.5)
    alpha = 0.01

    pde_function = (u, nodes, t) -> alpha * derivative_matrix!(nodes) * u
    bc_function = (u, x, t) -> 1.0  # Fixed boundary temperature
    ic_function = x -> zeros(length(x))
    boundary_function = (u, nodes, s) -> -1 * derivative_matrix!(nodes)[end, :] * u

    results = solve_moving_boundary_pde(
        domain,
        tspan,
        pde_function,
        bc_function,
        ic_function,
        boundary_function
    )

    # Analytical solutions
    analytical_u = x -> 1.0 * erfc.(x ./ (2 * sqrt.(alpha * results["time"])))
    analytical_s = t -> 2 * sqrt(alpha * t)

    # Plot results for boundary movement
    plot(
        results["time"],
        results["boundary"],
        xlabel = "Time",
        ylabel = "Boundary Position",
        title = "Semi-Infinite Conduction: Numerical vs Analytical Boundary Movement"
    )
    plot!(
        results["time"],
        analytical_s.(results["time"]),
        label = "Analytical Boundary",
        linestyle = :dash
    )
    println(
        "Semi-Infinite Conduction: Max Error in Boundary Position = ",
        maximum(abs.(results["boundary"] - analytical_s.(results["time"])))
    )

    # Plot results for temperature profile
    u_num = results["solution"][end]
    x_vals = linspace(domain[1], results["boundary"][end], length(u_num))
    plot(
        x_vals,
        u_num,
        xlabel = "x",
        ylabel = "Temperature",
        title = "Temperature Profile at Final Time",
        label = "Numerical Solution"
    )
    plot!(x_vals, analytical_u.(x_vals), label = "Analytical Solution", linestyle = :dash)
end

# 3. Solidification of a Pure Substance (Stefan Problem)
function test_solidification()
    domain = (0.0, 1.0)
    tspan = (0.0, 0.5)
    alpha = 0.01

    pde_function = (u, nodes, t) -> alpha * derivative_matrix!(nodes) * u
    bc_function = (u, x, t) -> 1.0  # Fixed boundary temperature at left
    ic_function = x -> zeros(length(x))  # Initial temperature is 0
    boundary_function = (u, nodes, s) -> -1 * derivative_matrix!(nodes)[end, :] * u

    results = solve_moving_boundary_pde(
        domain,
        tspan,
        pde_function,
        bc_function,
        ic_function,
        boundary_function
    )

    # Analytical solutions
    analytical_u = x -> 1.0 * erf.(x ./ (2 * sqrt.(alpha * results["time"])))
    analytical_s = t -> 2 * sqrt(alpha * t)

    # Plot results for boundary movement
    plot(
        results["time"],
        results["boundary"],
        xlabel = "Time",
        ylabel = "Boundary Position",
        title = "Solidification: Numerical vs Analytical Boundary Movement"
    )
    plot!(
        results["time"],
        analytical_s.(results["time"]),
        label = "Analytical Boundary",
        linestyle = :dash
    )
    println(
        "Solidification: Max Error in Boundary Position = ",
        maximum(abs.(results["boundary"] - analytical_s.(results["time"])))
    )

    # Plot results for temperature profile
    u_num = results["solution"][end]
    x_vals = linspace(domain[1], results["boundary"][end], length(u_num))
    plot(
        x_vals,
        u_num,
        xlabel = "x",
        ylabel = "Temperature",
        title = "Temperature Profile at Final Time",
        label = "Numerical Solution"
    )
    plot!(x_vals, analytical_u.(x_vals), label = "Analytical Solution", linestyle = :dash)
end

# Run all tests
test_stefan_problem()
test_semi_infinite_heat_conduction()
test_solidification()
