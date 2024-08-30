using .ProblemTypes

# Define the domain and time span for a cylinder growing in the axial direction
domain = ((0.0, 1.0), (0.0, 1.0))  # Initial domain: radius [0, 1], axial [0, 1]
tspan = (0.0, 0.5)  # Time span from 0 to 0.5 seconds

# Define PDE dynamics: heat equation in cylindrical coordinates
alpha = 0.01
pde_function = (u, nodes_r, nodes_z, t) -> alpha * (derivative_matrix!(nodes_r) * u + u * derivative_matrix!(nodes_z)')

# Define boundary conditions: zero temperature at the boundaries
bc_function = (u, x, t) -> 0.0

# Define initial condition: initial temperature is zero everywhere
ic_function = (nodes_r, nodes_z) -> zeros(length(nodes_r), length(nodes_z))

# Define moving boundary function: speed proportional to temperature gradient at the boundary
boundary_function = (u, nodes_z, s) -> -0.1 * derivative_matrix!(nodes_z)[end, :] * u[end, :]  # Example scaling factor

# Solve the moving boundary problem in cylindrical coordinates
results = solve_moving_boundary_pde_cylindrical(domain, tspan, pde_function, bc_function, ic_function, boundary_function)

# Plot results for boundary movement
plot(results["time"], results["boundary"], xlabel="Time", ylabel="Boundary Position (Axial)",
     title="Cylindrical Moving Boundary: Numerical vs Analytical", label="Numerical Boundary Position")

# Analytical boundary position for comparison (if available)
analytical_s = t -> 2 * sqrt(alpha * t)  # Example analytical boundary
plot!(results["time"], analytical_s.(results["time"]), label="Analytical Boundary", linestyle=:dash)

# Plot temperature profile at the final time step
heatmap(linspace(domain[1][1], domain[1][2], length(results["solution"][end])),
        linspace(domain[2][1], results["boundary"][end], length(results["solution"][end])),
        results["solution"][end], xlabel="Radius", ylabel="Axial Position", title="Temperature Profile at Final Time")
