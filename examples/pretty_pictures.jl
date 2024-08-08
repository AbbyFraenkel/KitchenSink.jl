# example_visualizations.jl

using KitchenSink.Solvers
using KitchenSink.Visualization

# 1. Solve and visualize an ODE
function lorenz_system(u, p, t)
    σ, ρ, β = p
    [σ*(u[2]-u[1]), u[1]*(ρ-u[3])-u[2], u[1]*u[2]-β*u[3]]
end

p = [10.0, 28.0, 8/3]
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz_system, u0, tspan, p)
sol = solve_and_visualize(prob, ODESolver())

# 2. Visualize phase portrait
plot_phase_portrait(prob, sol.u)

# 3. Solve and visualize a PDE (1D Heat Equation)
function heat_equation(u, p, t)
    α = p[1]
    Δu = diff(u, dims=1)
    α * Δu
end

u0 = sin.(range(0, π, length=100))
tspan = (0.0, 1.0)
p = [0.01]
domain = (0.0, π)
prob = PDEProblem(heat_equation, u0, tspan, p, domain)
sol = solve_and_visualize(prob, PDESolver())

# 4. Animate the solution
animate_solution(prob, sol.u, sol.t)

# 5. Visualize error (assuming we have an analytical solution)
analytical_solution = (x, t) -> exp(-p[1]*t) * sin(x)
t_final = 1.0
x = range(domain[1], domain[2], length=100)
exact_sol = analytical_solution.(x, t_final)
plot_error(prob, sol.u[:, end], exact_sol)

# 6. Visualize eigenspectrum of the discretized operator
A = discretize_operator(prob)
plot_eigenspectrum(A)

# 7. Plot stability region of a numerical method
plot_stability_region(backward_euler, 1)

# 8. Interactive solution explorer
GLMakie.activate()
interactive_solution_explorer(prob, sol.u, sol.t)

# 9. Parametric study visualization
α_range = 0.01:0.01:0.1
results = Dict(α => solve(PDEProblem(heat_equation, u0, tspan, [α], domain)) for α in α_range)
plot_parametric_study(results, "α")

# 10. Sensitivity analysis visualization
sensitivities = Dict("α" => 1.0, "L" => 0.5)  # Example sensitivities
plot_sensitivity_analysis(sensitivities)
