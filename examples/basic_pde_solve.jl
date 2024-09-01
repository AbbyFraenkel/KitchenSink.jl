using ..KitchenSink

# Define the PDE problem
function pde_equation(x, t, u, ∇u, ∇²u)
    return ∇²u - u  # Example: Heat equation
end

domain = (0.0, 1.0, 0.0, 1.0)  # 2D domain
bc(x, t) = 0.0  # Dirichlet boundary condition
ic(x) = sin(π * x[1]) * sin(π * x[2])  # Initial condition

problem = KSProblem(pde_equation, domain, bc, ic, nothing)

# Create solver
options = KSSolverOptions(100, 1e-6, true, :hp)
linear_solver = conjugate_gradient
preconditioner = jacobi_preconditioner
solver = KSSolver(options, linear_solver, preconditioner)

# Solve the PDE
solution = solve_equation(problem, solver)

# Visualize the solution
using Plots
plot(solution, title="Solution to Heat Equation", xlabel="x", ylabel="y")
