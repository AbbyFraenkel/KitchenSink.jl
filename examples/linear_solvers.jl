# Define the problem
prob = KSPDEProblem(equation, domain, boundary_conditions, initial_condition)

# Define the mesh
mesh = generate_mesh(prob.domain, discretization)

# Solver configurations
amg_solver = KSAMGSolver(max_iter=100, tol=1e-6, smoother=:jacobi)
direct_solver = KSDirectSolver(method=:lu)
iterative_solver = KSIterativeSolver(method=:cg, max_iter=100, tol=1e-6, preconditioner=nothing)

# Solve the problem using different solvers
solution_amg = solve_equation(prob, mesh, amg_solver)
solution_direct = solve_equation(prob, mesh, direct_solver)
solution_iterative = solve_equation(prob, mesh, iterative_solver)

# For multiple right-hand sides
rhs = hcat(b1, b2, b3)  # Multiple right-hand sides
solution_amg_multiple = solve_multiple_rhs(prob, mesh, rhs, amg_solver)
solution_direct_multiple = solve_multiple_rhs(prob, mesh, rhs, direct_solver)
solution_iterative_multiple = solve_multiple_rhs(prob, mesh, rhs, iterative_solver)
