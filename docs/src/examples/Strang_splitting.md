# Define your differential equation parts
# f1(u, t) = ...  # First part of the ODE
# f2(u, t) = ...  # Second part of the ODE

# Choose appropriate solvers for each part
solver1 = ExplicitRK4Solver()  # For example
solver2 = ImplicitEulerSolver()  # For example

# Set up initial conditions and time step
# u0 = [...]
t = 0.0
dt = 0.01

# Perform a Strang splitting step
u = strang_splitting!(u0, t, dt, f1, f2, solver1, solver2)
