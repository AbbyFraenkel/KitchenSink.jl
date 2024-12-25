using JuMP
using Ipopt
using OrdinaryDiffEq
using LinearAlgebra
using SparseArrays
using .JuMPIntegration
using .ProblemTypes
using .KSTypes

# Define a simple PDE problem
function simple_pde(u, x, t)
    return -u + sin(pi * x) * exp(-t)
end

# Define initial and boundary conditions
initial_conditions = x -> sin(pi * x)
boundary_conditions = KSDirichletBC{Float64}(
    value = (x, t) -> 0.0, boundary = (x, t) -> true)

# Define problem-specific parameters
spatial_domain = (0.0, 1.0)
temporal_domain = (0.0, 1.0)
num_spatial_elements = 10
degree = 3

# Solver options
solver_options = KSSolverOptions(
    max_iterations = 1000,
    tolerance = 1e-6,
    adaptive = false,
    max_levels = 1,
    smoothness_threshold = 0.1
)

# Create a PDE problem instance
problem = KSPDEProblem(
    pde = simple_pde,
    boundary_conditions = boundary_conditions,
    domain = (spatial_domain, temporal_domain),
    tspan = temporal_domain
)

# Discretize and optimize using the JuMPIntegration module
model, solution = JuMPIntegration.discretize_and_optimize(
    problem,
    num_spatial_elements,
    temporal_domain,
    degree,
    solver_options
)

println("Optimized solution: ", solution)
