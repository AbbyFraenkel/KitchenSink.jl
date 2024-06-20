module KitchenSink

# Include the main Types module
include("types/Types.jl")
using .Types

# Export all symbols from the Types module
export check_empty, check_positive, check_non_negative, check_type, check_length
export check_range, check_non_empty_elements, check_is_nothing, check_fields

export AbstractSystem, AbstractMesh, AbstractProblem, AbstractElement, AbstractTreeNode
export AbstractConstraint, AbstractBoundaryCondition, AbstractObjectiveFunction
export AbstractCostFunction, AbstractTensorProductMask, AbstractLocationMatrix
export AbstractRefinementStrategy, AbstractErrorEstimator, AbstractIntegrationKernel
export HierarchicalBasisFunction, DecayRate, TensorProductMask, LocationMatrix
export AbstractBasisFunction, AbstractHeiricialBasisFunction

export Connectivity, Node, BasisFunction, Element, HierarchicalElement, Mesh, HierarchicalGridLayer, Connectivity
export HRefinement, PRefinement, HPRefinement

export Equation, Domain, DirichletCondition, NeumannCondition, RobinCondition, BoundaryCondition, Weights, TimeSpan, DifferentialVars

export Solver, TimeSteppingSolver, OptimalControlSolver, SolverLog, SolverMonitor, ParallelOptions, DerivativeRecovery

export ResidualErrorEstimator, GoalOrientedErrorEstimator

end # module KitchenSink
