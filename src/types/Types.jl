module Types

using SparseArrays, DocStringExtensions

# Export Utilities
include("utils.jl")
export check_empty, check_positive, check_non_negative, check_type, check_length
export check_range, check_non_empty_elements, check_is_nothing, check_fields

# Export Core Types and Functions
include("core.jl")
export AbstractSystem, AbstractMesh, AbstractProblem, AbstractElement, AbstractTreeNode
export AbstractConstraint, AbstractBoundaryCondition, AbstractObjectiveFunction
export AbstractCostFunction, AbstractTensorProductMask, AbstractLocationMatrix
export AbstractRefinementStrategy, AbstractErrorEstimator, AbstractIntegrationKernel
export HierarchicalBasisFunction, DecayRate, TensorProductMask, LocationMatrix
export AbstractBasisFunction, AbstractHeiricialBasisFunction

# Export Mesh Types
include("meshes.jl")
export Connectivity, Node, BasisFunction, Element, HierarchicalElement, Mesh, HierarchicalGridLayer, Connectivity
export HRefinement, PRefinement, HPRefinement

# Export Problem Types
include("problems.jl")
export Equation, Domain, DirichletCondition, NeumannCondition, RobinCondition, BoundaryCondition, Weights, TimeSpan, DifferentialVars

# Export Solver Types
include("solvers.jl")
export Solver, TimeSteppingSolver, OptimalControlSolver, SolverLog, SolverMonitor, ParallelOptions, DerivativeRecovery

# Export Error Estimator Types
include("error_estimation.jl")
export ResidualErrorEstimator, GoalOrientedErrorEstimator

end # module Types
