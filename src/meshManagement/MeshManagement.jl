module MeshManagement

using LinearAlgebra, SparseArrays, IterativeSolvers

using .Types

# Export Solver Types
include("mesh_creation.jl")
export EnsureHigherOrderContinuity!, check_degrees_of_freedom, MeshAssembly

# Export Error Estimator Types

# Export preconditioning
include("preconditioning.jl")
export jacobi_preconditioner, ilu_preconditioner

# #xport Regularization
include("regularization.jl")
export CoordinateDescentRidge, CoordinateDescentLasso, CoordinateDescentElasticNet
end
