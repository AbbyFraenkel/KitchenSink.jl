module ParallelMethods


using LinearAlgebra, SparseArrays, IterativeSolvers, Base.threads

# Export mesh generation
include("parallel_regularization.jl")
export ParallelMeshAssembly, ParallelCoordinateDescentRidge, ParallelCoordinateDescentLasso
export ParallelCoordinateDescentElasticNet

# Export mesh generation
include("parallel_mesh.jl")
export Parallel_assemble_system, Parallel_compute_error_estimates
end
