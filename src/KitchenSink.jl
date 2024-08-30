module KitchenSink

using LinearAlgebra, SparseArrays, StaticArrays, JuMP, Ipopt, IterativeSolvers
using Plots, ColorSchemes, LaTeXStrings, Documenter, DocStringExtensions, AlgebraicMultigrid
using FastGaussQuadrature, IncompleteLU


include("KSTypes/KSTypes.jl")
include("CoordinateSystems/CoordinateSystems.jl")
include("SpectralMethods/SpectralMethods.jl")

include("CommonMethods/CommonMethods.jl")
include("IntergridOperators/IntergridOperators.jl")
include("ErrorEstimation/ErrorEstimation.jl")

include("Preprocessing/Preprocessing.jl")
include("AdaptiveMethods/AdaptiveMethods.jl")
include("MultiLevelMethods/MultiLevelMethods.jl")
# # Need to update
include("Preconditioners/Preconditioners.jl")
# # Need to update
include("LinearSolvers/LinearSolvers.jl")

include("ProblemTypes/ProblemTypes.jl")
# # Need to update
include("TimeStepping/TimeStepping.jl")
# # Need to update
include("Optimization/Optimization.jl")

include("BoundaryConditions/BoundaryConditions.jl")

include("DomainDecomposition/DomainDecomposition.jl")
# Need to update
include("Visualization/Visualization.jl")

end # module KitchenSink
