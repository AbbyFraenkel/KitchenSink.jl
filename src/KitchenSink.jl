module KitchenSink

using LinearAlgebra, SparseArrays, FastGaussQuadrature, Plots, ForwardDiff
using AlgebraicMultigrid, IterativeSolvers

include("KSTypes/KSTypes.jl")
include("SpectralMethods/SpectralMethods.jl")
include("AdaptiveMethods/AdaptiveMethods.jl")
include("CoordinateSystems/CoordinateSystems.jl")
include("CommonMethods/CommonMethods.jl")
include("MultiLevelMethods/MultiLevelMethods.jl")
include("Preprocessing/Preprocessing.jl")
include("LinearSolvers/LinearSolvers.jl")
include("ProblemTypes/ProblemTypes.jl")


include("IntegridOperators/IntegridOperators.jl")
include("Preconditioners/Preconditioners.jl")

include("Optimization/Optimization.jl")
include("TimeStepping/TimeStepping.jl")
include("Visualization/Visualization.jl")



end
