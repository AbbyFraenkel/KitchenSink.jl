module KitchenSink

using LinearAlgebra, SparseArrays, StaticArrays, JuMP, Ipopt, IterativeSolvers
using Plots, ColorSchemes, LaTeXStrings, Documenter, DocStringExtensions, AlgebraicMultigrid
using FastGaussQuadrature, IncompleteLU

include("KSTypes/KSTypes.jl")
include("CoordinateSystems/CoordinateSystems.jl")
include("SpectralMethods/SpectralMethods.jl")

# include("CommonMethods/CommonMethods.jl")
# include("IntergridOperators/IntergridOperators.jl")
# include("ErrorEstimation/ErrorEstimation.jl")

# include("Preprocessing/Preprocessing.jl")
# include("AdaptiveMethods/AdaptiveMethods.jl")
# include("MultiLevelMethods/MultiLevelMethods.jl")

# include("Preconditioners/Preconditioners.jl")

# include("LinearSolvers/LinearSolvers.jl")

# include("ProblemTypes/ProblemTypes.jl")

# include("TimeStepping/TimeStepping.jl")

# include("Optimization/Optimization.jl")

# include("BoundaryConditions/BoundaryConditions.jl")

# include("DomainDecomposition/DomainDecomposition.jl")

# include("Visualization/Visualization.jl")

end # module KitchenSink
