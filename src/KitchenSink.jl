module KitchenSink

using LinearAlgebra, SparseArrays, StaticArrays, JuMP, Ipopt, IterativeSolvers, Statistics
using Plots, ColorSchemes, LaTeXStrings, Documenter, DocStringExtensions, AlgebraicMultigrid
using FastGaussQuadrature, IncompleteLU, ForwardDiff, NaNMath, OrdinaryDiffEq
using Polyester, LinearOperators
using BSplineKit: BSplineBasis, BSplineOrder, KnotVector, evaluate
using BSplineKit

include("KSTypes/KSTypes.jl")
include("NumericUtilities/NumericUtilities.jl")
include("CacheManagement/CacheManagement.jl")
include("CoordinateSystems/CoordinateSystems.jl")
include("SpectralMethods/SpectralMethods.jl")
include("BoundaryConditions/BoundaryConditions.jl")
include("Transforms/Transforms.jl")
include("Preconditioners/Preconditioners.jl")
include("LinearSolvers/LinearSolvers.jl")
include("ProblemTypes/ProblemTypes.jl")
include("TimeStepping/TimeStepping.jl")
include("Optimization/Optimization.jl")
include("Preprocessing/Preprocessing.jl")
include("Visualization/Visualization.jl")

end # module KitchenSink
