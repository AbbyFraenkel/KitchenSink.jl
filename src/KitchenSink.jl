module KitchenSink

include("types/Types.jl")
include("adaptiveMultiGrid/AdaptiveMultigrid.jl")
include("visualization/Visualization.jl")
include("systemFormulation/SystemFormulation.jl")
include("refinement/Refinement.jl")
include("meshManagement/MeshManagement.jl")
include("parallelMethods/ParallelMethods.jl")
include("optimalControlProblems/OptimalControlProblems.jl")
include("meshManagement/MeshManagement.jl")
include("discretization/Discretization.jl")
include("timeStepping/TimeStepping.jl")
include("boundaryConditions/BoundaryConditions.jl")
include("elementOperations/ElementOperations.jl")

end
