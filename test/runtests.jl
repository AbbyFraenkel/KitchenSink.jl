using Test, SafeTestsets
# begin

# Passing
@safetestset "KSTypes" begin
	include("KSTypes/KSTypes_tests.jl")
end
# # #Passing
@safetestset "Coordinate Systems" begin
	include("CoordinateSystems/CoordinateSystems_test.jl")
end

# Passing
@safetestset "Spectral Methods" begin
	include("SpectralMethods/SpectralMethods_tests.jl")
end

# @safetestset "Common Methods" begin
# 	include("CommonMethods/CommonMethods_test.jl")
# end

# @safetestset "Error Estimation" begin
# 	include("ErrorEstimation/ErrorEstimation_tests.jl")
# end
# # # end

# @safetestset "Preprocessing" begin
# 	include("Preprocessing/Preprocessing_test.jl")
# end

# @safetestset "Preconditioners" begin
# 	include("Preconditioners/Preconditioners_tests.jl")
# end

# @safetestset "Adaptive Methods" begin
# 	include("AdaptiveMethods/AdaptiveMethods_tests.jl")
# end

# @safetestset "Intergrid Operators" begin
# 	include("IntergridOperators/IntergridOperators_tests.jl")
# end

# @safetestset "MultiLevel Methods" begin
# 	include("MultiLevelMethods/MultiLevelMethods_tests.jl")
# end

# @safetestset "Linear Solvers" begin
# 	include("LinearSolvers/LinearSolvers_tests.jl")
# end

# @safetestset "Problem Types" begin
# 	include("ProblemTypes/ProblemTypes_tests.jl")
# end
# # Todo
# @safetestset "Time Stepping" begin
# 	include("TimeStepping/TimeStepping_tests.jl")
# end
# # Todo
# @safetestset "Optimization" begin
# 	include("Optimization/Optimization_tests.jl")
# end
# # Todo
# @safetestset "Boundary Conditions" begin
# 	include("BoundaryConditions/BoundaryConditions_tests.jl")
# end
# # Todo
# @safetestset "Domain Decomposition" begin
# 	include("DomainDecomposition/DomainDecomposition_tests.jl")
# end
# # Todo
# @safetestset "Visualization" begin
# 	include("Visualization/Visualization_tests.jl")
# end
