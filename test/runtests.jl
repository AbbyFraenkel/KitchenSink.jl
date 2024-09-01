using Test, SafeTestsets
# begin
# begin

# Passing
# Passing
@safetestset "KSTypes" begin

	include("KSTypes/KSTypes_tests.jl")
end
# # #Passing
# # #Passing
@safetestset "Coordinate Systems" begin
	include("CoordinateSystems/CoordinateSystems_test.jl")
end

# Passing
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
# # end

# @safetestset "Preprocessing" begin
# 	include("Preprocessing/Preprocessing_test.jl")
# end
# @safetestset "Common Methods" begin
# 	include("CommonMethods/CommonMethods_test.jl")
# end

# @safetestset "Error Estimation" begin
# 	include("ErrorEstimation/ErrorEstimation_tests.jl")
# end
# # end

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


# @safetestset "Problem Types" begin
#     include("ProblemTypes/ProblemTypes.jl")
# end



# @safetestset "Optimization" begin
#     include("Optimization/Optimization.jl")
# end

# @safetestset "Time Stepping" begin
#     include("TimeStepping/TimeStepping.jl")
# end

# @safetestset "Visualization" begin
#     include("Visualization/Visualization.jl")
# end
