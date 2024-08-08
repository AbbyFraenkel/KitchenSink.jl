using SafeTestsets, Test
using .KitchenSink

@safetestset "KSTypes" begin
    include("KSTypes/KSTypes.jl")
end

@safetestset "Coordinate Systems" begin
    include("CoordinateSystems/CoordinateSystems.jl")
end

@safetestset "Spectral Methods" begin
    include("SpectralMethods/SpectralMethods.jl")
end


@safetestset "Preprocessing" begin
    include("Preprocessing/Preprocessing.jl")
end


@safetestset "Adaptive Methods" begin
    include("AdaptiveMethods/AdaptiveMethods.jl")
end



@safetestset "Multi Level Methods" begin
    include("MultiLevelMethods/MultiLevelMethods.jl")
end


@safetestset "Problem Types" begin
    include("ProblemTypes/ProblemTypes.jl")
end



@safetestset "Optimization" begin
    include("Optimization/Optimization.jl")
end

@safetestset "Time Stepping" begin
    include("TimeStepping/TimeStepping.jl")
end

@safetestset "Visualization" begin
    include("Visualization/Visualization.jl")
end
