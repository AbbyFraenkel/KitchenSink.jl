using Test, SafeTestsets



@testset "Types Module Tests" begin
    @time @safetestset "Core Types and Functions" begin include("types/core.jl") end
    @time @safetestset "Mesh Types" begin include("types/meshes.jl") end
    @time @safetestset "Problem Types" begin include("types/problems.jl") end
    @time @safetestset "Solver Types" begin include("types/solvers.jl") end
    @time @safetestset "Error Estimation Types" begin include("types/error_estimation.jl") end
    @time @safetestset "Utility Functions" begin include("types/utils.jl") end
end
