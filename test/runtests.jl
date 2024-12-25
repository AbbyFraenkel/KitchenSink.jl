using Test, SafeTestsets, Coverage

@testset "KitchenSink" begin

    # Passing
    @safetestset "KSTypes" begin
        include("KSTypes/KSTypes_tests.jl")
    end

    # Passing
    @safetestset "NumericUtilities" begin
        include("NumericUtilities/NumericUtilities_tests.jl")
    end

    # Passing
    @safetestset "CacheManagement" begin
        include("CacheManagement/CacheManagement_tests.jl")
    end

    # Passing
    @safetestset "Coordinate Systems" begin
        include("CoordinateSystems/CoordinateSystems_test.jl")
    end

    # Passing
    @safetestset "Spectral Methods" begin
        include("SpectralMethods/SpectralMethods_tests.jl")
    end

    # Passing
    @safetestset "BoundaryConditions" begin
        include("BoundaryConditions/BoundaryConditions_tests.jl")
    end

    # Passing
    @safetestset "Transforms" begin
        include("Transforms/Transforms_tests.jl")
    end

    # Passing
    @safetestset "Preconditioners" begin
        include("Preconditioners/Preconditioners_tests.jl")
    end

    #  Passing
    @safetestset "Linear Solvers" begin
        include("LinearSolvers/LinearSolvers_tests.jl")
    end

    # Not Passing
    @safetestset "Problem Types" begin
        include("ProblemTypes/ProblemTypes_tests.jl")
    end

    # # Not Passing
    # @safetestset "Time Stepping" begin
    #     include("TimeStepping/TimeStepping_tests.jl")
    # end
    # # Not Passing
    # @safetestset "Optimization" begin
    #     include("Optimization/Optimization_tests.jl")
    # end
    # #  Not Passing
    # @safetestset "Preprocessing" begin
    #     include("Preprocessing/Preprocessing_test.jl")
    # end

    # # Passing
    # @safetestset "Visualization" begin
    #     include("Visualization/Visualization_tests.jl")
    # end
end
