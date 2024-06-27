using Test
using KitchenSink.Types

@testset "Error Estimation Types" begin

    @testset "ResidualErrorEstimator" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError ResidualErrorEstimator(Float64[])
                @test_throws MethodError ResidualErrorEstimator(nothing)
                @test_throws MethodError ResidualErrorEstimator()
            end
            @testset "Correct Outputs" begin
                ree = ResidualErrorEstimator([0.1, 0.2])
                @test ree.residuals == [0.1, 0.2]
            end
        end
    end

    @testset "GoalOrientedErrorEstimator" begin
        goal_functional(x) = x
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError GoalOrientedErrorEstimator(
                    goal_functional,
                    Float64[],
                )
                @test_throws MethodError GoalOrientedErrorEstimator(nothing, [0.1, 0.2])
                @test_throws MethodError GoalOrientedErrorEstimator(
                    goal_functional,
                    nothing,
                )
                @test_throws MethodError GoalOrientedErrorEstimator()
            end
            @testset "Correct Outputs" begin
                goe = GoalOrientedErrorEstimator(goal_functional, [0.1, 0.2])
                @test goe.goal_functional(1.0) == 1.0
                @test goe.error_indicators == [0.1, 0.2]
            end
        end
    end

end
