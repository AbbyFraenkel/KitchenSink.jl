using Test
using KitchenSink.Types

@testset "Problem Types" begin

    @testset "Domain" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws BoundsError Domain(Float64[][])
                @test_throws MethodError Domain(nothing)
                @test_throws MethodError Domain()
            end
            @testset "Correct Outputs" begin
                domain = Domain([[0.0, 1.0], [1.0, 2.0]])
                @test length(domain.coordinates) == 2
            end
        end
    end

    @testset "DirichletCondition" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError DirichletCondition(1.0, Float64[])
                @test_throws MethodError DirichletCondition(1.0, nothing)
                @test_throws MethodError DirichletCondition()
            end
            @testset "Correct Outputs" begin
                dc = DirichletCondition(1.0, [0.0, 1.0])
                @test dc.value == 1.0
            end
        end
    end

    @testset "NeumannCondition" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError NeumannCondition(1.0, Float64[])
                @test_throws MethodError NeumannCondition(1.0, nothing)
                @test_throws MethodError NeumannCondition()
            end
            @testset "Correct Outputs" begin
                nc = NeumannCondition(1.0, [0.0, 1.0])
                @test nc.flux == 1.0
            end
        end
    end

    @testset "RobinCondition" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError RobinCondition(1.0, 0.5, Float64[])
                @test_throws MethodError RobinCondition(1.0, 0.5, nothing)
                @test_throws MethodError RobinCondition()
            end
            @testset "Correct Outputs" begin
                rc = RobinCondition(1.0, 0.5, [0.0, 1.0])
                @test rc.alpha == 1.0
                @test rc.beta == 0.5
            end
        end
    end

    @testset "Weights" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError Weights(Float64[])
                @test_throws MethodError Weights(nothing)
                @test_throws MethodError Weights()
            end
            @testset "Correct Outputs" begin
                weights = Weights([0.1, 0.2])
                @test weights.values == [0.1, 0.2]
            end
        end
    end

    @testset "TimeSpan" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError TimeSpan(Float64[])
                @test_throws MethodError TimeSpan(nothing)
                @test_throws MethodError TimeSpan()
            end
            @testset "Correct Outputs" begin
                ts = TimeSpan([0.0, 1.0])
                @test ts.span == [0.0, 1.0]
            end
        end
    end

    @testset "DifferentialVars" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError DifferentialVars(Bool[])
                @test_throws MethodError DifferentialVars(nothing)
                @test_throws MethodError DifferentialVars()
            end
            @testset "Correct Outputs" begin
                dv = DifferentialVars([true, false])
                @test dv.vars == [true, false]
            end
        end
    end

end
