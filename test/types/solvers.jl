using Test
using KitchenSink.Types

@testset "Solver Types" begin

    @testset "TimeSteppingSolver" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError TimeSteppingSolver(-0.1, 10)
                @test_throws ArgumentError TimeSteppingSolver(0.1, -10)
                @test_throws MethodError TimeSteppingSolver(0.1, nothing)
                @test_throws MethodError TimeSteppingSolver(nothing, 10)
                @test_throws MethodError TimeSteppingSolver()
            end
            @testset "Correct Outputs" begin
                tss = TimeSteppingSolver(0.1, 10)
                @test tss.dt == 0.1
                @test tss.max_steps == 10
            end
        end
    end

    @testset "OptimalControlSolver" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError OptimalControlSolver(-0.1, 10)
                @test_throws ArgumentError OptimalControlSolver(0.1, -10)
                @test_throws MethodError OptimalControlSolver(0.1, nothing)
                @test_throws MethodError OptimalControlSolver(nothing, 10)
                @test_throws MethodError OptimalControlSolver()
            end
            @testset "Correct Outputs" begin
                ocs = OptimalControlSolver(0.1, 10)
                @test ocs.tolerance == 0.1
                @test ocs.max_iterations == 10
            end
        end
    end

    @testset "SolverLog" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError SolverLog(-1, 0.0, 0.0)
                @test_throws ArgumentError SolverLog(1, -0.1, 0.0)
                @test_throws ArgumentError SolverLog(1, 0.0, -0.1)
                @test_throws MethodError SolverLog(1, 0.0, nothing)
                @test_throws MethodError SolverLog(1, nothing, 0.0)
                @test_throws MethodError SolverLog(nothing, 0.0, 0.0)
                @test_throws MethodError SolverLog()
            end
            @testset "Correct Outputs" begin
                log = SolverLog(1, 0.1, 0.1)
                @test log.iteration == 1
                @test log.residual == 0.1
                @test log.error_estimate == 0.1
            end
        end
    end

    @testset "SolverMonitor" begin
        @testset "Construction" begin
            log = SolverLog(1, 0.1, 0.1)
            @testset "Error Handling" begin
                @test_throws ArgumentError SolverMonitor(SolverLog[])
                @test_throws MethodError SolverMonitor(nothing)
                @test_throws MethodError SolverMonitor()
            end
            @testset "Correct Outputs" begin
                monitor = SolverMonitor([log])
                @test length(monitor.logs) == 1
            end
        end
    end

    @testset "ParallelOptions" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws MethodError ParallelOptions(true, -1, :mpi)
                @test_throws MethodError ParallelOptions(true, 4, nothing)
                @test_throws MethodError ParallelOptions(true, nothing, :mpi)
                @test_throws MethodError ParallelOptions(nothing, 4, :mpi)
                @test_throws MethodError ParallelOptions()
            end
            @testset "Correct Outputs" begin
                options = ParallelOptions(true, 4.0, :mpi)
                @test options.use_parallel == true
                @test options.num_threads == 4
                @test options.solver_type == :mpi
            end
        end
    end

    @testset "DerivativeRecovery" begin
        recovery_method(x) = x
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError DerivativeRecovery(recovery_method, Float64[])
                @test_throws MethodError DerivativeRecovery(nothing, [0.1, 0.2])
                @test_throws MethodError DerivativeRecovery(recovery_method, nothing)
                @test_throws MethodError DerivativeRecovery()
            end
            @testset "Correct Outputs" begin
                dr = DerivativeRecovery(recovery_method, [0.1, 0.2])
                @test dr.recovery_method(1.0) == 1.0
                @test dr.recovered_derivatives == [0.1, 0.2]
            end
        end
    end

end
