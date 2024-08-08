using Test, LinearAlgebra, SparseArrays, Logging, BenchmarkTools, IterativeSolvers
using .KSTypes, .LinearSolvers, .Preprocessing

@testset "LinearSolvers" begin
    @testset "solve_linear_system" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        x_expected = A \ b

        @testset "Direct Solver" begin
            solver = KSDirectSolver(:lu, 1e-5)
            x = solve_linear_system(A, b, solver)
            @test x ≈ x_expected
        end

        @testset "Iterative Solver" begin
            solver = KSIterativeSolver(:cg, 1000, 1e-8, nothing)
            x = solve_linear_system(A, b, solver)
            @test x ≈ x_expected
        end

        @testset "AMG Solver" begin
            solver = KSAMGSolver(1000, 1e-8, :gauss_seidel)
            x = solve_linear_system(A, b, solver)
            @test x ≈ x_expected
        end

        @testset "Error Handling" begin
            @test_throws ArgumentError solve_linear_system([1.0 2.0; 3.0 4.0; 5.0 6.0], [1.0, 2.0], KSDirectSolver(:lu, 1e-8))
            @test_throws ArgumentError solve_linear_system(A, [1.0, 2.0, 3.0], KSDirectSolver(:lu, 1e-8))
        end
    end

    # @testset "select_optimal_solver" begin
    # @testset "Small Dense Matrix" begin
    #         A = rand(100, 100)
    #         solver = select_optimal_solver(A, :general)
    #         @test solver isa KSDirectSolver
    #         @test solver.method == :lu
    #     end

    #     @testset "Large Sparse Matrix" begin
    #         A = sprand(1000, 1000, 0.01)
    #         solver = select_optimal_solver(A, :general)
    #         @test solver isa KSIterativeSolver
    #         @test solver.method == :gmres
    #     end

    #     @testset "Symmetric Positive Definite Matrix" begin
    #         A = sprand(1000, 1000, 0.01)
    #         A = A * A' + I  # Make SPD
    #         solver = select_optimal_solver(A, :spd)
    #         @test solver isa KSIterativeSolver
    #         @test solver.method == :cg
    #     end

    #     @testset "Elliptic Problem" begin
    #         A = sprand(1000, 1000, 0.01)
    #         solver = select_optimal_solver(A, :elliptic)
    #         @test solver isa KSAMGSolver
    #     end
    # end

    @testset "solve_with_logging" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        solver = KSDirectSolver(:lu, 1e-8)

        # Capture logging output
        log_output = IOBuffer()
        with_logger(SimpleLogger(log_output)) do
            x = solve_with_logging(A, b, solver)
            @test x ≈ A \ b
        end

        log_str = String(take!(log_output))
        @test occursin("Starting linear system solve", log_str)
        @test occursin("Linear system solved", log_str)
        @test occursin("time", log_str)
        @test occursin("residual", log_str)
    end



    include("direct_solvers.jl")
    include("iterative_solvers.jl")
    # include("amg_solvers.jl")
end
