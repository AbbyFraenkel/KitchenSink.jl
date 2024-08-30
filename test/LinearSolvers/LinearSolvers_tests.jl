using Test
using LinearAlgebra, SparseArrays
using KitchenSink.LinearSolvers

@testset "LinearSolvers" begin
    # Helper functions to generate test matrices
    function create_spd_matrix(n::Int)
        A = randn(n, n)
        return A' * A + n * I
    end

    function create_sparse_matrix(n::Int, density::Float64)
        return sprandn(n, n, density)
    end

    @testset "KSDirectSolver" begin
        @testset "LU method" begin
            A = [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 3.0]
            b = [1.0, 2.0, 3.0]
            solver = KSDirectSolver(:lu)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-10)
        end

        @testset "QR method" begin
            A = [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 3.0]
            b = [1.0, 2.0, 3.0]
            solver = KSDirectSolver(:qr)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-10)
        end

        @testset "Invalid method" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 1.0]
            solver = KSDirectSolver(:invalid)
            @test_throws ArgumentError solve_linear_system(A, b, solver)
        end
    end

    @testset "KSIterativeSolver" begin
        @testset "CG method" begin
            A = create_spd_matrix(100)
            b = randn(100)
            solver = KSIterativeSolver(:cg, 1000, 1e-8, nothing)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-6)
        end

        @testset "GMRES method" begin
            A = randn(100, 100)
            b = randn(100)
            solver = KSIterativeSolver(:gmres, 1000, 1e-8, nothing)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-6)
        end

        @testset "With preconditioner" begin
            A = create_spd_matrix(100)
            b = randn(100)
            precond = Preconditioners.jacobi_preconditioner(A)
            solver = KSIterativeSolver(:cg, 1000, 1e-8, precond)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-6)
        end

        @testset "Invalid method" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 1.0]
            solver = KSIterativeSolver(:invalid, 100, 1e-6, nothing)
            @test_throws ArgumentError solve_linear_system(A, b, solver)
        end
    end

    @testset "KSAMGSolver" begin
        @testset "Jacobi smoother" begin
            A = create_sparse_matrix(100, 0.1)
            A = A + A' + 100I  # Make diagonally dominant
            b = randn(100)
            solver = KSAMGSolver(1000, 1e-8, :jacobi)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-6)
        end

        @testset "Gauss-Seidel smoother" begin
            A = create_sparse_matrix(100, 0.1)
            A = A + A' + 100I  # Make diagonally dominant
            b = randn(100)
            solver = KSAMGSolver(1000, 1e-8, :gauss_seidel)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-6)
        end

        @testset "SOR smoother" begin
            A = create_sparse_matrix(100, 0.1)
            A = A + A' + 100I  # Make diagonally dominant
            b = randn(100)
            solver = KSAMGSolver(1000, 1e-8, :sor)
            x = solve_linear_system(A, b, solver)
            @test isapprox(A * x, b, rtol=1e-6)
        end

        @testset "Invalid smoother" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 1.0]
            solver = KSAMGSolver(100, 1e-6, :invalid)
            @test_throws ArgumentError solve_linear_system(A, b, solver)
        end
    end

    @testset "select_optimal_solver" begin
        @testset "Small dense matrix" begin
            A = randn(50, 50)
            solver = select_optimal_solver(A, :general)
            @test solver isa KSDirectSolver
            @test solver.method == :lu
        end

        @testset "Large sparse SPD matrix" begin
            A = create_spd_matrix(1000)
            A = sparse(A)
            solver = select_optimal_solver(A, :general)
            @test solver isa KSIterativeSolver
            @test solver.method == :cg
        end

        @testset "Elliptic problem" begin
            A = create_sparse_matrix(1000, 0.001)
            A = A + A' + 1000I  # Make diagonally dominant
            solver = select_optimal_solver(A, :elliptic)
            @test solver isa KSAMGSolver
        end

        @testset "General large problem" begin
            A = create_sparse_matrix(1000, 0.01)
            solver = select_optimal_solver(A, :general)
            @test solver isa KSIterativeSolver
            @test solver.method == :gmres
        end

        @testset "Invalid problem type" begin
            A = [1.0 0.0; 0.0 1.0]
            @test_throws ArgumentError select_optimal_solver(A, :invalid)
        end
    end

    @testset "Error handling" begin
        @testset "Non-square matrix" begin
            A = [1.0 2.0 3.0; 4.0 5.0 6.0]
            b = [1.0, 2.0]
            solver = KSDirectSolver(:lu)
            @test_throws DimensionMismatch solve_linear_system(A, b, solver)
        end

        @testset "Dimension mismatch" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 2.0, 3.0]
            solver = KSDirectSolver(:lu)
            @test_throws DimensionMismatch solve_linear_system(A, b, solver)
        end

        @testset "Invalid solver type" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 1.0]
            solver = "invalid"
            @test_throws ArgumentError solve_linear_system(A, b, solver)
        end
    end
end
