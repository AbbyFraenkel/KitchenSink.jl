using Test
using LinearAlgebra, SparseArrays, LinearOperators, IterativeSolvers
using IncompleteLU, Base.Threads, Statistics
import AlgebraicMultigrid: ruge_stuben, aspreconditioner, GaussSeidel

using KitchenSink
using KitchenSink.LinearSolvers
using KitchenSink.NumericUtilities
using KitchenSink.Preconditioners
using KitchenSink.KSTypes

include("../test_utils.jl")

@testset "LinearSolvers" begin
    @testset "Basic Matrix Types" begin
        n = 50
        A_spd = create_spd_matrix(n)
        A_dd = make_diag_dominant_matrix(n)
        A_ill = gen_ill_conditioned(n)
        A_sparse = create_sparse_matrix(n)

        b = randn(n)

        for (name, A) in [
            "SPD" => A_spd,
            "Diagonally Dominant" => A_dd,
            "Ill-conditioned" => A_ill,
            "Sparse" => A_sparse,
        ]
            @testset "$name matrix" begin
                solver = KSIterativeSolver(:cg, 1000, 1e-6)
                x = solve_linear_system(A, b, solver)
                rel_residual = norm(A * x - b) / norm(b)

                # Adjust tolerance for ill-conditioned matrices
                tol = name == "Ill-conditioned" ? 1e-4 : 1e-6
                @test rel_residual < tol
            end
        end
    end

    @testset "Matrix Suite Tests" begin
        n = 50
        matrix_suite = create_matrix_suite(n)

        for (name, (A, props)) in matrix_suite
            b = randn(n)

            @testset "$name matrix" begin
                solver = KSIterativeSolver(:cg, 1000, 1e-6)
                x = solve_linear_system(A, b, solver)
                rel_residual = norm(A * x - b) / norm(b)
                @test rel_residual < 1e-6
            end
        end
    end

    @testset "Preconditioner Integration" begin
        n = 50

        @testset "Matrix Type: $name" for (name, A) in [
            "SPD" => create_spd_matrix(n),
            "Diagonally Dominant" => make_diag_dominant_matrix(n),
            "Sparse" => create_sparse_matrix(n),
        ]
            b = randn(n)
            props = compute_base_matrix_properties(A)

            x_direct = solve_linear_system(A, b, KSDirectSolver(:lu))
            @test norm(A * x_direct - b) / norm(b) < 1e-6

            P = select_preconditioner(A)
            x_iter = solve_iterative(A, b, :cg, 1000, 1e-6, P)
            @test norm(A * x_iter - b) / norm(b) < 1e-6
        end
    end

    @testset "Cache Management" begin
        n = 4
        num_threads = min(4, Threads.nthreads())

        matrices = [create_spd_matrix(n) for _ in 1:num_threads]
        vectors = [randn(n) for _ in 1:num_threads]

        clear_solver_cache!()

        solutions = Vector{Vector{Float64}}(undef, num_threads)

        Threads.@threads for i in 1:num_threads
            solutions[i] = solve_linear_system(matrices[i], vectors[i], KSDirectSolver(:lu))
        end

        for i in 1:num_threads
            rel_residual = norm(matrices[i] * solutions[i] - vectors[i]) / norm(vectors[i])
            @test rel_residual < 1e-6
        end

        Threads.@threads for i in 1:num_threads
            invalidate_cache_for_matrix!(matrices[i])
        end
    end

    @testset "Edge Cases" begin
        n = 10

        @testset "Invalid Inputs" begin
            A = rand(n, n)
            b = rand(n)
            @test_throws ArgumentError solve_linear_system(A, b, KSDirectSolver(:invalid))
        end

        @testset "Numerical Edge Cases" begin
            A = rand(n, n)
            b = rand(n)
            solver = KSIterativeSolver(:cg, 1000, 1e-6)
            x = solve_linear_system(A, b, solver)
            rel_residual = norm(A * x - b) / norm(b)
            @test rel_residual < 1e-6
        end
    end
end
