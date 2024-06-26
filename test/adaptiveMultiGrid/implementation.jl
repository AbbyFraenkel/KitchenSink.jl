# test_adaptive_multigrid.jl

using Test, LinearAlgebra, SparseArrays, Logging, FastGaussQuadrature
using ..Types
using ..AdaptiveMultigrid

@testset "Test AdaptiveMultigrid Module" begin

    @testset "Test create_grid_hierarchy" begin
        n_dims = 2
        levels = 3
        p = 3  # Polynomial degree
        mesh = create_grid_hierarchy(n_dims, levels, p)
        @test isa(mesh, MultiLayerMesh)
        @test length(mesh.layers) == levels
        @test mesh.ndims == ndims
    end

    @testset "Test create_operators" begin
        ndims = 2
        levels = 3
        p = 3  # Polynomial degree
        mesh = create_grid_hierarchy(ndims, levels, p)
        operators = create_operators(mesh)
        @test length(operators) == levels
        @test isa(operators[1], SparseMatrixCSC)
    end

    @testset "Test adaptive_multigrid_solver" begin
        struct SimpleProblem <: AbstractProblem
            domain::Vector{Vector{Float64}}
        end

        struct SimpleSolver <: Solver
            max_iterations::Int
            tolerance::Float64
        end

        problem = SimpleProblem([[0.0, 1.0], [0.0, 1.0]])
        solver = SimpleSolver(100, 1e-6)
        levels = 3
        error_threshold = 0.1
        p = 3  # Polynomial degree

        solution = adaptive_multigrid_solver(problem, solver, levels, error_threshold, p)
        @test norm(solution) > 0
    end

end
