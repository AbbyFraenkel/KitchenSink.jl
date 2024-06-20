# test_adaptive_multigrid.jl

using Test
using LinearAlgebra
using SparseArrays
using Logging
using FastGaussQuadrature
using ..Types
using ..AdaptiveMultigrid

  # @testset "Test AdaptiveMultigrid Module" begin

    # @testset "Test create_grid_hierarchy" begin
      n_dims = 2
      levels = 3
      p = 3  # Polynomial degree
      mesh = create_grid_hierarchy(n_dims, levels, p)
      @test isa(mesh, MultiLayerMesh)
      @test length(mesh.layers) == levels
      @test mesh.ndims == ndims
    end

    # @testset "Test create_operators" begin
      ndims = 2
      levels = 3
      p = 3  # Polynomial degree
      mesh = create_grid_hierarchy(ndims, levels, p)
      operators = create_operators(mesh)
      @test length(operators) == levels
      @test isa(operators[1], SparseMatrixCSC)
    end

    @testset "Test apply_boundary_conditions!" begin
      A = spzeros(5, 5)
      b = ones(5)
      boundary_conditions = [(1, 0.0), (5, 0.0)]
      A, b = apply_boundary_conditions!(A, b, boundary_conditions)
      @test A[1, 1] == 1.0
      @test A[1, 2] == 0.0
      @test b[1] == 0.0
    end

    @testset "Test smooth" begin
      A = spdiagm(0 => fill(2.0, 5), -1 => fill(-1.0, 4), 1 => fill(-1.0, 4))
      b = ones(5)
      x = zeros(5)
      x = smooth(A, b, x, 10)
      @test norm(A * x - b) < 1e-6
    end

    @testset "Test gauss_seidel" begin
      A = spdiagm(0 => fill(2.0, 5), -1 => fill(-1.0, 4), 1 => fill(-1.0, 4))
      b = ones(5)
      x = zeros(5)
      x = gauss_seidel(A, b, x, 10)
      @test norm(A * x - b) < 1e-6
    end

    @testset "Test sor" begin
      A = spdiagm(0 => fill(2.0, 5), -1 => fill(-1.0, 4), 1 => fill(-1.0, 4))
      b = ones(5)
      x = zeros(5)
      omega = 1.5
      x = sor(A, b, x, omega, 10)
      @test norm(A * x - b) < 1e-6
    end

    @testset "Test restrict" begin
      residual_fine = rand(5, 5)
      residual_coarse = restrict(residual_fine)
      @test size(residual_coarse) == (3, 3)
    end

    @testset "Test prolong" begin
      correction_coarse = rand(3, 3)
      correction_fine = prolong(correction_coarse)
      @test size(correction_fine) == (5, 5)
    end

    @testset "Test v_cycle" begin
      ndims = 2
      levels = 3
      p = 3  # Polynomial degree
      mesh = create_grid_hierarchy(ndims, levels, p)
      operators = create_operators(mesh)
      b = ones(length(mesh.layers[1].elements))
      x = zeros(length(mesh.layers[1].elements))
      logger = ConsoleLogger()
      x = v_cycle(1, levels, mesh, operators, b, x, 3, logger)
      @test norm(operators[1] * x - b) < 1e-6
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
