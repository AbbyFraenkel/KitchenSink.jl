using Test
using LinearAlgebra, StaticArrays
using JuMP
using Ipopt
using KitchenSink.KSTypes
using KitchenSink.ProblemTypes
using KitchenSink.Optimization
using KitchenSink.CacheManagement, KitchenSink.NumericUtilities

include("../test_utils_with_deps.jl")

@testset "Optimization" begin
	@testset "solve_optimal_control_problem" begin
		for dim in TEST_DIMENSIONS
			problem = create_test_optimal_control_problem()
			mesh, _ = create_test_mesh(dim)

			A = create_spd_matrix(problem.num_vars)
			b = randn(problem.num_vars)

			solution = Optimization.solve_optimal_control_problem(A, b, problem, mesh)

			@test size(solution.state, 1) == problem.num_vars
			@test size(solution.state, 2) == problem.num_time_steps
			@test size(solution.control, 1) == problem.num_controls
			@test size(solution.control, 2) == problem.num_time_steps - 1
			@test solution.state[:, 1] ≈ problem.initial_state
			@test solution.objective >= 0
		end
	end

	@testset "create_jump_model" begin
		for dim in TEST_DIMENSIONS
			problem = create_test_optimal_control_problem()
			mesh, _ = create_test_mesh(dim)

			A = create_spd_matrix(problem.num_vars)
			b = randn(problem.num_vars)

			model = Optimization.create_jump_model(A, b, problem, mesh)

			@test model isa JuMP.Model
			@test num_variables(model) ==
				problem.num_vars * problem.num_time_steps +
				  problem.num_controls * (problem.num_time_steps - 1)
			@test num_constraints(model) ==
				problem.num_vars * (problem.num_time_steps - 1) + problem.num_vars  # dynamics constraints + initial conditions
		end
	end

	@testset "extract_solution" begin
		problem = create_test_optimal_control_problem()

		# Create a mock JuMP model and optimize it
		model = Model(Ipopt.Optimizer)
		@variable(model, x[1:(problem.num_vars), 1:(problem.num_time_steps)])
		@variable(model, u[1:(problem.num_controls), 1:(problem.num_time_steps - 1)])
		@constraint(model, x[:, 1] .== problem.initial_state)
		@objective(model,
			Min,
			sum(x[i, j]^2
				for i in 1:(problem.num_vars), j in 1:(problem.num_time_steps)) + sum(
				u[i, j]^2
				for i in 1:(problem.num_controls), j in 1:(problem.num_time_steps - 1)
			))
		optimize!(model)

		solution = Optimization.extract_solution(model, problem)

		@test size(solution.state) == (problem.num_vars, problem.num_time_steps)
		@test size(solution.control) == (problem.num_controls, problem.num_time_steps - 1)
		@test solution.state[:, 1] ≈ problem.initial_state
		@test solution.objective >= 0
	end

	@testset "discretize_and_optimize" begin
		for dim in TEST_DIMENSIONS
			problem = create_test_problem(dim, KSDirichletBC)
			num_elements = ntuple(_ -> 10, dim)
			tspan = (0.0, 1.0)
			degree = 3
			solver_options = create_test_solver_options()

			model, solution = Optimization.discretize_and_optimize(
				problem,
				num_elements,
				tspan,
				degree,
				solver_options,
			)

			@test model isa JuMP.Model
			@test size(solution, 2) == solver_options.num_steps + 1
			@test size(solution, 1) == prod(num_elements .+ 1)
		end
	end

	@testset "solve_pde_constrained_optimization" begin
		for dim in TEST_DIMENSIONS
			pde_problem = create_test_problem(dim, KSDirichletBC)
			objective_function = (u, c) -> sum(u .^ 2) + sum(c .^ 2)
			control_bounds = [(-1.0, 1.0) for _ in 1:dim]
			mesh, _ = create_test_mesh(dim)

			solution = Optimization.solve_pde_constrained_optimization(
				pde_problem,
				objective_function,
				control_bounds,
				mesh,
			)

			@test length(solution.state) > 0
			@test length(solution.control) == length(control_bounds)
			@test all(
				all(control_bounds[i][1] .<= solution.control[i] .<= control_bounds[i][2])
				for i in eachindex(control_bounds)
			)
			@test solution.objective >= 0
		end
	end

	@testset "Error handling" begin
		@testset "Invalid control bounds" begin
			problem = create_test_optimal_control_problem(; control_bounds = [(1.0, -1.0)])
			mesh, _ = create_test_mesh(1)
			A = create_spd_matrix(problem.num_vars)
			b = randn(problem.num_vars)

			@test_throws ArgumentError Optimization.solve_optimal_control_problem(
				A,
				b,
				problem,
				mesh,
			)
		end

		@testset "Invalid time span" begin
			problem = create_test_optimal_control_problem(; time_span = (1.0, 0.0))
			mesh, _ = create_test_mesh(1)
			A = create_spd_matrix(problem.num_vars)
			b = randn(problem.num_vars)

			@test_throws ArgumentError Optimization.solve_optimal_control_problem(
				A,
				b,
				problem,
				mesh,
			)
		end

		@testset "Inconsistent dimensions" begin
			problem = create_test_optimal_control_problem()
			mesh, _ = create_test_mesh(1)
			A = create_spd_matrix(problem.num_vars + 1)  # Inconsistent dimension
			b = randn(problem.num_vars)

			@test_throws DimensionMismatch Optimization.solve_optimal_control_problem(
				A,
				b,
				problem,
				mesh,
			)
		end
	end

	@testset "Performance" begin
		@testset "Large-scale problem" begin
			N = 1000
			problem = create_test_optimal_control_problem(;
				num_vars = N,
				num_controls = 1,
				num_time_steps = 100,
			)
			mesh, _ = create_test_mesh(1, (N,))
			A = create_sparse_matrix(N, 0.01)
			b = randn(N)

			time_taken = @elapsed solution = Optimization.solve_optimal_control_problem(
				A, b, problem, mesh)

			@test time_taken < 60.0  # Adjust this threshold as needed
			@test size(solution.state, 1) == N
			@test size(solution.control, 1) == 1
		end
	end

	@testset "Nonlinear optimization" begin
		problem = create_test_optimal_control_problem(;
			state_equations = [x -> [x[2], -sin(x[1]) + u[1]] for u in 1:1],
			cost_functions = [(x, u) -> x[1]^2 + x[2]^2 + u[1]^2],
			terminal_cost = x -> x[1]^2 + x[2]^2,
			initial_state = [π, 0.0],
			time_span = (0.0, 10.0),
			control_bounds = [(-2.0, 2.0)],
			num_vars = 2,
			num_controls = 1,
			num_time_steps = 101,
		)
		mesh, _ = create_test_mesh(1, (2,))
		A = create_spd_matrix(2)
		b = randn(2)

		solution = Optimization.solve_optimal_control_problem(A, b, problem, mesh)

		@test size(solution.state, 1) == 2
		@test size(solution.control, 1) == 1
		@test solution.state[:, end] ≈ [0.0, 0.0] atol = 1e-2  # Should approach origin
		@test solution.objective >= 0
	end
end
