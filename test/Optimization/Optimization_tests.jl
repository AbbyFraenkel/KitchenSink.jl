using Test, LinearAlgebra, JuMP, Ipopt

using KitchenSink.Optimization

@testset "Optimization Tests" begin
    @testset "Problem Discretization" begin
        @testset "discretize_and_optimize" begin
            # Test problem: minimize ∫(u^2 + (∂u/∂x)^2)dx subject to -∂²u/∂x² = f(x), u(0) = u(1) = 0
            problem = KSPDEProblem(
                (u, x, t) -> -sum(∇²(u)),
                (u, x) -> 0.0,
                (0.0, 1.0),
                ((0.0, 1.0),),
                x -> sin(π * x[1])
            )
            num_elements = (10,)
            tspan = (0.0, 1.0)
            degree = 3
            solver_options = KSSolverOptions(100, 1e-6, true, 3, 0.1, 10, 2)

            model, solution = discretize_and_optimize(problem, num_elements, tspan, degree, solver_options)

            @test model isa JuMP.Model
            @test solution isa Matrix
            @test size(solution, 1) == 11  # 10 elements + 1 boundary
            @test size(solution, 2) == 2   # Start and end time points

            # Check boundary conditions
            @test all(isapprox.(solution[1, :], 0.0, atol=1e-6))
            @test all(isapprox.(solution[end, :], 0.0, atol=1e-6))

            # Check if solution is reasonable (should be close to sin(πx))
            x = range(0, 1, length=size(solution, 1))
            analytical_solution = sin.(π * x)
            @test isapprox(solution[:, end], analytical_solution, rtol=1e-2)
        end
    end

    @testset "Optimal Control Problems" begin
        @testset "solve_optimal_control_problem" begin
            # Test problem: minimize ∫(x²(t) + u²(t))dt subject to dx/dt = -x + u, x(0) = 1
            problem = KSOptimalControlProblem{Float64}(
                (x, u, t) -> -x + u,           # State equation
                (x, u) -> x^2 + u^2,           # Cost function
                x -> 0.0,                      # Terminal cost
                [1.0],                         # Initial state
                (0.0, 1.0),                    # Time span
                [(-Inf, Inf)]                  # Control bounds
            )
            solver = KSNewtonOptimizer(100, 1e-6)

            solution = solve_optimal_control_problem(problem, solver)

            @test solution isa OptimizationSolution
            @test size(solution.state) == (1, 101)  # Assuming 100 time steps
            @test size(solution.control) == (1, 101)
            @test solution.state[1, 1] ≈ 1.0
            @test solution.state[1, end] < 0.1  # Should decrease significantly
            @test all(solution.control .>= -Inf) && all(solution.control .<= Inf)
        end
    end

    @testset "JuMP Integration" begin
        @testset "create_jump_model" begin
            problem = KSOptimalControlProblem{Float64}(
                (x, u, t) -> -x + u,
                (x, u) -> x^2 + u^2,
                x -> 0.0,
                [1.0],
                (0.0, 1.0),
                [(-1.0, 1.0)]
            )

            model = create_jump_model(problem)

            @test model isa JuMP.Model
            @test num_variables(model) == 202  # 101 each for state and control
            @test num_constraints(model) == 301  # 100 dynamics constraints, 201 bound constraints

            # Test if the model can be solved
            set_optimizer(model, Ipopt.Optimizer)
            optimize!(model)
            @test termination_status(model) == MOI.LOCALLY_SOLVED
        end
    end

    @testset "Solution Extraction" begin
        @testset "extract_solution" begin
            problem = KSOptimalControlProblem{Float64}(
                (x, u, t) -> -x + u,
                (x, u) -> x^2 + u^2,
                x -> 0.0,
                [1.0],
                (0.0, 1.0),
                [(-1.0, 1.0)]
            )
            model = create_jump_model(problem)
            set_optimizer(model, Ipopt.Optimizer)
            optimize!(model)

            solution = extract_solution(model, problem)

            @test solution isa OptimizationSolution
            @test size(solution.state) == (1, 101)
            @test size(solution.control) == (1, 101)
            @test solution.objective > 0
            @test solution.state[1, 1] ≈ 1.0
            @test all(-1.0 .<= solution.control .<= 1.0)
        end
    end
end
