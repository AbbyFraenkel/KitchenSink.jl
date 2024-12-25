using Test
using LinearAlgebra, SparseArrays
using OrdinaryDiffEq
using KitchenSink.KSTypes
using KitchenSink.TimeStepping
using KitchenSink.CoordinateSystems
using KitchenSink.SpectralMethods

@testset "TimeStepping Tests" begin


    @testset "ODE Integration" begin
        domain, coord_sys, mesh = create_test_setup()

        # Create simple heat equation
        problem = KSPDEProblem(;
            pde = (x, D) -> (D[1] + D[2], zeros(1)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = x -> exp(-norm(x)^2), # Gaussian IC
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Test problem conversion
        ode_prob = to_ode_problem(problem, mesh)
        @test ode_prob isa ODEProblem
        @test ode_prob.tspan == problem.tspan

        # Test solution
        sol = solve_time_dependent_problem(
            problem, mesh, dt=0.1)
        @test sol.retcode == :Success
        @test length(sol.t) > 1
    end

    @testset "DAE Integration" begin
        domain, coord_sys, mesh = create_test_setup()

        problem = KSDAEProblem(;
            dae = (t, x, D) -> (D[1], zeros(2)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = zeros(2),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 2,
            num_algebraic_vars = 1
        )

        opts = process_problem_options(problem, 0.1)
        @test haskey(opts, :diffstates)
        @test length(opts[:diffstates]) == problem.num_vars

        # Test solution
        sol = solve_time_dependent_problem(
            problem, mesh, alg=IDA())
        @test sol.retcode == :Success
    end

    @testset "Moving Boundary Integration" begin
        domain, coord_sys, mesh = create_test_setup()

        problem = KSMovingBoundaryPDEProblem(;
            pde = (x, D) -> (D[1], zeros(1)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = zeros(1),
            tspan = (0.0, 1.0),
            boundary_motion = (x, t) -> x .+ 0.1t,  # Simple translation
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Test options
        opts = process_problem_options(problem, 0.1)
        @test opts[:adaptive] == true
        @test opts[:force_dtmin] == true

        # Test solution
        sol = solve_time_dependent_problem(
            problem, mesh, dt=0.1)
        @test sol.retcode == :Success

        # Test mesh motion updates
        A = spzeros(10, 10)
        b = zeros(10)
        t = 0.5
        update_time_dependent_terms!(A, b, t, problem, mesh)
        @test !iszero(A)  # Should have ALE terms
    end

    @testset "IDE Integration" begin
        domain, coord_sys, mesh = create_test_setup()

        problem = KSIDEProblem(;
            ide = (t, x, D) -> (D[1], zeros(1)),
            kernel = (x, y) -> exp(-norm(x - y)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = zeros(1),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Test integral term updates
        A = spzeros(10, 10)
        b = zeros(10)
        t = 0.5
        update_time_dependent_terms!(A, b, t, problem, mesh)

        # Test solution
        sol = solve_time_dependent_problem(
            problem, mesh, dt=0.1)
        @test sol.retcode == :Success
    end

    @testset "PIDE Integration" begin
        domain, coord_sys, mesh = create_test_setup()

        problem = KSPIDEProblem(;
            pide = (x, D) -> (D[1], zeros(1)),
            kernel = (x, y) -> exp(-norm(x - y)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = zeros(1),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Test both PDE and integral updates
        A = spzeros(10, 10)
        b = zeros(10)
        t = 0.5
        update_time_dependent_terms!(A, b, t, problem, mesh)

        # Test solution
        sol = solve_time_dependent_problem(
            problem, mesh, dt=0.1)
        @test sol.retcode == :Success
    end

    @testset "Solution Processing" begin
        domain, coord_sys, mesh = create_test_setup()

        # Create test problem
        problem = KSPDEProblem(;
            pde = (x, D) -> (D[1] + D[2], zeros(1)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = x -> exp(-norm(x)^2),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Solve
        sol = solve_time_dependent_problem(
            problem, mesh, dt=0.1)

        # Test solution extraction
        t_test = 0.5
        u_test = extract_solution_at_time(
            sol, t_test, problem, mesh)

        @test length(u_test) == get_total_dof(mesh)
        @test !any(isnan, u_test)
        @test !any(isinf, u_test)
    end

    @testset "Algorithm Selection" begin
        domain, coord_sys, mesh = create_test_setup()

        # Test different algorithms
        algorithms = [
            Tsit5(),
            Rodas4(),
            QNDF(),
            RadauIIA5()
        ]

        problem = KSPDEProblem(;
            pde = (x, D) -> (D[1] + D[2], zeros(1)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = zeros(1),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        for alg in algorithms
            sol = solve_time_dependent_problem(
                problem, mesh, alg=alg)
            @test sol.retcode == :Success
        end
    end

    @testset "Performance and Stability" begin
        domain, coord_sys, mesh = create_test_setup()

        # Create stiff problem
        λ = -50.0  # Stiff eigenvalue
        problem = KSODEProblem(;
            ode = (t, x, D) -> (λ*x, zeros(1)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = ones(1),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Test with explicit solver
        sol_explicit = solve_time_dependent_problem(
            problem, mesh, alg=RK4())

        # Test with implicit solver
        sol_implicit = solve_time_dependent_problem(
            problem, mesh, alg=Rodas4())

        # Implicit should use fewer steps
        @test length(sol_implicit.t) < length(sol_explicit.t)
    end

    @testset "Error Control" begin
        domain, coord_sys, mesh = create_test_setup()

        problem = KSPDEProblem(;
            pde = (x, D) -> (D[1] + D[2], zeros(1)),
            boundary_conditions = [],
            domain = domain,
            initial_conditions = zeros(1),
            tspan = (0.0, 1.0),
            coordinate_system = coord_sys,
            num_vars = 1
        )

        # Test different tolerances
        tolerances = [1e-3, 1e-6, 1e-9]
        steps = Int[]

        for tol in tolerances
            sol = solve_time_dependent_problem(
                problem, mesh,
                abstol=tol, reltol=tol)
            push!(steps, length(sol.t))
        end

        # Stricter tolerance should need more steps
        @test issorted(steps)
    end
end
