@testset "Adaptive TimeStepping" begin


    function mock_problem(u, t, dt, solver_options)
        # Simulate a problem that halves the error with each iteration
        local_error = norm(u) * 0.5
        u_new = u * 0.9 # Simulate some state update
        return u_new, local_error
    end

    @testset "adaptive_timestep! Tests" begin
        tol = 0.1
        initial_dt = 1.0
        t = 0.0
        u = [1.0, 2.0]
        solver_options = MockKSSolverOptions()

        @testset "Tolerance Met" begin
            u_new, dt_new = adaptive_timestep!(copy(u), t, initial_dt, mock_problem, solver_options, tol)
            @test dt_new < initial_dt
            @test norm(u_new) < norm(u)
        end

        @testset "Tolerance Already Met" begin
            local_tol = 10.0 # Set a tolerance that is already met
            u_new, dt_new = adaptive_timestep!(copy(u), t, initial_dt, mock_problem, solver_options, local_tol)
            @test dt_new == initial_dt
            @test all(u_new .== u)
        end

        @testset "Error Reduction Check" begin
            u_new, dt_new = adaptive_timestep!(copy(u), t, initial_dt, mock_problem, solver_options, tol)
            @test dt_new < initial_dt # Check that dt is reduced
        end

        @testset "Return Values" begin
            u_new, dt_new = adaptive_timestep!(copy(u), t, initial_dt, mock_problem, solver_options, tol)
            @test u_new isa AbstractVector
            @test dt_new isa Real
        end

        @testset "adaptive_timestep!" begin
            # Test ODE: dy/dt = -y, y(0) = 1
            problem = KSODEProblem((t, y) -> -y, (0.0, 1.0), [1.0])
            t, y = 0.0, [1.0]
            dt = 0.1
            tol = 1e-4
            while t < 1.0
                y, dt = adaptive_timestep!(y, t, dt, problem.ode, KSSolverOptions(100, tol, true), tol)
                t += dt
            end
            @test y[1] ≈ exp(-1.0) atol = 1e-3
        end
    end

end
