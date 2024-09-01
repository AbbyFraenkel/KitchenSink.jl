using Test, LinearAlgebra, SparseArrays

using KitchenSink.TimeStepping
@testset "Implicit methods" begin

	# Helper functions
	function mock_solver(residual, u_init)
		u_new = u_init
		for _ in 1:10
			u_new = residual(u_new) + u_new
		end
		return u_new
	end

	f_linear(u, t; a = 1.0, b = 0.0) = a * u + b
	f_nonlinear(u, t) = u^2 + t

	@testset "Implicit Methods" begin
		@testset "backward_euler!" begin
			@testset "Linear function" begin
				u0, t0, dt = 1.0, 0.0, 0.1
				a, b = 1.0, 0.0
				u_expected = (u0 + dt * b) / (1 - dt * a)
				u_computed = backward_euler!([u0], t0, dt, (u, t) -> f_linear(u, t, a = a, b = b), mock_solver)
				@test u_computed[1] ≈ u_expected
			end

			@testset "Nonlinear function" begin
				u0, t0, dt = 1.0, 0.0, 0.1
				u_expected = 1.1 # Approximated value
				u_computed = backward_euler!([u0], t0, dt, f_nonlinear, mock_solver)
				@test u_computed[1]≈u_expected atol=1e-2
			end

			@testset "Stiff ODE" begin
				problem = KSODEProblem((y) -> -1000 * y + 3000 - 2000 * exp(-t), (0.0, 0.1), [0.0])
				t, y = 0.0, [0.0]
				for _ in 1:10
					y = backward_euler!(y, t, 0.01, problem.ode, (F, x) -> x .- F(x))
					t += 0.01
				end
				exact = 3 - 0.998 * exp(-1000 * 0.1) - 2.002 * exp(-0.1)
				@test y[1]≈exact atol=1e-2
			end
		end

		@testset "implicit_midpoint!" begin
			@testset "Linear function" begin
				u0, t0, dt = 1.0, 0.0, 0.1
				a, b = 1.0, 0.0
				u_expected = (u0 + dt * b) / (1 - dt * a)
				u_computed = implicit_midpoint!([u0], t0, dt, (u, t) -> f_linear(u, t, a = a, b = b), mock_solver)
				@test u_computed[1] ≈ u_expected
			end

			@testset "Nonlinear function" begin
				u0, t0, dt = 1.0, 0.0, 0.1
				u_expected = 1.1 # Approximated value
				u_computed = implicit_midpoint!([u0], t0, dt, f_nonlinear, mock_solver)
				@test u_computed[1]≈u_expected atol=1e-2
			end

			@testset "Simple ODE" begin
				problem = KSODEProblem((y) -> -y, (0.0, 1.0), [1.0])
				t, y = 0.0, [1.0]
				for _ in 1:10
					y = implicit_midpoint!(y, t, 0.1, problem.ode, (F, x) -> x .- F(x))
					t += 0.1
				end
				@test y[1]≈exp(-1.0) atol=1e-3
			end
		end

		@testset "PDE with Implicit Euler" begin
			pde = KSPDEProblem((u, x, t) -> [sum(∇²(u))],
							   (u, x) -> 0.0,
							   (0.0, 1.0),
							   ((0.0, 1.0),))
			mesh = Preprocessing.create_mesh(pde.domain, (20,), 2, KSCartesianCoordinates{1, Float64}((0.0,)))
			u0 = [sin(π * p.coordinates[1]) for p in mesh.nodes]
			solution = solve_pde(pde, mesh, u0, KSTimeSteppingSolver(:implicit_euler, 0.01, 1.0, 1e-6))
			@test maximum(abs.(solution[end])) < maximum(abs.(u0))
		end

		@testset "DAE with Implicit Euler" begin
			L = 1.0
			dae = KSDAEProblem((y, yp) -> [yp[1] - y[2], yp[2] + y[4] * y[1], y[1]^2 + y[3]^2 - L^2],
							   (0.0, 1.0),
							   [L, 0.0, 0.0, 0.0])
			solution = solve_dae(dae, KSTimeSteppingSolver(:implicit_euler, 0.01, 1.0, 1e-6))
			@test isapprox(solution[end][1]^2 + solution[end][3]^2, L^2, atol = 1e-3)
		end
	end
end

@testset "Explicit methods" begin
	using Test, LinearAlgebra, SparseArrays
	using ..KSTypes, ..ProblemTypes, ..TimeStepping, ..SpectralMethods, ..Preprocessing, ..CoordinateSystems

	# Helper functions
	function simple_decay(y)
		-y
	end

	function lotka_volterra(y)
		α, β, δ, γ = 1.5, 1.0, 3.0, 1.0
		[α * y[1] - β * y[1] * y[2], δ * y[1] * y[2] - γ * y[2]]
	end

	function harmonic_oscillator(y)
		[y[2], -y[1]]
	end

	@testset "Explicit Methods" begin
		@testset "Simple Decay ODE" begin
			problem = KSODEProblem(simple_decay, (0.0, 1.0), [1.0])
			exact_solution(t) = exp(-t)

			@testset "Forward Euler" begin
				t, y = 0.0, [1.0]
				dt = 0.001
				for _ in 1:1000
					y = forward_euler!(y, t, dt, problem.ode)
					t += dt
				end
				@test y[1]≈exact_solution(1.0) atol=1e-2
			end

			@testset "RK4" begin
				t, y = 0.0, [1.0]
				dt = 0.1
				for _ in 1:10
					y = rk4!(y, t, dt, problem.ode)
					t += dt
				end
				@test y[1]≈exact_solution(1.0) atol=1e-4
			end

			@testset "RK45" begin
				t, y = 0.0, [1.0]
				dt = 0.1
				y, error_estimate = rk45!(y, t, dt, problem.ode)
				@test y[1]≈exact_solution(dt) atol=1e-5
				@test error_estimate < 1e-5
			end
		end

		@testset "Lotka-Volterra System" begin
			problem = KSODEProblem(lotka_volterra, (0.0, 10.0), [1.0, 1.0])
			solution = solve_ode(problem, KSTimeSteppingSolver(:rk4, 0.01, 10.0, 1e-6))

			@test length(solution) == 2
			@test all(x -> x > 0, solution[end])

			# Check if the solution is oscillatory
			peaks = findlocalmaxima(solution.u[1])
			@test length(peaks) > 1
		end

		@testset "Symplectic Euler for Harmonic Oscillator" begin
			problem = KSODEProblem(harmonic_oscillator, (0.0, 1000.0), [1.0, 0.0])
			solution = solve_ode(problem, KSTimeSteppingSolver(:symplectic_euler, 0.1, 1000.0, 1e-6))

			initial_energy = 0.5 * (problem.initial_conditions[1]^2 + problem.initial_conditions[2]^2)
			final_energy = 0.5 * (solution[end][1]^2 + solution[end][2]^2)
			@test isapprox(initial_energy, final_energy, rtol = 1e-3)

			# Check if the solution is periodic
			@test isapprox(solution[1], solution[end], rtol = 1e-2)
		end

		@testset "Stability Test" begin
			stiff_problem = KSODEProblem((y) -> -50 * y, (0.0, 1.0), [1.0])

			@testset "Forward Euler Instability" begin
				t, y = 0.0, [1.0]
				dt = 0.1
				@test_throws DomainError begin
					for _ in 1:10
						y = forward_euler!(y, t, dt, stiff_problem.ode)
						@assert !isnan(y[1])&&abs(y[1]) < 1e10 "Solution blew up"
						t += dt
					end
				end
			end

			@testset "RK4 Stability" begin
				t, y = 0.0, [1.0]
				dt = 0.1
				for _ in 1:10
					y = rk4!(y, t, dt, stiff_problem.ode)
					t += dt
				end
				@test !isnan(y[1]) && abs(y[1]) < 1.0
			end
		end
	end
end

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
		# solver_options = MockKSSolverOptions()

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
			@test u_new isa AbstractArray
			@test dt_new isa Real
		end

		@testset "adaptive_timestep!" begin
			# Test ODE: dy/dt = -y, y(0) = 1
			problem = KSODEProblem((y) -> -y, (0.0, 1.0), [1.0])
			t, y = 0.0, [1.0]
			dt = 0.1
			tol = 1e-4
			while t < 1.0
				y, dt = adaptive_timestep!(y, t, dt, problem.ode, KSSolverOptions(100, tol, true), tol)
				t += dt
			end
			@test y[1]≈exp(-1.0) atol=1e-3
		end
	end
end
