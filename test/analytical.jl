using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
using KitchenSink.KSTypes
using KitchenSink.CoordinateSystems
using KitchenSink.SpectralMethods
using KitchenSink.ProblemTypes

@testset "ProblemTypes Comprehensive Tests" begin
	@testset "ODE Problems" begin
		@testset "Exponential Decay" begin
			ode(t, y) = -y
			analytical_solution(t) = exp(-t)

			tspan = (0.0, 1.0)
			initial_condition = [1.0]

			problem = KSODEProblem(ode, t -> 0.0, ((0.0, 1.0),), initial_condition, tspan)

			num_elements = 10
			poly_degree = 4

			mesh = create_ocfc_mesh(
				KSCartesianCoordinates(((0.0, 1.0),)),
				num_elements,
				poly_degree,
				0,
				1,
				1,
			)

			solver_options = KSSolverOptions(
				100,
				1e-6,
				false,
				1,
				1e-3,
				num_elements,
				poly_degree,
				false,
				1,
				false,
				0.1,
				10,
				1e-3,
			)

			u, _ = solve_problem(problem, mesh, solver_options, mesh.cells)

			t_test = LinRange(0, 1, 100)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(t))
				for t in t_test
			]
			u_exact = analytical_solution.(t_test)

			@test isapprox(u_numerical, u_exact, rtol = 1e-4)
		end

		@testset "Coupled ODEs" begin
			ode1(t, y) = -y[1] + y[2]
			ode2(t, y) = -y[2] + y[1]
			analytical_solution(t) = [exp(-t) * cosh(t), exp(-t) * sinh(t)]

			tspan = (0.0, 1.0)
			initial_conditions = [1.0, 0.0]

			problem1 = KSODEProblem(
				ode1, t -> 0, ((0.0, 1.0),), [initial_conditions[1]], tspan)
			problem2 = KSODEProblem(
				ode2, t -> 0, ((0.0, 1.0),), [initial_conditions[2]], tspan)

			coupling_terms = Dict((1, 2) => (mesh) -> I, (2, 1) => (mesh) -> I)
			coupled_problem = KSCoupledProblem([problem1, problem2], coupling_terms)

			num_elements = 10
			poly_degree = 4

			mesh = create_ocfc_mesh(
				KSCartesianCoordinates(((0.0, 1.0),)),
				num_elements,
				poly_degree,
				0,
				1,
				1,
			)

			solver_options = KSSolverOptions(
				100,
				1e-6,
				false,
				1,
				1e-3,
				num_elements,
				poly_degree,
				false,
				1,
				false,
				0.1,
				10,
				1e-3,
			)

			u, _ = solve_problem(coupled_problem, mesh, solver_options, mesh.cells)

			t_test = LinRange(0, 1, 100)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(t))
				for t in t_test
			]
			u_exact = analytical_solution.(t_test)

			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end
	end

	@testset "PDE Problems" begin
		@testset "1D Poisson Equation" begin
			poisson_pde(x, D, Q) = -D[1]^2
			rhs(x) = π^2 * sin(π * x[1])
			bc(x) = 0.0
			analytical_solution(x) = sin(π * x[1])

			domain = ((0.0, 1.0),)
			coord_system = KSCartesianCoordinates(domain)
			problem = KSPDEProblem(poisson_pde, bc, domain, rhs, nothing, coord_system)

			num_elements = 10
			poly_degree = 4

			mesh = create_ocfc_mesh(coord_system, num_elements, poly_degree, 1, domain)

			A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)
			apply_bcs!(A, b, problem, mesh, domain)

			u = A \ b

			x_points = LinRange(0, 1, 100)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(x))
				for x in x_points
			]
			u_exact = [analytical_solution(x) for x in x_points]

			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end

		@testset "2D Poisson Equation" begin
			poisson_pde(x, D, Q) = -(D[1]^2 + D[2]^2)
			rhs(x) = 2π^2 * sin(π * x[1]) * sin(π * x[2])
			bc(x) = 0.0
			analytical_solution(x) = sin(π * x[1]) * sin(π * x[2])

			domain = ((0.0, 1.0), (0.0, 1.0))
			coord_system = KSCartesianCoordinates(domain)
			problem = KSPDEProblem(poisson_pde, bc, domain, rhs, nothing, coord_system)

			num_elements = (10, 10)
			poly_degree = (4, 4)

			mesh = create_ocfc_mesh(coord_system, num_elements, poly_degree, 1, domain)

			A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)
			apply_bcs!(A, b, problem, mesh, domain)

			u = A \ b

			x_points = LinRange(0, 1, 20)
			y_points = LinRange(0, 1, 20)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(x, y))
				for
				x in x_points, y in y_points
			]
			u_exact = [analytical_solution([x, y]) for x in x_points, y in y_points]

			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end

		@testset "1D Heat Equation" begin
			heat_pde(x, D, Q) = D[2] - D[1]^2
			bc(x) = 0.0
			initial_condition(x) = sin(π * x[1])
			analytical_solution(x) = exp(-π^2 * x[2]) * sin(π * x[1])

			domain = ((0.0, 1.0), (0.0, 1.0))
			coord_system = KSCartesianCoordinates(domain)
			problem = KSPDEProblem(
				heat_pde,
				bc,
				domain,
				initial_condition,
				(0.0, 1.0),
				coord_system,
			)

			num_elements = (20, 20)
			poly_degree = (4, 4)

			mesh = create_ocfc_mesh(coord_system, num_elements, poly_degree, 1, domain)

			A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)
			apply_bcs!(A, b, problem, mesh, domain)

			u = A \ b

			x_points = LinRange(0, 1, 20)
			t_points = LinRange(0, 1, 20)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(x, t))
				for
				x in x_points, t in t_points
			]
			u_exact = [analytical_solution([x, t]) for x in x_points, t in t_points]

			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end
	end

	@testset "DAE Problems" begin
		@testset "Simple DAE" begin
			dae(t, y, dy) = [dy[1] - y[2], y[1]^2 + y[2]^2 - 1]
			analytical_solution(t) = [cos(t), sin(t)]
			boundary_conditions(t) = [0.0, 0.0]
			domain = ((0.0, 1.0),)
			initial_conditions = [1.0, 0.0]
			tspan = (0.0, 1.0)
			coordinate_system = KSCartesianCoordinates(domain)

			problem = KSDAEProblem(
				dae,
				boundary_conditions,
				domain,
				initial_conditions,
				tspan,
				coordinate_system,
			)

			num_elements = 10
			poly_degree = 4

			mesh = create_ocfc_mesh(coordinate_system, num_elements, poly_degree, 0, 1)

			A, b = create_dae_system_matrix_and_vector(problem, mesh)

			u = A \ b

			t_test = LinRange(0, 1, 100)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(t))
				for t in t_test
			]
			u_exact = [analytical_solution(t) for t in t_test]

			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end
	end

	@testset "IDE Problems" begin
		@testset "Simple IDE" begin
			ide(t, y, int_y) = -y + int_y
			K(t, s) = exp(s - t)
			analytical_solution(t) = exp(-t)

			tspan = (0.0, 1.0)
			initial_condition = [1.0]

			problem = KSIDEProblem(ide, K, t -> 0, ((0.0, 1.0),), initial_condition, tspan)

			num_elements = 10
			poly_degree = 4

			mesh = create_ocfc_mesh(
				KSCartesianCoordinates(((0.0, 1.0),)),
				num_elements,
				poly_degree,
				0,
				1,
				1,
			)

			solver_options = KSSolverOptions(
				100,
				1e-6,
				false,
				1,
				1e-3,
				num_elements,
				poly_degree,
				false,
				1,
				false,
				0.1,
				10,
				1e-3,
			)

			u, _ = solve_problem(problem, mesh, solver_options, mesh.cells)

			t_test = LinRange(0, 1, 100)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(t))
				for t in t_test
			]
			u_exact = analytical_solution.(t_test)

			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end
	end

	@testset "PIDE Problems" begin
		@testset "Simple PIDE" begin
			pide(t, x, u, ux, uxx, int_u) = uxx + int_u
			K(x, s) = exp(-abs(x - s))
			analytical_solution(t, x) = exp(-t) * sin(π * x)

			tspan = (0.0, 1.0)
			domain = ((0.0, 1.0),)
			initial_condition(x) = sin(π * x)
			boundary_conditions(x) = 0.0

			problem = KSPIDEProblem(
				pide,
				K,
				boundary_conditions,
				domain,
				initial_condition,
				tspan,
			)

			num_elements = (10, 10)
			poly_degree = (4, 4)

			mesh = create_ocfc_mesh(
				KSCartesianCoordinates(domain),
				num_elements,
				poly_degree,
				1,
				domain,
			)

			solver_options = KSSolverOptions(
				100,
				1e-6,
				true,
				5,
				1e-3,
				maximum(num_elements),
				maximum(poly_degree),
				false,
				1,
				false,
				0.01,
				100,
				1e-3,
			)

			u, _ = solve_problem(problem, mesh, solver_options, mesh.cells)

			t_test = LinRange(0, 1, 10)
			x_test = LinRange(0, 1, 10)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(t, x))
				for
				t in t_test, x in x_test
			]
			u_exact = [analytical_solution(t, x) for t in t_test, x in x_test]

			@test isapprox(u_numerical, u_exact, rtol = 1e-2)
		end
	end

	@testset "Moving Boundary PDE Problems" begin
		@testset "1D Stefan Problem" begin
			L = 1.0
			α = 0.1
			T_m = 0.0  # Melting temperature
			T_h = 1.0  # Temperature at the heated end

			function pde(x, t, u, ux, ut)
				return ut - α * ux^2
			end

			function boundary_motion(t, s, u, ux)
				return -ux[1]
			end

			initial_condition(x) = T_m
			boundary_conditions(x, t) = x[1] == 0 ? T_h : T_m

			function analytical_solution(x, t)
				λ = 0.62  # Obtained by solving transcendental equation
				s = 2 * λ * sqrt(α * t)
				if x <= s
					return T_h - (T_h - T_m) * erf(x / (2 * sqrt(α * t))) / erf(λ)
				else
					return T_m
				end
			end

			domain = ((0.0, L), (0.0, 1.0))
			coord_system = KSCartesianCoordinates(domain)
			problem = KSMovingBoundaryPDEProblem(
				pde,
				boundary_conditions,
				domain,
				initial_condition,
				(0.0, 1.0),
				boundary_motion,
				coord_system,
			)

			num_elements = 20
			poly_degree = 4

			mesh = create_ocfc_mesh(coord_system, num_elements, poly_degree, 1, domain)

			A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)
			apply_bcs!(A, b, problem, mesh, domain)

			u = A \ b

			x_points = LinRange(0, L, 100)
			t_final = 1.0
			u_numerical = [
				SpectralMethods.interpolate_solution(
					mesh, u, SVector(x, t_final)) for
				x in x_points
			]
			u_exact = [analytical_solution(x, t_final) for x in x_points]

			@test all(0 .<= u_numerical .<= 1)
			@test isapprox(u_numerical, u_exact, rtol = 1e-2)
		end
	end

	@testset "Nonlinear PDE Tests" begin
		@testset "2D Nonlinear Heat Equation" begin
			L_x = 1.0
			L_y = 1.0
			α = 0.1

			function pde(x, t, u, ux, uy, ut)
				return ut - α * (ux^2 + uy^2) - u^2
			end

			function initial_condition(x)
				return sin(π * x[1] / L_x) * sin(π * x[2] / L_y)
			end

			function boundary_conditions(x, t)
				return 0.0
			end

			function analytical_solution(x, t)
				# Since it's nonlinear, let's use a simple analytical solution
				return exp(-π^2 * α * t) * sin(π * x[1] / L_x) * sin(π * x[2] / L_y)
			end

			domain = ((0.0, L_x), (0.0, L_y))
			tspan = (0.0, 1.0)

			problem = KSPDEProblem(
				pde, boundary_conditions, domain, initial_condition, tspan)

			num_elements = (20, 20)
			poly_degree = (4, 4)

			mesh = create_ocfc_mesh(
				KSCartesianCoordinates(domain),
				num_elements,
				poly_degree,
				1,
				domain,
			)

			A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)
			apply_bcs!(A, b, problem, mesh, domain)

			u = A \ b

			x_points = LinRange(0, L_x, 20)
			y_points = LinRange(0, L_y, 20)
			u_numerical = [
				SpectralMethods.interpolate_solution(mesh, u, SVector(x, y))
				for
				x in x_points, y in y_points
			]
			u_exact = [analytical_solution([x, y], 1.0) for x in x_points, y in y_points]

			@test all(0 .<= u_numerical .<= 1)
			@test isapprox(u_numerical, reverse(u_numerical; dims = 1), rtol = 1e-3)
			@test isapprox(u_numerical, reverse(u_numerical; dims = 2), rtol = 1e-3)
			@test isapprox(u_numerical, u_exact, rtol = 1e-3)
		end
	end
end
