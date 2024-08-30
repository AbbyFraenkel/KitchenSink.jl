using Test
using StaticArrays, SparseArrays
using KitchenSink.KSTypes

@testset "KSTypes" begin
	@testset "Abstract Types" begin
		@test isabstracttype(AbstractKSElement)
		@test isabstracttype(AbstractKSMesh)
		@test isabstracttype(AbstractKSProblem)
		@test isabstracttype(AbstractKSCoordinateSystem)
		@test isabstracttype(AbstractKSBoundaryCondition)
		@test isabstracttype(AbstractKSBasisFunction)
		@test isabstracttype(AbstractKSSolver)
		@test isabstracttype(AbstractKSOptimizationProblem)
		@test isabstracttype(AbstractKSLinearSolver)
	end

	@testset "KSBasisFunction" begin
		@testset "Valid cases" begin
			bf = KSBasisFunction(1, x -> x[1]^2)
			@test bf.id == 1
			@test bf.function_handle([2.0, 3.0]) == 4.0
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSBasisFunction(0, x -> x[1]^2)
			@test_throws MethodError KSBasisFunction(1, "not a function")
		end
	end

	@testset "StandardElement" begin
		points_with_boundary = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
		collocation_points = [(0.5, 0.5)]
		collocation_weights = [1.0]
		differentiation_matrices = [rand(2, 2)]
		quadrature_matrices = [rand(2, 2)]
		level = 2

		@testset "Valid case" begin
			element = StandardElement(points_with_boundary, collocation_points,
									  collocation_weights, differentiation_matrices,
									  quadrature_matrices, level)

			@test element.points_with_boundary == points_with_boundary
			@test element.collocation_points == collocation_points
			@test element.collocation_weights == collocation_weights
			@test element.differentiation_matrices == differentiation_matrices
			@test element.quadrature_matrices == quadrature_matrices
			@test element.level == level
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError StandardElement(points_with_boundary, collocation_points,
													   collocation_weights, differentiation_matrices,
													   quadrature_matrices, -1)
		end
	end

	@testset "KSElement" begin
		@testset "Valid cases" begin
			element = KSElement{Float64, 2}(1, 2, (3, 3))
			@test element.id == 1
			@test element.level == 2
			@test element.polynomial_degree == (3, 3)
			@test element.parent === nothing
			@test element.children === nothing
			@test element.neighbors === nothing
			@test element.is_leaf == true
			@test element.error_estimate == 0.0
			@test element.legendre_decay_rate == 0.0

			element_with_options = KSElement{Float64, 2}(1, 2, (3, 3), parent = 2, children = [3, 4], neighbors = [5, 6],
														 is_leaf = false, error_estimate = 0.1, legendre_decay_rate = 0.01)
			@test element_with_options.parent == 2
			@test element_with_options.children == [3, 4]
			@test element_with_options.neighbors == [5, 6]
			@test element_with_options.is_leaf == false
			@test element_with_options.error_estimate == 0.1
			@test element_with_options.legendre_decay_rate == 0.01
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSElement{Float64, 2}(0, 2, (3, 3))
			@test_throws ArgumentError KSElement{Float64, 2}(1, -1, (3, 3))
			@test_throws ArgumentError KSElement{Float64, 2}(1, 2, (-1, 3))
		end
	end

	@testset "KSMesh" begin
		elements = [KSElement{Float64, 2}(i, 1, (2, 2)) for i in 1:3]
		tensor_product_masks = [trues(2, 2) for _ in 1:3]
		location_matrices = [Dict{Int, Int}() for _ in 1:3]

		@testset "Valid cases" begin
			mesh = KSMesh(elements)
			@test mesh.elements == elements
			@test isempty(mesh.tensor_product_masks)
			@test isempty(mesh.location_matrices)
			@test mesh.global_error_estimate == 0.0

			full_mesh = KSMesh(elements, tensor_product_masks, location_matrices, 0.1)
			@test full_mesh.elements == elements
			@test full_mesh.tensor_product_masks == tensor_product_masks
			@test full_mesh.location_matrices == location_matrices
			@test full_mesh.global_error_estimate == 0.1
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSMesh(KSElement{Float64, 2}[])
		end
	end

	@testset "KSProblem" begin
		equation = (u, p, t) -> -u
		domain = ((0.0, 1.0), (0.0, 1.0))
		boundary_conditions = (u, p, t) -> 0
		initial_conditions = [1.0]
		tspan = (0.0, 1.0)

		@testset "Valid cases" begin
			problem = KSProblem(equation, boundary_conditions, domain, initial_conditions, tspan)
			@test problem.equation == equation
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
			@test problem.continuity_order == 2

			problem_with_options = KSProblem(equation, boundary_conditions, domain, initial_conditions, tspan, 3)
			@test problem_with_options.continuity_order == 3
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSProblem("not a function", boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSProblem(equation, "not a function", domain, initial_conditions, tspan)
			@test_throws MethodError KSProblem(equation, boundary_conditions, "not a tuple", initial_conditions, tspan)
			@test_throws ArgumentError KSProblem(equation, boundary_conditions, domain, initial_conditions, tspan, -1)
		end
	end

	@testset "KSODEProblem" begin
		ode = (u, p, t) -> -u
		boundary_conditions = (u, p, t) -> 0
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = [1.0]
		tspan = (0.0, 1.0)

		@testset "Valid case" begin
			problem = KSODEProblem(ode, boundary_conditions, domain, initial_conditions, tspan)
			@test problem.ode == ode
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSODEProblem("not a function", boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSODEProblem(ode, "not a function", domain, initial_conditions, tspan)
			@test_throws MethodError KSODEProblem(ode, boundary_conditions, "not a tuple", initial_conditions, tspan)
		end
	end

	@testset "KSBVDAEProblem" begin
		f = (t, y) -> y
		g = (t, y) -> y[1] - y[2]
		bc = (ya, yb) -> ya[1] - yb[1]
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = [1.0, 0.0]
		algebraic_vars = [false, true]
		tspan = (0.0, 1.0)

		@testset "Valid case" begin
			problem = KSBVDAEProblem(f, g, bc, domain, initial_conditions, algebraic_vars, tspan)
			@test problem.f == f
			@test problem.g == g
			@test problem.bc == bc
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.algebraic_vars == algebraic_vars
			@test problem.tspan == tspan
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSBVDAEProblem("not a function", g, bc, domain, initial_conditions, algebraic_vars, tspan)
			@test_throws MethodError KSBVDAEProblem(f, "not a function", bc, domain, initial_conditions, algebraic_vars, tspan)
			@test_throws MethodError KSBVDAEProblem(f, g, "not a function", domain, initial_conditions, algebraic_vars, tspan)
			@test_throws MethodError KSBVDAEProblem(f, g, bc, "not a tuple", initial_conditions, algebraic_vars, tspan)
			@test_throws ArgumentError KSBVDAEProblem(f, g, bc, domain, [1.0], algebraic_vars, tspan)
		end
	end

	@testset "KSCoupledProblem" begin
		problem1 = KSODEProblem((u, p, t) -> -u, (u, p, t) -> 0, ((0.0, 1.0),), [1.0], (0.0, 1.0))
		problem2 = KSODEProblem((u, p, t) -> -2u, (u, p, t) -> 0, ((0.0, 1.0),), [2.0], (0.0, 1.0))
		coupling_terms = [nothing (u1, u2)->u1; (u1, u2)->u2   nothing]

		@testset "Valid case" begin
			coupled_problem = KSCoupledProblem([problem1, problem2], coupling_terms)
			@test coupled_problem.problems == [problem1, problem2]
			@test coupled_problem.coupling_terms == coupling_terms
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSCoupledProblem([problem1, problem2], Matrix{Union{Nothing, Function}}(undef, 3, 3))
		end
	end

	@testset "KSIDEProblem" begin
		f = (u, x, t) -> -u
		K = (u, x, y, t) -> exp(-abs(x - y))
		boundary_conditions = (u, x, t) -> 0
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = [1.0, 0.0]
		tspan = (0.0, 1.0)

		@testset "Valid case" begin
			problem = KSIDEProblem(f, K, boundary_conditions, domain, initial_conditions, tspan)
			@test problem.f == f
			@test problem.K == K
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSIDEProblem("not a function", K, boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSIDEProblem(f, "not a function", boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSIDEProblem(f, K, "not a function", domain, initial_conditions, tspan)
			@test_throws MethodError KSIDEProblem(f, K, boundary_conditions, "not a tuple", initial_conditions, tspan)
		end
	end

	@testset "KSPDEProblem" begin
		pde = (u, x, t) -> -u
		boundary_conditions = (u, x, t) -> 0
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = (x, y) -> sin(π * x) * sin(π * y)
		tspan = (0.0, 1.0)

		@testset "Valid case" begin
			problem = KSPDEProblem(pde, boundary_conditions, domain, initial_conditions, tspan)
			@test problem.pde == pde
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSPDEProblem("not a function", boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSPDEProblem(pde, "not a function", domain, initial_conditions, tspan)
			@test_throws MethodError KSPDEProblem(pde, boundary_conditions, "not a tuple", initial_conditions, tspan)
		end
	end

	@testset "KSDAEProblem" begin
		dae = (du, u, p, t) -> du[1] - (u[1] - u[2]^2)
		boundary_conditions = (u, p, t) -> 0
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = [1.0, 0.0]
		tspan = (0.0, 1.0)

		@testset "Valid case" begin
			problem = KSDAEProblem(dae, boundary_conditions, domain, initial_conditions, tspan)
			@test problem.dae == dae
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSDAEProblem("not a function", boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSDAEProblem(dae, "not a function", domain, initial_conditions, tspan)
			@test_throws MethodError KSDAEProblem(dae, boundary_conditions, "not a tuple", initial_conditions, tspan)
		end
	end

	@testset "KSMovingBoundaryPDEProblem" begin
		pde = (u, x, t) -> -u
		boundary_conditions = (u, x, t) -> 0
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = [1.0, 0.0]
		tspan = (0.0, 1.0)
		boundary_motion = t -> 1.0 + 0.1 * t

		@testset "Valid case" begin
			problem = KSMovingBoundaryPDEProblem(pde, boundary_conditions, domain, initial_conditions, tspan, boundary_motion)
			@test problem.pde == pde
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
			@test problem.boundary_motion == boundary_motion
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSMovingBoundaryPDEProblem("not a function", boundary_conditions, domain, initial_conditions, tspan, boundary_motion)
			@test_throws MethodError KSMovingBoundaryPDEProblem(pde, "not a function", domain, initial_conditions, tspan, boundary_motion)
			@test_throws MethodError KSMovingBoundaryPDEProblem(pde, boundary_conditions, "not a tuple", initial_conditions, tspan, boundary_motion)
			@test_throws MethodError KSMovingBoundaryPDEProblem(pde, boundary_conditions, domain, initial_conditions, tspan, "not a function")
		end
	end

	@testset "KSPIDEProblem" begin
		pide = (u, x, t) -> -u
		kernel = (u, x, y, t) -> exp(-abs(x - y))
		boundary_conditions = (u, x, t) -> 0
		domain = ((0.0, 1.0), (0.0, 1.0))
		initial_conditions = (x, y) -> sin(π * x) * sin(π * y)
		tspan = (0.0, 1.0)

		@testset "Valid case" begin
			problem = KSPIDEProblem(pide, kernel, boundary_conditions, domain, initial_conditions, tspan)
			@test problem.pide == pide
			@test problem.kernel == kernel
			@test problem.boundary_conditions == boundary_conditions
			@test problem.domain == domain
			@test problem.initial_conditions == initial_conditions
			@test problem.tspan == tspan
		end

		@testset "Invalid cases" begin
			@test_throws MethodError KSPIDEProblem("not a function", kernel, boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSPIDEProblem(pide, "not a function", boundary_conditions, domain, initial_conditions, tspan)
			@test_throws MethodError KSPIDEProblem(pide, kernel, "not a function", domain, initial_conditions, tspan)
			@test_throws MethodError KSPIDEProblem(pide, kernel, boundary_conditions, "not a tuple", initial_conditions, tspan)
		end
	end

	@testset "KSDiscretizedProblem" begin
		time_nodes = [0.0, 0.5, 1.0]
		spatial_nodes = ([0.0, 1.0], [0.0, 1.0])
		system_matrix = [1.0 0.0; 0.0 1.0]
		initial_conditions = [0.0, 0.0]
		problem_functions = (x -> x, x -> x^2)

		@testset "Valid case" begin
			problem = KSDiscretizedProblem(time_nodes, spatial_nodes, system_matrix, initial_conditions, problem_functions)
			@test problem.time_nodes == time_nodes
			@test problem.spatial_nodes == spatial_nodes
			@test problem.system_matrix == system_matrix
			@test problem.initial_conditions == initial_conditions
			@test problem.problem_functions == problem_functions
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSDiscretizedProblem(Float64[], spatial_nodes, system_matrix, initial_conditions, problem_functions)
			@test_throws MethodError KSDiscretizedProblem(time_nodes, spatial_nodes, system_matrix, initial_conditions, (x -> x, "not a function"))
		end
	end

	@testset "KSOptimalControlProblem" begin
		state_eq = [(x, u, t) -> x + u, (x, u, t) -> x - u]
		cost_func = [(x, u, t) -> x^2 + u^2, (x, u, t) -> x^2 - u^2]
		terminal_cost = x -> sum(x .^ 2)
		initial_state = [0.0, 1.0]
		time_span = (0.0, 1.0)
		control_bounds = [(0.0, 1.0), (0.0, 1.0)]

		@testset "Valid case" begin
			problem = KSOptimalControlProblem(state_eq, cost_func, terminal_cost, initial_state, time_span, control_bounds)
			@test problem.state_equations == state_eq
			@test problem.cost_functions == cost_func
			@test problem.terminal_cost == terminal_cost
			@test problem.initial_state == initial_state
			@test problem.time_span == time_span
			@test problem.control_bounds == control_bounds
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSOptimalControlProblem(state_eq, cost_func[1:1], terminal_cost, initial_state, time_span, control_bounds)
			@test_throws ArgumentError KSOptimalControlProblem(state_eq, cost_func, terminal_cost, initial_state[1:1], time_span, control_bounds)
			@test_throws ArgumentError KSOptimalControlProblem(state_eq, cost_func, terminal_cost, initial_state, time_span, control_bounds[1:1])
		end
	end

	@testset "Coordinate Systems" begin
		@testset "KSCartesianCoordinates" begin
			coords = KSCartesianCoordinates(((0.0, 1.0), (-1, 1), (0, 2.0)))  # Mixing Int and Float
			@test coords.ranges == ((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
			# @test active_dimensions(coords) == ((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
		end

	                @testset "KSPolarCoordinates" begin
                coords = KSPolarCoordinates((0, 2.0), (0.0, 2π))  # Mixing Int and Float
                @test coords.r == (0.0, 2.0)
                @test all(isapprox.(coords.theta, (0.0, 2π), atol=1e-9))
                # @test active_dimensions(coords) == ((0.0, 2.0), (0.0, 2π))

                coords_partial = KSPolarCoordinates((0, 2.0), nothing)
                @test coords_partial.r == (0.0, 2.0)
                @test coords_partial.theta === nothing
                # @test active_dimensions(coords_partial) == ((0.0, 2.0),)
            end

            @testset "KSCylindricalCoordinates" begin
                coords = KSCylindricalCoordinates((0, 2.0), (0.0, 2π), (-1, 1))  # Mixing Int and Float
                @test coords.r == (0.0, 2.0)
                @test all(isapprox.(coords.theta, (0.0, 2π), atol=1e-9))
                @test coords.z == (-1.0, 1.0)
                # @test active_dimensions(coords) == ((0.0, 2.0), (0.0, 2π), (-1.0, 1.0))

                coords_partial = KSCylindricalCoordinates((0, 2.0), (0.0, 2π), nothing)
                @test coords_partial.r == (0.0, 2.0)
                @test all(isapprox.(coords_partial.theta, (0.0, 2π), atol=1e-9))
                @test coords_partial.z === nothing
                # @test active_dimensions(coords_partial) == ((0.0, 2.0), (0.0, 2π))
            end

            @testset "KSSphericalCoordinates" begin
                coords = KSSphericalCoordinates((0, 2.0), (0.0, π), (0, 2π))  # Mixing Int and Float
                @test coords.r == (0.0, 2.0)
                @test all(isapprox.(coords.theta, (0.0, π), atol=1e-9))
                @test all(isapprox.(coords.phi, (0.0, 2π), atol=1e-9))
                # @test active_dimensions(coords) == ((0.0, 2.0), (0.0, π), (0.0, 2π))

                coords_partial = KSSphericalCoordinates((0, 2.0), (0.0, π), nothing)
                @test coords_partial.r == (0.0, 2.0)
                @test all(isapprox.(coords_partial.theta, (0.0, π), atol=1e-9))
                @test coords_partial.phi === nothing
                # @test active_dimensions(coords_partial) == ((0.0, 2.0), (0.0, π))
            end
	end

	@testset "Boundary Conditions" begin
		@testset "KSDirichletBC" begin
			value = (x, t) -> sin(π * x)
			boundary = x -> x == 0 || x == 1

			bc = KSDirichletBC(value, boundary)
			@test bc.value == value
			@test bc.boundary == boundary

			@test_throws MethodError KSDirichletBC("not a function", boundary)
			@test_throws MethodError KSDirichletBC(value, "not a function")
		end

		@testset "KSRobinBC" begin
			alpha = (x, t) -> 1.0
			beta = (x, t) -> 2.0
			value = (x, t) -> sin(π * x)
			boundary = x -> x == 0 || x == 1

			bc = KSRobinBC(alpha, beta, value, boundary)
			@test bc.alpha == alpha
			@test bc.beta == beta
			@test bc.value == value
			@test bc.boundary == boundary

			@test_throws MethodError KSRobinBC("not a function", beta, value, boundary)
			@test_throws MethodError KSRobinBC(alpha, "not a function", value, boundary)
			@test_throws MethodError KSRobinBC(alpha, beta, "not a function", boundary)
			@test_throws MethodError KSRobinBC(alpha, beta, value, "not a function")
		end

		@testset "KSNeumannBC" begin
			flux = (x, t) -> cos(π * x)
			boundary = x -> x == 0 || x == 1

			bc = KSNeumannBC(flux, boundary)
			@test bc.flux == flux
			@test bc.boundary == boundary

			@test_throws MethodError KSNeumannBC("not a function", boundary)
			@test_throws MethodError KSNeumannBC(flux, "not a function")
		end
	end

	@testset "KSSolverOptions" begin
		@testset "Valid case" begin
			options = KSSolverOptions(100, 1e-6, true, 5, 0.1, 10, 3, false, 1, false, 0.01, 100, 0.1)
			@test options.max_iterations == 100
			@test options.tolerance == 1e-6
			@test options.adaptive == true
			@test options.max_levels == 5
			@test options.smoothness_threshold == 0.1
			@test options.initial_elements == 10
			@test options.initial_degree == 3
			@test options.use_domain_decomposition == false
			@test options.num_subdomains == 1
			@test options.use_strang_splitting == false
			@test options.dt == 0.01
			@test options.num_steps == 100
			@test options.legendre_threshold == 0.1
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSSolverOptions(0, 1e-6, true, 5, 0.1, 10, 3, false, 1, false, 0.01, 100, 0.1)
			@test_throws ArgumentError KSSolverOptions(100, 0.0, true, 5, 0.1, 10, 3, false, 1, false, 0.01, 100, 0.1)
			@test_throws ArgumentError KSSolverOptions(100, 1e-6, true, 0, 0.1, 10, 3, false, 1, false, 0.01, 100, 0.1)
			@test_throws ArgumentError KSSolverOptions(100, 1e-6, true, 5, 0.0, 10, 3, false, 1, false, 0.01, 100, 0.1)
			@test_throws ArgumentError KSSolverOptions(100, 1e-6, true, 5, 0.1, 0, 3, false, 1, false, 0.01, 100, 0.1)
			@test_throws ArgumentError KSSolverOptions(100, 1e-6, true, 5, 0.1, 10, 0, false, 1, false, 0.01, 100, 0.1)
			@test_throws ArgumentError KSSolverOptions(100, 1e-6, true, 5, 0.1, 10, 3, false, 1, false, 0.01, 100, 0.0)
		end
	end

	@testset "Linear Solvers" begin
		@testset "KSDirectSolver" begin
			solver = KSDirectSolver(:lu)
			@test solver.method == :lu
		end

		@testset "KSIterativeSolver" begin
			solver = KSIterativeSolver(:cg, 1000, 1e-6)
			@test solver.method == :cg
			@test solver.max_iter == 1000
			@test solver.tolerance == 1e-6
			@test solver.preconditioner === nothing

			precond = x -> x
			solver_with_precond = KSIterativeSolver(:gmres, 500, 1e-8, precond)
			@test solver_with_precond.method == :gmres
			@test solver_with_precond.max_iter == 500
			@test solver_with_precond.tolerance == 1e-8
			@test solver_with_precond.preconditioner == precond

			@test_throws ArgumentError KSIterativeSolver(:cg, 0, 1e-6)
			@test_throws ArgumentError KSIterativeSolver(:cg, 1000, 0.0)
		end

		@testset "KSAMGSolver" begin
			solver = KSAMGSolver(1000, 1e-6, :jacobi)
			@test solver.max_iter == 1000
			@test solver.tolerance == 1e-6
			@test solver.smoother == :jacobi

			@test_throws ArgumentError KSAMGSolver(0, 1e-6, :jacobi)
			@test_throws ArgumentError KSAMGSolver(1000, 0.0, :jacobi)
		end
	end

	@testset "Helper Functions" begin
		@testset "tuple_if_active" begin
			@test tuple_if_active((1, 2), nothing, (3, 4)) == ((1, 2), (3, 4))
			@test tuple_if_active(nothing, (5, 6), nothing) == ((5, 6),)
			@test tuple_if_active(nothing, nothing, nothing) == ()
			@test tuple_if_active((1, 2), (3, 4), (5, 6)) == ((1, 2), (3, 4), (5, 6))
		end

		@testset "is_callable" begin
			@test is_callable(x -> x^2) == true
			@test is_callable(sin) == true
			@test is_callable("not a function") == false
			@test is_callable(1) == false

			struct CallableStruct
				f::Function
			end
			(c::CallableStruct)(x) = c.f(x)
			callable_struct = CallableStruct(x -> 2x)
			@test is_callable(callable_struct) == false
		end
	end
end
