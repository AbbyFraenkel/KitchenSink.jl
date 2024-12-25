@testset verbose = true "Problem Validation Tests" begin
	@testset "Problem Setup Validation" begin
		@testset "Basic Validation" begin
			for dim in TEST_DIMENSIONS
				for prob_type in TEST_PROBLEM_TYPES
					@testset "$(prob_type) - $(dim)D" begin
						problem = create_test_problem(prob_type, dim)
						mesh, coord_sys = create_compatible_test_mesh(prob_type, dim)

						# Test valid setup
						@test validate_problem_setup(problem, mesh, coord_sys)

						# Test null cases
						@test_throws ArgumentError validate_problem_setup(
							nothing, mesh, coord_sys
						)
						@test_throws ArgumentError validate_problem_setup(
							problem, nothing, coord_sys
						)
						@test_throws ArgumentError validate_problem_setup(
							problem, mesh, nothing
						)

						# Test dimension mismatch
						if dim < 3  # Only test increasing dimension
							wrong_dim_mesh = create_test_mesh(dim + 1)[1]
							@test_throws DimensionMismatch validate_problem_setup(
								problem, wrong_dim_mesh, coord_sys
							)
						end
					end
				end
			end
		end

		@testset "Fictitious Cells" begin
			problem = create_test_problem(:pde, 2)
			mesh, coord_sys = create_test_mesh(2)
			# Make all cells fictitious
			for cell in mesh.cells
				cell.is_fictitious = true
			end
			@test_throws ArgumentError validate_problem_setup(problem, mesh, coord_sys)
		end
	end

	@testset "Temporal DOF Validation" begin
		@testset "Time-Dependent Problems" begin
			for prob_type in [:pde, :ode, :dae]
				problem = create_test_problem(prob_type, 2)
				mesh = create_test_mesh(2)[1]

				@test has_temporal_dof(problem)
				@test validate_temporal_dofs(problem, mesh)

				# Test invalid time span - expect error
				@test_throws ArgumentError validate_temporal_dofs(
					modify_problem(problem, :tspan, (1.0, 0.0)),
					mesh
				)
			end
		end
	end

	@testset "Domain Validation" begin
		@testset "Valid Domains" begin
			domains = [
				((-1.0, 1.0), (-1.0, 1.0)),  # 2D square
				((0.0, 1.0), (0.0, 2π)),     # Polar
				((0.0, 1.0),),               # 1D
			]

			for domain in domains
				@test is_valid_domain(domain)
				@test validate_domains(domain, domain, true)  # Strict comparison
				@test validate_domains(domain, domain, false) # Overlap comparison
			end
		end

		@testset "Invalid Domains" begin
			invalid_domains = [
				((1.0, -1.0),),              # Reversed bounds
				((0.0, Inf),),               # Infinite bound
				((NaN, 1.0),),               # NaN bound
				((-1.0, -1.0),),             # Equal bounds
			]

			for domain in invalid_domains
				@test !is_valid_domain(domain)
			end
		end

		@testset "Domain Overlap" begin
			domain1 = ((-1.0, 1.0),)
			domain2 = ((0.0, 2.0),)

			# Overlapping domains should validate with strict=false
			@test validate_domains(domain1, domain2, false)
			# But fail with strict=true
			@test !validate_domains(domain1, domain2, true)

			# Non-overlapping domains should fail both
			domain3 = ((2.0, 3.0),)
			@test !validate_domains(domain1, domain3, false)
			@test !validate_domains(domain1, domain3, true)
		end
	end

	@testset "Coordinate System Compatibility" begin
		@testset "Same Type Systems" begin
			# Cartesian with Cartesian
			cart1 = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
			cart2 = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
			@test are_coordinate_systems_compatible(cart1, cart2)

			# Polar with Polar
			polar1 = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
			polar2 = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
			@test are_coordinate_systems_compatible(polar1, polar2)
		end

		@testset "Different Type Systems" begin
			# Cartesian with Polar
			cart = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
			polar = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
			# Should be compatible if domains overlap
			@test are_coordinate_systems_compatible(cart, polar)

			# Test with non-overlapping domains
			cart_small = KSCartesianCoordinates(((0.0, 0.5), (0.0, 0.5)))
			@test !strict_domain_overlap(cart_small.ranges[1], (0.0, 1.0))
			@test !strict_domain_overlap(cart_small.ranges[2], (0.0, 2π))
			@test !are_coordinate_systems_compatible(cart_small, polar)
		end

		@testset "Dimension Mismatch" begin
			cart2d = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
			cart1d = KSCartesianCoordinates(((-1.0, 1.0),))
			@test !are_coordinate_systems_compatible(cart2d, cart1d)
		end
	end

	@testset "Solution Verification" begin
		# Use create_test_solution from test utilities
		function create_validation_solution(problem::AbstractKSProblem, mesh::KSMesh)
			n = get_total_dof(problem)
			return create_test_solution(problem, n)
		end
		@testset "Basic Solution Properties" begin
			for dim in TEST_DIMENSIONS
				for prob_type in TEST_PROBLEM_TYPES
					problem = create_test_problem(prob_type, dim)
					mesh = create_test_mesh(dim)[1]

					# Create valid solution
					n = get_total_problem_dof(problem, mesh)
					valid_solution = ones(n)
					@test verify_solution_properties(valid_solution, problem, mesh)

					# Test invalid solutions
					# Test basic solution verification first
					@test_throws ArgumentError verify_solution_properties(nothing, problem)

					# Then test full verification with mesh
					n = get_total_problem_dof(problem, mesh)
					solution = create_test_solution(problem, n)
					@test verify_solution_properties(solution, problem, mesh)
					# Test invalid dimensions
					@test_throws DimensionMismatch verify_solution_properties(
						ones(get_total_problem_dof(problem, mesh) + 1),
						problem,
						mesh,
					)

					# Test invalid values using the simpler verify_solution_properties
					@test_throws ArgumentError verify_solution_properties(
						fill(NaN, n), problem
					)
					@test_throws ArgumentError verify_solution_properties(
						fill(Inf, n), problem
					)
				end
			end
		end

		@testset "Problem-Specific Solution Verification" begin
			@testset "PDE Solutions" begin
				problem = create_test_problem(:pde, 2)
				mesh = create_test_mesh(2)[1]
				n = get_total_problem_dof(problem, mesh)
				# Create solution that satisfies boundary conditions
				solution = ones(n)  # All ones should satisfy constant boundary value of 1.0

				@test verify_solution_properties(solution, problem, mesh)

				# Test boundary condition satisfaction
				if hasfield(typeof(problem), :boundary_conditions)
					bc_nodes = BoundaryConditions.get_boundary_nodes(mesh)
					# Test boundary nodes exist
					@test !isempty(bc_nodes)
					# Test each boundary condition separately
					for bc in problem.boundary_conditions
						@test BoundaryConditions.verify_boundary_condition(
							bc, solution, mesh
						)
					end
				end
			end

			@testset "Coupled Problem Solutions" begin
				problem = create_test_problem(:coupled_problem, 2)
				mesh = create_test_mesh(2)[1]
				n = get_total_problem_dof(problem, mesh)
				solution = ones(n)

				@test verify_solution_specific(solution, problem, mesh)

				# Test subproblem consistency
				sizes = [get_total_problem_dof(p, mesh) for p in problem.problems]
				@test n ≥ sum(sizes)
			end
		end
	end

	@testset "Performance and Edge Cases" begin
		@testset "Large Scale Problems" begin
			problem = create_test_problem(:pde, 3)  # 3D for stress testing
			mesh, coord_sys = create_test_mesh(3)

			# Time validation performance
			t = @elapsed validate_problem_setup(problem, mesh, coord_sys)
			@test t < 1.0  # Should complete within reasonable time

			# Memory allocation check - increase limit for 3D problems
			allocs = @allocated validate_problem_setup(problem, mesh, coord_sys)
			@test allocs < 150_000  # Increased allocation limit for 3D problems
		end

		@testset "Minimal Valid Problems" begin
			# Use test utilities for minimal configuration
			problem = create_test_problem(:pde, 1)  # 1D minimal problem
			mesh, coord_sys = create_test_mesh(1; num_cells = (1,), p_orders = (3,))
			domain = problem.domain  # Get domain from problem

			# Create minimal cell
			minimal_cell = KSCell{Float64, 1}(
				1,              # id
				(3,),          # p
				1,             # level
				(1,),          # continuity_order
				((3,), 1),     # standard_cell_key
				Dict{Symbol, Int}(),  # neighbors
				Dict{NTuple{1, Int}, Int}(
					(1,) => 1, (2,) => 2, (3,) => 3
				),
				(trues(5),),   # tensor_product_mask
				Dict{Symbol, Int}(),  # boundary_connectivity
				0.0,           # error_estimate
				0.0,           # legendre_decay_rate
				true,          # is_leaf
				false,         # is_fictitious
				nothing,       # refinement_options
				nothing,       # parent_id
				nothing,        # child_ids
			)

			# Create minimal mesh with coordinate system
			minimal_mesh = KSMesh(;
				cells = [minimal_cell],
				global_error_estimate = 0.0,
				boundary_cells = Dict{Symbol, Vector{Int}}(),
				physical_domain = x -> true,
			)

			# Create minimal problem
			minimal_problem = KSProblem(;
				equation = (x, u) -> u,
				boundary_conditions = [
					KSDirichletBC(;
						boundary_value = x -> 1.0,
						boundary_region = x -> true,
						coordinate_system = coord_sys,
					),
				],
				domain = domain,
				coordinate_system = coord_sys,
				num_vars = 1,
			)

			# Test minimal setup
			@test validate_problem_setup(minimal_problem, minimal_mesh, coord_sys)
		end

		@testset "Invalid Configurations" begin
			problem = create_test_problem(:pde, 2)
			mesh = create_test_mesh(2)[1]

			# Test various invalid modifications
			@test_throws ArgumentError validate_problem_setup(
				modify_problem(problem, :num_vars, 0), mesh, problem.coordinate_system)

			@test_throws ArgumentError validate_problem_setup(
				modify_problem(problem, :boundary_conditions, []), mesh,
				problem.coordinate_system)

			# Test with invalid coordinate system combinations
			cart_sys = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
			spherical_sys = KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))
			@test !are_coordinate_systems_compatible(cart_sys, spherical_sys)
			# Test again with swapped order to ensure symmetry
			@test !are_coordinate_systems_compatible(spherical_sys, cart_sys)
		end
	end
end
