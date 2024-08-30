using Test
using LinearAlgebra
using KitchenSink.BoundaryConditions

@testset "BoundaryConditions Tests" begin

	# Helper function to create a simple test mesh
	function create_test_mesh()
		domain = [(0.0, 1.0), (0.0, 1.0)]
		coord_system = KSCartesianCoordinates(2)
		num_elements = [2, 2]
		polynomial_degree = 2
		return create_mesh(domain, coord_system, num_elements, polynomial_degree)
	end

	@testset "apply_boundary_condition!" begin
		@testset "Valid cases" begin
			mesh = create_test_mesh()
			A = spzeros(total_dofs(mesh), total_dofs(mesh))
			b = zeros(total_dofs(mesh))
			bc = KSDirichletBC(x -> sin(π * x[1]) * sin(π * x[2]), x -> true)

			apply_boundary_condition!(A, b, mesh, bc)
			@test !all(iszero, diag(A))  # Some diagonal elements should be non-zero
			@test !all(iszero, b)  # Some elements of b should be non-zero
		end

		@testset "Edge cases" begin
			mesh = create_test_mesh()
			A = spzeros(total_dofs(mesh), total_dofs(mesh))
			b = zeros(total_dofs(mesh))
			bc = KSDirichletBC(x -> 0.0, x -> true)  # Zero boundary condition

			apply_boundary_condition!(A, b, mesh, bc)
			@test !all(iszero, diag(A))  # Some diagonal elements should be non-zero
			@test all(iszero, b)  # All elements of b should be zero
		end

		@testset "Complex cases" begin
			mesh = create_test_mesh()
			A = spzeros(total_dofs(mesh), total_dofs(mesh))
			b = zeros(total_dofs(mesh))
			bc = KSNeumannBC(x -> x[1] + x[2], x -> true)

			apply_boundary_condition!(A, b, mesh, bc)
			@test !all(iszero, A)  # Some elements of A should be non-zero
			@test !all(iszero, b)  # Some elements of b should be non-zero
		end
	end

	@testset "apply_bc_at_point!" begin
		@testset "Valid cases - Dirichlet" begin
			mesh = create_test_mesh()
			A = spzeros(total_dofs(mesh), total_dofs(mesh))
			b = zeros(total_dofs(mesh))
			point = KSPoint([0.0, 0.0])
			bc = KSDirichletBC(x -> 1.0, x -> true)

			apply_bc_at_point!(A, b, 1, point, bc)
			@test A[1, 1] ≈ 1.0
			@test all(iszero, A[1, 2:end])
			@test b[1] ≈ 1.0
		end

		@testset "Edge cases - Neumann" begin
			mesh = create_test_mesh()
			A = spzeros(total_dofs(mesh), total_dofs(mesh))
			b = zeros(total_dofs(mesh))
			point = KSPoint([0.0, 0.0])
			bc = KSNeumannBC(x -> 0.0, x -> true)  # Zero flux condition

			apply_bc_at_point!(A, b, 1, point, bc)
			@test !all(iszero, A[1, :])  # Some elements in the first row should be non-zero
			@test iszero(b[1])
		end

		@testset "Complex cases - Robin" begin
			mesh = create_test_mesh()
			A = spzeros(total_dofs(mesh), total_dofs(mesh))
			b = zeros(total_dofs(mesh))
			point = KSPoint([0.0, 0.0])
			bc = KSRobinBC(x -> 1.0, x -> 2.0, x -> 3.0, x -> true)

			apply_bc_at_point!(A, b, 1, point, bc)
			@test !all(iszero, A[1, :])
			@test !iszero(b[1])
		end

		@testset "is_boundary_point" begin
			mesh = create_test_mesh()

			@test is_boundary_point(KSPoint([0.0, 0.0]), mesh)
			@test is_boundary_point(KSPoint([1.0, 1.0]), mesh)
			@test is_boundary_point(KSPoint([0.5, 0.0]), mesh)
			@test !is_boundary_point(KSPoint([0.5, 0.5]), mesh)
		end

		@testset "compute_normal_vector" begin
			mesh = create_test_mesh()

			@test compute_normal_vector(KSPoint([0.0, 0.5]), mesh) ≈ [-1.0, 0.0]
			@test compute_normal_vector(KSPoint([1.0, 0.5]), mesh) ≈ [1.0, 0.0]
			@test compute_normal_vector(KSPoint([0.5, 0.0]), mesh) ≈ [0.0, -1.0]
			@test compute_normal_vector(KSPoint([0.5, 1.0]), mesh) ≈ [0.0, 1.0]
		end

		@testset "gradient_operator" begin
			mesh = create_test_mesh()
			point = KSPoint([0.5, 0.5])

			grad_op = gradient_operator(point, mesh)
			@test size(grad_op) == (2, total_dofs(mesh))
			@test !all(iszero, grad_op)
		end
	end
end
