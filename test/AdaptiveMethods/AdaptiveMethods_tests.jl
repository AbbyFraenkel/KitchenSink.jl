using Test
using LinearAlgebra, SparseArrays, AlgebraicMultigrid

using KitchenSink.AdaptiveMethods
# Mock types and functions for testing
struct MockProblem <: AbstractKSProblem end

function create_mock_element(id::Int, dimension::Int, polynomial_degree::Int)
    points = [KSPoint([i/polynomial_degree for _ in 1:dimension]) for i in 0:polynomial_degree]
    basis_functions = [KSBasisFunction(i, x -> sum(x.^i)) for i in 1:polynomial_degree+1]
    collocation_points = [KSPoint([i/(polynomial_degree+1) for _ in 1:dimension]) for i in 1:polynomial_degree+1]
    diff_matrices = [rand(polynomial_degree+1, polynomial_degree+1) for _ in 1:dimension]

    KSElement(
        id,
        points,
        basis_functions,
        nothing,
        nothing,
        nothing,
        1,
        polynomial_degree,
        0.0,
        collocation_points,
        diff_matrices
    )
end

function create_mock_mesh(num_elements::Int, dimension::Int, polynomial_degree::Int)
    elements = [create_mock_element(i, dimension, polynomial_degree) for i in 1:num_elements]
    tensor_product_masks = [trues(tuple(fill(polynomial_degree + 1, dimension)...)) for _ in 1:num_elements]
    location_matrices = [Dict(i => i for i in 1:(polynomial_degree+1)^dimension) for _ in 1:num_elements]
    basis_functions = elements[1].basis_functions

    KSMesh(
        elements,
        tensor_product_masks,
        location_matrices,
        basis_functions,
        0.0,
        dimension
    )
end

@testset "AdaptiveMethods" begin
    @testset "h_refine" begin
        element = create_mock_element(1, 2, 3)
        refined_elements = AdaptiveMethods.h_refine(element)

        @test length(refined_elements) == 2^element.dimension
        @test all(e.level == element.level + 1 for e in refined_elements)
        @test all(e.polynomial_degree == element.polynomial_degree for e in refined_elements)
    end

    @testset "p_refine" begin
        element = create_mock_element(1, 2, 3)
        refined_element = AdaptiveMethods.p_refine(element)

        @test refined_element.polynomial_degree == element.polynomial_degree + 1
        @test length(refined_element.collocation_points) > length(element.collocation_points)
        @test refined_element.level == element.level
    end

    @testset "hp_refine" begin
        element = create_mock_element(1, 2, 3)

        # Test h-refinement case
        refined_elements_h = AdaptiveMethods.hp_refine(element, 1.1, 0.1, 1.0)
        @test length(refined_elements_h) == 2^element.dimension

        # Test p-refinement case
        refined_elements_p = AdaptiveMethods.hp_refine(element, 1.1, 2.0, 1.0)
        @test length(refined_elements_p) == 1
        @test refined_elements_p[1].polynomial_degree == element.polynomial_degree + 1

        # Test no refinement case
        refined_elements_none = AdaptiveMethods.hp_refine(element, 0.9, 0.1, 1.0)
        @test length(refined_elements_none) == 1
        @test refined_elements_none[1] == element
    end

    @testset "select_subelement_nodes" begin
        points = [KSPoint([0.0, 0.0]), KSPoint([1.0, 0.0]), KSPoint([0.0, 1.0]), KSPoint([1.0, 1.0])]
        mid_points = [[0.5, 0.5]]

        selected_nodes = AdaptiveMethods.select_subelement_nodes(points, mid_points, 1)
        @test length(selected_nodes) == 1
        @test selected_nodes[1].coordinates == [0.0, 0.0]

        selected_nodes = AdaptiveMethods.select_subelement_nodes(points, mid_points, 4)
        @test length(selected_nodes) == 1
        @test selected_nodes[1].coordinates == [1.0, 1.0]
    end

    @testset "hp_refine_superposition" begin
        element = create_mock_element(1, 2, 3)

        refined_elements = AdaptiveMethods.hp_refine_superposition(element, 1.1, 0.1, 1.0)
        @test length(refined_elements) == 2^element.dimension

        refined_elements = AdaptiveMethods.hp_refine_superposition(element, 1.1, 2.0, 1.0)
        @test length(refined_elements) == 1
        @test refined_elements[1].polynomial_degree == element.polynomial_degree + 1

        refined_elements = AdaptiveMethods.hp_refine_superposition(element, 0.9, 0.1, 1.0)
        @test length(refined_elements) == 1
        @test refined_elements[1] == element
    end

    @testset "superconvergence_refinement" begin
        element = create_mock_element(1, 2, 3)
        solution = rand(length(element.collocation_points))
        problem = MockProblem()

        # Mock the estimate_error_and_smoothness function
        AdaptiveMethods.estimate_error_and_smoothness(::KSElement, ::Vector{Float64}, ::MockProblem) = (1.0, 0.1)

        refined_elements = AdaptiveMethods.superconvergence_refinement(element, solution, problem)
        @test length(refined_elements) == 2^element.dimension

        # Change the mock to return high smoothness
        AdaptiveMethods.estimate_error_and_smoothness(::KSElement, ::Vector{Float64}, ::MockProblem) = (1.0, 2.0)

        refined_elements = AdaptiveMethods.superconvergence_refinement(element, solution, problem)
        @test length(refined_elements) == 1
        @test refined_elements[1].polynomial_degree == element.polynomial_degree + 1
    end

    @testset "adapt_mesh_superposition" begin
        mesh = create_mock_mesh(4, 2, 3)
        solution = rand(sum(length(e.collocation_points) for e in mesh.elements))
        problem = MockProblem()

        # Mock the superconvergence_refinement function
        AdaptiveMethods.superconvergence_refinement(::KSElement, ::Vector{Float64}, ::MockProblem) = [create_mock_element(1, 2, 4)]

        adapted_mesh = AdaptiveMethods.adapt_mesh_superposition(mesh, solution, problem, 1.0)

        @test length(adapted_mesh.elements) == length(mesh.elements)
        @test all(e.polynomial_degree == 4 for e in adapted_mesh.elements)
    end

    @testset "estimate_error_and_smoothness" begin
        element = create_mock_element(1, 2, 3)
        solution = rand(length(element.collocation_points))
        problem = MockProblem()

        # Mock the CommonMethods functions
        CommonMethods.estimate_error(::KSElement, ::Vector{Float64}, ::MockProblem) = 1.0
        CommonMethods.estimate_smoothness(::KSElement, ::Vector{Float64}) = 0.5

        error, smoothness = AdaptiveMethods.estimate_error_and_smoothness(element, solution, problem)

        @test error ≈ 1.0
        @test smoothness ≈ 0.5
    end

    @testset "compute_error_indicator" begin
        element = create_mock_element(1, 2, 3)
        solution = rand(length(element.collocation_points))
        problem = MockProblem()

        # Mock the estimate_error_and_smoothness function
        AdaptiveMethods.estimate_error_and_smoothness(::KSElement, ::Vector{Float64}, ::MockProblem) = (1.0, 0.5)

        error, smoothness = AdaptiveMethods.compute_error_indicator(element, solution, problem)

        @test error ≈ 1.0
        @test smoothness ≈ 0.5
    end
end
@testset "AdaptiveMethods Tests" begin
	function create_test_mesh()
		domain = [(0.0, 1.0), (0.0, 1.0)]
		coord_system = KSCartesianCoordinates(2)
		num_elements = [2, 2]
		polynomial_degree = 2
		return create_mesh(domain, coord_system, num_elements, polynomial_degree)
	end

	@testset "h_refine" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]
		refined_elements = h_refine(element)

		@test length(refined_elements) == 4  # 2^2 for 2D
		@test all(e.level == element.level + 1 for e in refined_elements)
		@test all(e.polynomial_degree == element.polynomial_degree for e in refined_elements)
	end

	@testset "p_refine" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]
		refined_element = p_refine(element)

		@test refined_element.polynomial_degree == element.polynomial_degree + 1
		@test length(refined_element.collocation_points) > length(element.collocation_points)
	end

	@testset "hp_refine" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]

		# Test h-refinement case
		refined_elements_h = hp_refine(element, 1.0, 0.1, 0.5)
		@test length(refined_elements_h) == 4

		# Test p-refinement case
		refined_elements_p = hp_refine(element, 1.0, 0.9, 0.5)
		@test length(refined_elements_p) == 1
		@test refined_elements_p[1].polynomial_degree == element.polynomial_degree + 1
	end

	@testset "select_subelement_nodes" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]
		mid_points = [(node1.coordinates .+ node2.coordinates) ./ 2 for node1 in element.points for node2 in element.points if node1 != node2]
		unique_mid_points = unique(mid_points)

		subelement_nodes = select_subelement_nodes(element.points, unique_mid_points, 1)
		@test length(subelement_nodes) == 4  # For 2D element
		@test all(n -> n in element.points, subelement_nodes)
	end

	@testset "hp_refine_superposition" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]

		refined_elements = hp_refine_superposition(element, 1.0, 0.5, 0.5)
		@test length(refined_elements) > 0
		@test all(e -> e isa KSElement, refined_elements)
	end

	@testset "superconvergence_refinement" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]

		# Example of creating a symmetric sparse matrix
		A = Symmetric(sparse(rand(10, 10)))

		# Extract the underlying sparse matrix
		sparse_A = A.data

		# Pass the extracted sparse matrix to amg_preconditioner
		preconditioner = amg_preconditioner(sparse_A)

		@test preconditioner isa AMGPreconditioner
	end

	@testset "adapt_mesh_superposition" begin
		mesh = create_test_mesh()
		solution = ones(total_dofs(mesh))
		problem = KSPDEProblem((u, x, t) -> -Δu + u, (x, t) -> 0.0, (0.0, 1.0), [(0.0, 1.0), (0.0, 1.0)], x -> sin(π * x[1]) * sin(π * x[2]))

		adapted_mesh = adapt_mesh_superposition(mesh, solution, problem, 0.1)
		@test adapted_mesh isa KSMesh
		@test length(adapted_mesh.elements) >= length(mesh.elements)
	end

	@testset "estimate_error_and_smoothness" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]
		solution = ones(length(element.collocation_points))
		problem = KSPDEProblem((u, x, t) -> -Δu + u, (x, t) -> 0.0, (0.0, 1.0), [(0.0, 1.0), (0.0, 1.0)], x -> sin(π * x[1]) * sin(π * x[2]))

		error, smoothness = estimate_error_and_smoothness(element, solution, problem)
		@test error isa Real
		@test smoothness isa Real
		@test error >= 0
	end

	@testset "compute_error_indicator" begin
		mesh = create_test_mesh()
		element = mesh.elements[1]
		solution = ones(length(element.collocation_points))
		problem = KSPDEProblem((u, x, t) -> -Δu + u, (x, t) -> 0.0, (0.0, 1.0), [(0.0, 1.0), (0.0, 1.0)], x -> sin(π * x[1]) * sin(π * x[2]))

		error, smoothness = compute_error_indicator(element, solution, problem)
		@test error isa Real
		@test smoothness isa Real
		@test error >= 0
	end
end
