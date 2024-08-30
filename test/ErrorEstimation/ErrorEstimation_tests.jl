
using Test
using LinearAlgebra
# using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..ErrorEstimation, ..CommonMethods
# using ..ErrorEstimation

using KitchenSink.ErrorEstimation

@testset "ErrorEstimation Tests 2.0" begin
    # Helper functions
    function create_test_mesh()
        domain = [(0.0, 1.0), (0.0, 1.0)]
        coord_system = KSCartesianCoordinates(domain)
        num_elements = [2, 2]
        polynomial_degree = 2
        return CommonMethods.create_mesh(domain, num_elements, polynomial_degree, coord_system)
    end

    function create_test_problem()
        return KSProblem(
            (x, derivatives) -> -sum(derivatives) + derivatives[1][1],  # -Δu + u
            [(0.0, 1.0), (0.0, 1.0)],
            (x) -> 0.0
        )
    end

    function exact_solution(x)
        return sin(π * x[1]) * sin(π * x[2])
    end

    @testset "estimate_error" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]
        problem = create_test_problem()

        error = ErrorEstimation.estimate_error(element, solution, problem)
        @test error isa Real
        @test error >= 0
        @test error < 1e-2  # The error should be small for the exact solution

        # Test with a constant solution (should have non-zero error)
        constant_solution = ones(length(element.collocation_points))
        constant_error = ErrorEstimation.estimate_error(element, constant_solution, problem)
        @test constant_error > error  # Constant solution should have larger error

        # Test with different types
        solution_f32 = Float32.(solution)
        error_f32 = ErrorEstimation.estimate_error(element, solution_f32, problem)
        @test error_f32 isa Float32
        @test error_f32 >= 0
        @test isapprox(error, error_f32, rtol=1e-5)
    end

    @testset "compute_residual" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]
        problem = create_test_problem()

        residual = ErrorEstimation.compute_residual(element, solution, problem)
        @test length(residual) == length(solution)
        @test norm(residual) < 1e-10  # Residual should be very small for the exact solution

        # Test with a constant solution (should have non-zero residual)
        constant_solution = ones(length(element.collocation_points))
        constant_residual = ErrorEstimation.compute_residual(element, constant_solution, problem)
        @test norm(constant_residual) > norm(residual)

        # Test with different types
        solution_f32 = Float32.(solution)
        residual_f32 = ErrorEstimation.compute_residual(element, solution_f32, problem)
        @test eltype(residual_f32) == Float32
        @test length(residual_f32) == length(solution_f32)
        @test isapprox(norm(residual), norm(residual_f32), rtol=1e-5)
    end

    @testset "compute_error_indicator" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]
        problem = create_test_problem()

        error, smoothness = ErrorEstimation.compute_error_indicator(element, solution, problem)
        @test error isa Real
        @test smoothness isa Real
        @test error >= 0
        @test error < 1e-2  # Error should be small for the exact solution
        @test smoothness > 0  # Smoothness should be positive for a smooth solution

        # Test with a constant solution
        constant_solution = ones(length(element.collocation_points))
        constant_error, constant_smoothness = ErrorEstimation.compute_error_indicator(element, constant_solution, problem)
        @test constant_error > error  # Constant solution should have larger error
        @test constant_smoothness > smoothness  # Constant solution should be smoother

        # Test with different types
        solution_f32 = Float32.(solution)
        error_f32, smoothness_f32 = ErrorEstimation.compute_error_indicator(element, solution_f32, problem)
        @test error_f32 isa Float32
        @test smoothness_f32 isa Float32
        @test isapprox(error, error_f32, rtol=1e-5)
        @test isapprox(smoothness, smoothness_f32, rtol=1e-5)
    end

    @testset "compute_smoothness_indicator" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]

        smoothness = ErrorEstimation.compute_smoothness_indicator(element, solution)
        @test smoothness isa Real
        @test smoothness > 0  # Smoothness should be positive for a smooth solution

        # Test with a constant solution (should be very smooth)
        constant_solution = ones(length(element.collocation_points))
        constant_smoothness = ErrorEstimation.compute_smoothness_indicator(element, constant_solution)
        @test constant_smoothness > smoothness

        # Test with a highly oscillatory solution (should be less smooth)
        oscillatory_solution = [sin(10π * p.coordinates[1]) * sin(10π * p.coordinates[2]) for p in element.collocation_points]
        oscillatory_smoothness = ErrorEstimation.compute_smoothness_indicator(element, oscillatory_solution)
        @test oscillatory_smoothness < smoothness

        # Test with different types
        solution_f32 = Float32.(solution)
        smoothness_f32 = ErrorEstimation.compute_smoothness_indicator(element, solution_f32)
        @test smoothness_f32 isa Float32
        @test isapprox(smoothness, smoothness_f32, rtol=1e-5)
    end

    @testset "apply_differential_operator" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]
        problem = create_test_problem()

        result = ErrorEstimation.apply_differential_operator(solution, element, problem)
        @test length(result) == length(solution)

        # For the exact solution u = sin(πx)sin(πy), -Δu + u = (2π^2 + 1)u
        expected_result = [(2π^2 + 1) * s for s in solution]
        @test isapprox(result, expected_result, rtol=1e-5)

        # Test with a constant solution
        constant_solution = ones(length(element.collocation_points))
        constant_result = ErrorEstimation.apply_differential_operator(constant_solution, element, problem)
        @test all(isapprox.(constant_result, 1.0, atol=1e-10))

        # Test with different types
        solution_f32 = Float32.(solution)
        result_f32 = ErrorEstimation.apply_differential_operator(solution_f32, element, problem)
        @test eltype(result_f32) == Float32
        @test length(result_f32) == length(solution_f32)
        @test isapprox(result, result_f32, rtol=1e-5)
    end

    @testset "estimate_smoothness" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]

        smoothness = ErrorEstimation.estimate_smoothness(element, solution)
        @test smoothness isa Real
        @test smoothness > 0  # Smoothness should be positive for a smooth solution

        # Test with a constant solution (should be very smooth)
        constant_solution = ones(CommonMethods.element_dofs(element))
        constant_smoothness = ErrorEstimation.estimate_smoothness(element, constant_solution)
        @test constant_smoothness > smoothness

        # Test with a highly oscillatory solution (should be less smooth)
        oscillatory_solution = [sin(10π * p.coordinates[1]) * sin(10π * p.coordinates[2]) for p in element.collocation_points]
        oscillatory_smoothness = ErrorEstimation.estimate_smoothness(element, oscillatory_solution)
        @test oscillatory_smoothness < smoothness

        # Test error handling
        @test_throws DimensionMismatch ErrorEstimation.estimate_smoothness(element, ones(CommonMethods.element_dofs(element) + 1))

        # Test with different types
        solution_f32 = Float32.(solution)
        smoothness_f32 = ErrorEstimation.estimate_smoothness(element, solution_f32)
        @test smoothness_f32 isa Float32
        @test isapprox(smoothness, smoothness_f32, rtol=1e-5)
    end

    @testset "compute_expansion_coefficients" begin
        mesh = create_test_mesh()
        element = mesh.elements[1]
        solution = [exact_solution(p.coordinates) for p in element.collocation_points]

        coeffs = ErrorEstimation.compute_expansion_coefficients(element, solution)
        @test coeffs isa Vector{Float64}
        @test length(coeffs) == length(element.basis_functions)
        @test norm(coeffs) > 0  # Coefficients should not be all zero for a non-trivial solution

        # Test reconstruction
        reconstructed_solution = [sum(c * bf.function_handle(p.coordinates) for (c, bf) in zip(coeffs, element.basis_functions)) for p in element.collocation_points]
        @test isapprox(solution, reconstructed_solution, rtol=1e-5)

        # Test error handling
        @test_throws ArgumentError ErrorEstimation.compute_expansion_coefficients(element, solution, lambda = -1.0)
        @test_throws DimensionMismatch ErrorEstimation.compute_expansion_coefficients(element, ones(length(solution) + 1))

        # Test with different types
        solution_f32 = Float32.(solution)
        coeffs_f32 = ErrorEstimation.compute_expansion_coefficients(element, solution_f32)
        @test coeffs_f32 isa Vector{Float32}
        @test length(coeffs_f32) == length(element.basis_functions)
        @test isapprox(coeffs, coeffs_f32, rtol=1e-5)

        # Test with near-singular matrix
        element_singular = deepcopy(element)
        element_singular.basis_functions = [KSBasisFunction(1, x -> 1.0) for _ in 1:length(element_singular.basis_functions)]
        @test_throws LinearAlgebra.SingularException ErrorEstimation.compute_expansion_coefficients(element_singular, solution)
    end
end
