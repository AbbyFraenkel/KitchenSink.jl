@testset "Error Estimation and Indicators" begin
    @testset "estimate_error" begin
        element = create_test_element_1D()
        solution = [1.0, 2.0]
        problem, f, u_exact = create_test_problem(1, :cartesian)

        @testset "Valid Inputs" begin
            estimated_error = estimate_error(element, solution, problem)
            @test estimated_error >= 0
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError estimate_error(element, solution, "invalid_problem")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError estimate_error(empty_element, solution, problem)
        end
    end

    @testset "compute_residual" begin
        element = create_test_element_1D()
        solution = [1.0, 2.0]
        problem, f, u_exact = create_test_problem(1, :cartesian)

        @testset "Valid Inputs" begin
            residual = compute_residual(element, solution, problem)
            expected_residual = [-1.0, -2.0]  # Updated expected result based on the problem definition
            @test isapprox(residual, expected_residual, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError compute_residual(element, solution, "invalid_problem")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError compute_residual(empty_element, solution, problem)
        end
    end

    @testset "compute_error_indicator" begin
        element = create_test_element_1D()
        solution = [1.0, 2.0]
        problem, f, u_exact = create_test_problem(1, :cartesian)

        @testset "Valid Inputs" begin
            error, smoothness = compute_error_indicator(element, solution, problem)
            @test error >= 0
            @test smoothness >= 0
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError compute_error_indicator(element, solution, "invalid_problem")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError compute_error_indicator(empty_element, solution, problem)
        end
    end

    @testset "apply_differential_operator" begin
        element = create_test_element_1D()
        solution = [1.0, 2.0]
        problem, f, u_exact = create_test_problem(1, :cartesian)

        @testset "Valid Inputs" begin
            diff_op_result = apply_differential_operator(solution, element, problem)
            expected_result = -1.0 .* solution  # Updated expected result based on the problem definition
            @test isapprox(diff_op_result, expected_result, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError apply_differential_operator(solution, element, "invalid_problem")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError apply_differential_operator(solution, empty_element, problem)
        end
    end

    @testset "compute_expansion_coefficients" begin
        element = create_test_element_1D()
        solution = [1.0, 2.0]

        @testset "Valid Inputs" begin
            coeffs = compute_expansion_coefficients(element, solution)
            @test length(coeffs) == length(element.basis_functions)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError compute_expansion_coefficients("invalid_element", solution)
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError compute_expansion_coefficients(empty_element, solution)
        end
    end
end
