using Test
using LinearAlgebra, SparseArrays, StaticArrays
# using ..KSTypes, ..CoordinateSystems, ..SpectralMethods, ..CommonMethods, ..Preprocessing
using KitchenSink.Preprocessing

@testset "Preprocessing Tests" begin

    # Helper function to create a simple test problem
    function create_test_problem()
        pde(u, x, t) = -Δu + u
        bc(x, t) = 0.0
        domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
        tspan = (0.0, 1.0)
        initial_condition(x) = sin(π * x[1]) * sin(π * x[2])
        return KSPDEProblem(pde, bc, tspan, domain, initial_condition)
    end

    # Helper function to create a complex test problem
    function create_complex_problem()
        pde(u, x, t) = -Δ(Δu) + u^2
        bc(x, t) = 0.0
        domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
        tspan = (0.0, 1.0)
        initial_condition(x) = exp(-10 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2))
        return KSPDEProblem(pde, bc, tspan, domain, initial_condition)
    end

    @testset "preprocess_mesh" begin
        @testset "Valid cases" begin
            problem = create_test_problem()
            domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
            coord_system = KSCartesianCoordinates(2)  # 2D coordinate system
            num_elements = [2, 2]
            polynomial_degree = 3
            max_levels = 3
            tolerance = 1e-6

            mesh = preprocess_mesh(problem, domain, coord_system, num_elements, polynomial_degree, max_levels, tolerance)
            @test mesh isa KSMesh
            @test length(mesh.elements) >= prod(num_elements)
            @test mesh.dimensions == 2  # Check 2D consistency
        end

        @testset "Edge cases" begin
            problem = create_test_problem()
            domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
            coord_system = KSCartesianCoordinates(2)  # 2D coordinate system
            num_elements = [1, 1]
            polynomial_degree = 3
            max_levels = 1
            tolerance = 1e-2

            mesh = preprocess_mesh(problem, domain, coord_system, num_elements, polynomial_degree, max_levels, tolerance)
            @test mesh isa KSMesh
            @test length(mesh.elements) == 1
        end

        @testset "Complex cases" begin
            problem = create_complex_problem()
            domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
            coord_system = KSCartesianCoordinates(2)  # 2D coordinate system
            num_elements = [4, 4]
            polynomial_degree = 4
            max_levels = 5
            tolerance = 1e-8

            mesh = preprocess_mesh(problem, domain, coord_system, num_elements, polynomial_degree, max_levels, tolerance)
            @test mesh isa KSMesh
            @test length(mesh.elements) > prod(num_elements)
        end

        @testset "Invalid cases" begin
            problem = create_test_problem()

            @test_throws DimensionMismatch preprocess_mesh(problem, [(1.0, 0.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 3, 3, 1e-6)
            @test_throws DimensionMismatch preprocess_mesh(problem, [(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(1), [2, 2], 3, 3, 1e-6)
        end
    end

    @testset "generate_initial_mesh" begin
        @testset "Valid cases" begin
            domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
            coord_system = KSCartesianCoordinates(2)  # 2D coordinate system
            num_elements = [2, 2]
            polynomial_degree = 3

            mesh = generate_initial_mesh(domain, coord_system, num_elements, polynomial_degree)
            @test mesh isa KSMesh
            @test length(mesh.elements) == prod(num_elements)
            @test all(el.polynomial_degree == polynomial_degree for el in mesh.elements)
        end

        @testset "Edge cases" begin
            domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
            coord_system = KSCartesianCoordinates(2)  # 2D coordinate system
            num_elements = [1, 1]
            polynomial_degree = 3

            mesh = generate_initial_mesh(domain, coord_system, num_elements, polynomial_degree)
            @test mesh isa KSMesh
            @test length(mesh.elements) == 1
            @test mesh.elements[1].polynomial_degree == 1
        end

        @testset "Complex cases" begin
            domain = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]  # 3D domain
            coord_system = KSCartesianCoordinates(3)  # 3D coordinate system
            num_elements = [2, 3, 4]
            polynomial_degree = 3

            mesh = generate_initial_mesh(domain, coord_system, num_elements, polynomial_degree)
            @test mesh isa KSMesh
            @test length(mesh.elements) == prod(num_elements)
            @test mesh.dimensions == 3  # Check 3D consistency
        end

        @testset "Invalid cases" begin
            domain = [(0.0, 1.0), (0.0, 1.0)]  # 2D domain
            coord_system = KSCartesianCoordinates(2)  # 2D coordinate system

            @test_throws DimensionMismatch generate_initial_mesh(domain, KSCartesianCoordinates(3), [2, 2], 3)
            @test_throws DimensionMismatch generate_initial_mesh([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], coord_system, [2, 2, 2], 3)
        end
    end

    @testset "refine_mesh" begin
        @testset "Valid cases" begin
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 2)
            error_estimates = [0.1, 0.2, 0.3, 0.4]
            tolerance = 0.25

            refined_mesh = refine_mesh(mesh, error_estimates, tolerance)
            @test refined_mesh isa KSMesh
            @test length(refined_mesh.elements) > length(mesh.elements)
        end

        @testset "Edge cases" begin
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [1, 1], 1)
            error_estimates = [0.1]
            tolerance = 0.2

            refined_mesh = refine_mesh(mesh, error_estimates, tolerance)
            @test refined_mesh isa KSMesh
            @test length(refined_mesh.elements) == length(mesh.elements)
        end

        @testset "Complex cases" begin
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(3), [2, 2, 2], 3)
            error_estimates = rand(8)
            tolerance = 0.5

            refined_mesh = refine_mesh(mesh, error_estimates, tolerance)
            @test refined_mesh isa KSMesh
            @test length(refined_mesh.elements) > length(mesh.elements)
            @test refined_mesh.dimensions == 3  # Check 3D consistency
        end

        @testset "Invalid cases" begin
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 2)

            @test_throws ArgumentError refine_mesh(mesh, [0.1, -0.2, 0.3, 0.4], 0.25)
            @test_throws ArgumentError refine_mesh(mesh, [0.1, 0.2], 0.25)
        end
    end

    @testset "estimate_mesh_error" begin
        @testset "Valid cases" begin
            problem = create_test_problem()
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 2)

            error_estimates = estimate_mesh_error(mesh, problem)
            @test length(error_estimates) == length(mesh.elements)
            @test all(e -> e >= 0, error_estimates)
        end

        @testset "Edge cases" begin
            problem = create_test_problem()
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [1, 1], 1)

            error_estimates = estimate_mesh_error(mesh, problem)
            @test length(error_estimates) == 1
            @test error_estimates[1] >= 0
        end

        @testset "Complex cases" begin
            # Create a problem with a more complex solution
            complex_problem = KSPDEProblem((u, x, t) -> -Δu + u^3 - u,
                                           (x, t) -> 0.0,
                                           (0.0, 1.0),
                                           [(0.0, 1.0), (0.0, 1.0)],
                                           x -> exp(-10 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2)))
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [4, 4], 3)

            error_estimates = estimate_mesh_error(mesh, complex_problem)
            @test length(error_estimates) == length(mesh.elements)
            @test any(e -> e > 0, error_estimates)  # Expect some non-zero errors
        end

        @testset "Invalid cases" begin
            problem = create_test_problem()
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 2)

            @test_throws ArgumentError estimate_mesh_error(mesh, KSPDEProblem(nothing, nothing, nothing, nothing, nothing))
        end
    end

    @testset "create_OCFE_discretization" begin
        @testset "Valid cases" begin
            problem = create_test_problem()
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 2)
            max_derivative_order = 2

            A = create_OCFE_discretization(mesh, problem, max_derivative_order)
            @test size(A, 1) == size(A, 2)
            @test size(A, 1) > 0
            @test issparse(A)
        end

        @testset "Edge cases" begin
            problem = create_test_problem()
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [1, 1], 1)
            max_derivative_order = 1

            A = create_OCFE_discretization(mesh, problem, max_derivative_order)
            @test size(A, 1) == size(A, 2)
            @test size(A, 1) == 3  # For 2D linear element
        end

        @testset "Complex cases" begin
            # Create a problem with higher-order derivatives
            complex_problem = KSPDEProblem((u, x, t) -> -Δ(Δu) + u,
                                           (x, t) -> 0.0,
                                           (0.0, 1.0),
                                           [(0.0, 1.0), (0.0, 1.0)],
                                           x -> sin(2π * x[1]) * sin(2π * x[2]))
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [3, 3], 4)
            max_derivative_order = 4

            A = create_OCFE_discretization(mesh, complex_problem, max_derivative_order)
            @test size(A, 1) == size(A, 2)
            @test size(A, 1) >= 0
            @test issparse(A)
        end

        @testset "Invalid cases" begin
            problem = create_test_problem()
            mesh = generate_initial_mesh([(0.0, 1.0), (0.0, 1.0)], KSCartesianCoordinates(2), [2, 2], 2)

            @test_throws ArgumentError create_OCFE_discretization(mesh, problem, -1)
            @test_throws ArgumentError create_OCFE_discretization(mesh, problem, 0)
        end
    end
end
