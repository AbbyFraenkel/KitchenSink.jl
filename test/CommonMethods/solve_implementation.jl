@testset "Solve Implementation Tests" begin
    @testset "solve_equation" begin
        mesh = create_test_mesh_1D()
        problem = KSProblem((u, x, ∇u, ∇²u) -> ∇²u - u, ((0.0, 1.0),), KSDirichletBC(x -> 0.0, x -> true), x -> sin(pi * x[1]), nothing)

        @testset "Valid Inputs" begin
            solution = solve_equation(problem, mesh)
            expected_solution = [0.0, 0.0]  # Example expected result for a simple mesh and problem
            @test length(solution) == length(mesh.elements)
            @test isapprox(solution, expected_solution, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError solve_equation(problem, "invalid_mesh")
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError solve_equation(problem, empty_mesh)
        end
    end

    @testset "solve_multi_level" begin
        mesh = create_test_mesh_1D()
        hierarchy = [mesh]
        f = x -> sin(pi * x[1])

        @testset "Valid Inputs" begin
            solution = solve_multi_level(hierarchy, f, 1)
            expected_solution = [0.0, 0.0]  # Example expected result for a simple hierarchy and source function
            @test length(solution) == sum(length(el.basis_functions) for el in mesh.elements)
            @test isapprox(solution, expected_solution, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError solve_multi_level("invalid_hierarchy", f, 1)
        end

        @testset "Edge Cases" begin
            empty_hierarchy = [create_test_mesh_1D()]
            empty_hierarchy[1].elements = []
            @test_throws BoundsError solve_multi_level(empty_hierarchy, f, 1)
        end
    end

    @testset "converged" begin
        old_solution = [rand(2), rand(2)]
        new_solution = [rand(2), rand(2)]
        tolerance = 1e-6

        @testset "Valid Inputs" begin
            @test converged(old_solution, new_solution, tolerance) in [true, false]
        end

        @testset "Invalid Inputs" begin
            @test_throws ArgumentError converged(old_solution, [rand(3)], tolerance)
        end

        @testset "Edge Cases" begin
            @test converged(old_solution, old_solution, tolerance) == true
        end
    end

    @testset "initialize_solution" begin
        mesh = create_test_mesh_1D()

        @testset "Valid Inputs" begin
            solution = initialize_solution(mesh)
            expected_solution = [zeros(2)]  # Example expected result for initializing the solution
            @test length(solution) == length(mesh.elements)
            @test all(x -> all(y -> y == 0.0, x), solution)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError initialize_solution("invalid_mesh")
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test length(initialize_solution(empty_mesh)) == 0
        end
    end

    @testset "solve_coarse_problem" begin
        mesh = create_test_mesh_1D()
        f = x -> sin(pi * x[1])

        @testset "Valid Inputs" begin
            solution = solve_coarse_problem(mesh, f)
            expected_solution = [0.0, 0.0]  # Example expected result for a simple mesh and source function
            @test length(solution) == sum(length(el.basis_functions) for el in mesh.elements)
            @test isapprox(solution, expected_solution, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError solve_coarse_problem("invalid_mesh", f)
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError solve_coarse_problem(empty_mesh, f)
        end
    end

    @testset "smooth" begin
        mesh = create_test_mesh_1D()
        f = x -> sin(pi * x[1])
        u = [0.0, 0.0]

        @testset "Valid Inputs" begin
            smoothed_u = smooth(mesh, f, u)
            expected_smoothed_u = [0.0, 0.0]  # Example expected result for a simple smoothing operation
            @test length(smoothed_u) == length(u)
            @test isapprox(smoothed_u, expected_smoothed_u, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError smooth("invalid_mesh", f, u)
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError smooth(empty_mesh, f, u)
        end
    end

    @testset "gauss_seidel_iteration" begin
        mesh = create_test_mesh_1D()
        f = x -> sin(pi * x[1])
        u = [0.0, 0.0]

        @testset "Valid Inputs" begin
            new_u = gauss_seidel_iteration(mesh, f, u)
            expected_new_u = [0.0, 0.0]  # Example expected result for a simple Gauss-Seidel iteration
            @test length(new_u) == length(u)
            @test isapprox(new_u, expected_new_u, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError gauss_seidel_iteration("invalid_mesh", f, u)
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError gauss_seidel_iteration(empty_mesh, f, u)
        end
    end

    @testset "apply_boundary_conditions!" begin
        solution = [0.0, 0.0]
        problem = KSProblem((u, x, ∇u, ∇²u) -> ∇²u - u, ((0.0, 1.0),), KSDirichletBC(x -> 1.0, x -> x[1] == 0.0 || x[1] == 1.0), x -> sin(pi * x[1]), nothing)
        element = create_test_element_1D()
        mask = [true, true]

        @testset "Valid Inputs" begin
            apply_boundary_conditions!(solution, problem, element, mask)
            expected_solution = [1.0, 1.0]  # Example expected result for applying boundary conditions
            @test isapprox(solution, expected_solution, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError apply_boundary_conditions!(solution, problem, "invalid_element", mask)
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError apply_boundary_conditions!(solution, problem, empty_element, mask)
        end
    end

    @testset "enforce_continuity!" begin
        mesh = create_test_mesh_1D()
        solution = [zeros(2)]

        @testset "Valid Inputs" begin
            enforce_continuity!(solution, mesh)
            expected_solution = [zeros(2)]  # Example expected result for enforcing continuity
            @test length(solution) == length(mesh.elements)
            @test all(x -> all(y -> y == 0.0, x), solution)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError enforce_continuity!("invalid_solution", mesh)
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError enforce_continuity!(solution, empty_mesh)
        end
    end

    @testset "is_boundary_node" begin
        element = create_test_element_1D()
        point = element.points[1]

        @testset "Valid Inputs" begin
            @test is_boundary_node(point, element) == true
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError is_boundary_node("invalid_point", element)
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError is_boundary_node(point, empty_element)
        end
    end
end
