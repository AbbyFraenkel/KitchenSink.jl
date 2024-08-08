@testset "Mesh Creation" begin
    @testset "create_mesh" begin
        problem = create_test_problem(2)
        mesh = create_mesh(problem.domain, problem.coordinate_system, (5, 5), 3)

        @test mesh isa KSMesh{Float64,2}
        @test length(mesh.elements) == 25
        @test all(e -> e.polynomial_degree == 3, mesh.elements)

        @test_throws ArgumentError create_mesh(problem.domain, problem.coordinate_system, (0, 5), 3)
        @test_throws ArgumentError create_mesh(problem.domain, problem.coordinate_system, (5, 5), 0)
    end

    @testset "create_nodes" begin
        problem = create_test_problem(2)
        nodes = create_nodes(problem.domain, problem.coordinate_system, (5, 5))

        @test length(nodes) == 36  # (5+1) * (5+1)
        @test all(n -> n isa KSPoint{Float64}, nodes)
        @test all(n -> all(0 .<= n.coordinates .<= 1), nodes)

        @test_throws ArgumentError create_nodes(problem.domain, problem.coordinate_system, (0, 5))
    end

    @testset "create_elements" begin
        problem = create_test_problem(2)
        nodes = create_nodes(problem.domain, problem.coordinate_system, (5, 5))
        elements = create_elements(nodes, (5, 5), 3, problem.coordinate_system)

        @test length(elements) == 25
        @test all(e -> e isa KSElement{Float64,2}, elements)
        @test all(e -> e.polynomial_degree == 3, elements)

        @test_throws ArgumentError create_elements(KSPoint{Float64}[], (5, 5), 3, problem.coordinate_system)
        @test_throws ArgumentError create_elements(nodes, (5, 5), 0, problem.coordinate_system)
    end

    @testset "create_collocation_points" begin
        problem = create_test_problem(2)
        nodes = create_nodes(problem.domain, problem.coordinate_system, (5, 5))
        collocation_points = create_collocation_points(nodes[1:4], 3, problem.coordinate_system)

        @test collocation_points isa KSPoints{Float64}
        @test length(collocation_points.points) == 16  # 4^2 for 2D and polynomial degree 3
        @test length(collocation_points.weights) == 16

        @test_throws ArgumentError create_collocation_points(KSPoint{Float64}[], 3, problem.coordinate_system)
        @test_throws ArgumentError create_collocation_points(nodes[1:4], 0, problem.coordinate_system)
    end
end
