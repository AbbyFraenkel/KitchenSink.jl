@testset "Discretization" begin

    @testset "preprocess_mesh" begin
        problem, f, u_exact = create_test_problem(2, :cartesian)
        mesh = preprocess_mesh(problem, problem[:domain], problem[:coordinate_system], (5, 5), 3, 3, 1e-6)

        @test mesh isa KSMesh{Float64,2}
        @test length(mesh.elements) >= 25
        @test all(e -> e.polynomial_degree <= 3, mesh.elements)

        @test_throws ArgumentError preprocess_mesh(problem, problem[:domain], problem[:coordinate_system], (0, 5), 3, 3, 1e-6)
        @test_throws ArgumentError preprocess_mesh(problem, problem[:domain], problem[:coordinate_system], (5, 5), 0, 3, 1e-6)
        @test_throws ArgumentError preprocess_mesh(problem, problem[:domain], problem[:coordinate_system], (5, 5), 3, 3, -1e-6)
    end

    @testset "create_OCFE_discretization" begin
        problem, f, u_exact = create_test_problem(1, :cartesian)
        mesh = generate_initial_mesh(problem[:domain], problem[:coordinate_system], (10,), 3)
        A = create_OCFE_discretization(mesh, problem, 2)

        @test size(A, 1) == size(A, 2)
        @test issparse(A)
        @test !any(isnan, A.nzval)

        @test_throws ArgumentError create_OCFE_discretization(KSMesh{Float64,0}([], [], [], [], 0.0, 0), problem, 2)
    end

    @testset "generate_initial_mesh" begin
        problem, f, u_exact = create_test_problem(2, :cartesian)
        mesh = generate_initial_mesh(problem[:domain], problem[:coordinate_system], (5, 5), 3)

        @test mesh isa KSMesh{Float64,2}
        @test length(mesh.elements) == 25
        @test all(e -> e.polynomial_degree == 3, mesh.elements)

        @test_throws ArgumentError generate_initial_mesh(problem[:domain], problem[:coordinate_system], (0, 5), 3)
        @test_throws ArgumentError generate_initial_mesh(problem[:domain], problem[:coordinate_system], (5, 5), 0)
    end

    @testset "refine_mesh" begin
        problem, f, u_exact = create_test_problem(2, :cartesian)
        mesh = generate_initial_mesh(problem[:domain], problem[:coordinate_system], (5, 5), 3)
        error_estimates = rand(length(mesh.elements))
        refined_mesh = refine_mesh(mesh, error_estimates, 0.5)

        @test refined_mesh isa KSMesh{Float64,2}
        @test length(refined_mesh.elements) > length(mesh.elements)

        @test_throws ArgumentError refine_mesh(mesh, error_estimates[1:end-1], 0.5)
        @test_throws ArgumentError refine_mesh(mesh, error_estimates, -0.5)
    end

    @testset "estimate_mesh_error" begin
        problem, f, u_exact = create_test_problem(2, :cartesian)
        mesh = generate_initial_mesh(problem[:domain], problem[:coordinate_system], (5, 5), 3)
        error_estimates = estimate_mesh_error(mesh, problem)

        @test length(error_estimates) == length(mesh.elements)
        @test all(e -> e >= 0, error_estimates)

        empty_mesh = KSMesh{Float64,2}([], [], [], [], 0.0, 2)
        @test_throws ArgumentError estimate_mesh_error(empty_mesh, problem)
    end

end
