@testset "Prolongation and Restriction" begin
    @testset "prolongate and restrict" begin
        @testset "Valid Cases" begin
            for dim in 1:3
                problem, _, _ = create_test_problem(dim, :cartesian)
                coarse_mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 4, dim), 2, problem.coordinate_system)
                fine_mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 8, dim), 2, problem.coordinate_system)
                coarse_solution = rand(length(coarse_mesh.elements) * 3^dim)
                fine_solution = prolongate(coarse_solution, coarse_mesh, fine_mesh)
                @test length(fine_solution) == length(fine_mesh.elements) * 3^dim
                restricted_solution = restrict(fine_solution, fine_mesh, coarse_mesh)
                @test length(restricted_solution) == length(coarse_solution)
                @test norm(restricted_solution - coarse_solution) / norm(coarse_solution) < 1e-2
            end
        end

        @testset "Error Handling" begin
            problem, _, _ = create_test_problem(3, :cartesian)
            coarse_mesh = Preprocessing.create_mesh(problem.domain, (4, 4, 4), 2, problem.coordinate_system)
            fine_mesh = Preprocessing.create_mesh(problem.domain, (8, 8, 8), 2, problem.coordinate_system)
            @test_throws DimensionMismatch prolongate(rand(5), coarse_mesh, fine_mesh)
            @test_throws DimensionMismatch restrict(rand(5), fine_mesh, coarse_mesh)
        end
    end
end



@testset "Prolongation and Restriction" begin
    @testset "prolongate" begin
        @testset "Valid Cases" begin
            problem, _ = create_test_problem(2, :cartesian)
            coarse_mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
            fine_mesh = Preprocessing.create_mesh(problem.domain, (8, 8), 2, problem.coordinate_system)
            coarse_solution = rand(length(coarse_mesh.elements) * 9)
            fine_solution = prolongate(coarse_solution, coarse_mesh, fine_mesh)
            @test length(fine_solution) == length(fine_mesh.elements) * 9
            # Add more specific tests for the accuracy of prolongation
        end

        @testset "Error Handling" begin
            problem, _ = create_test_problem(2, :cartesian)
            coarse_mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
            fine_mesh = Preprocessing.create_mesh(problem.domain, (8, 8), 2, problem.coordinate_system)
            @test_throws DimensionMismatch prolongate(rand(5), coarse_mesh, fine_mesh)
        end
    end

    @testset "restrict" begin
        @testset "Valid Cases" begin
            problem, _ = create_test_problem(2, :cartesian)
            coarse_mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
            fine_mesh = Preprocessing.create_mesh(problem.domain, (8, 8), 2, problem.coordinate_system)
            fine_solution = rand(length(fine_mesh.elements) * 9)
            coarse_solution = restrict(fine_solution, fine_mesh, coarse_mesh)
            @test length(coarse_solution) == length(coarse_mesh.elements) * 9
            # Add more specific tests for the accuracy of restriction
        end

        @testset "Error Handling" begin
            problem, _ = create_test_problem(2, :cartesian)
            coarse_mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
            fine_mesh = Preprocessing.create_mesh(problem.domain, (8, 8), 2, problem.coordinate_system)
            @test_throws DimensionMismatch restrict(rand(5), fine_mesh, coarse_mesh)
        end
    end

    @testset "compute_local_prolongation and restriction" begin
        problem, _, _ = create_test_problem(3, :cartesian)
        coarse_mesh = Preprocessing.create_mesh(problem.domain, (2, 2, 2), 1, problem.coordinate_system)
        fine_mesh = Preprocessing.create_mesh(problem.domain, (4, 4, 4), 1, problem.coordinate_system)
        P = compute_local_prolongation(coarse_mesh.elements[1], fine_mesh.elements[1])
        R = compute_local_restriction(coarse_mesh.elements[1], fine_mesh.elements[1])
        @test size(P) == (8, 2)  # 8 nodes in fine element, 2 nodes in coarse element for linear 3D elements
        @test size(R) == (2, 8)
        @test isapprox(R, P', rtol=1e-10)
    end

end
