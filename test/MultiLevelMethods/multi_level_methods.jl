@testset "Multi-Level Methods" begin
    @testset "Mesh Hierarchy" begin
        @testset "create_mesh_hierarchy" begin
            @test_throws MethodError create_mesh_hierarchy(KSMesh([], [], [], [], 0.0, 2), -1)

            for dim in 1:3
                problem, _, _ = create_test_problem(dim, :cartesian)
                base_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, ntuple(_ -> 4, dim), 2)
                hierarchy = create_mesh_hierarchy(base_mesh, 3)

                @test length(hierarchy) == 3
                @test all(mesh -> mesh isa KSMesh, hierarchy)
                @test all(i -> length(hierarchy[i].elements) >= length(hierarchy[i-1].elements), 2:3)
            end
        end

        @testset "refine_mesh_uniformly" begin
            problem, _, _ = create_test_problem(2, :cartesian)
            mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (4, 4), 2)
            refined_mesh = refine_mesh_uniformly(mesh)

            @test length(refined_mesh.elements) == 4 * length(mesh.elements)
            @test all(e -> e.polynomial_degree == mesh.elements[1].polynomial_degree, refined_mesh.elements)
        end

        @testset "refine_mesh_hierarchy" begin
            problem, _, _ = create_test_problem(2, :cartesian)
            base_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (4, 4), 2)
            hierarchy = create_mesh_hierarchy(base_mesh, 3)
            marked_elements = [1]

            refined_hierarchy = refine_mesh_hierarchy(hierarchy, 2, marked_elements)
            @test length(refined_hierarchy) == length(hierarchy)
            @test length(refined_hierarchy[2].elements) > length(hierarchy[2].elements)

            @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 0, [1])
            @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 4, [1])
        end
    end

    @testset "Utility Functions" begin
        problem, _, _ = create_test_problem(3, :cartesian)
        coarse_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (2, 2, 2), 1)
        fine_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (4, 4, 4), 1)

        P = compute_local_prolongation(coarse_mesh.elements[1], fine_mesh.elements[1])
        R = compute_local_restriction(coarse_mesh.elements[1], fine_mesh.elements[1])
        @test size(P) == (8, 2)
        @test size(R) == (2, 8)
        @test isapprox(R, P', rtol=1e-10)

        mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (4, 4, 4), 2)
        Preprocessing.update_tensor_product_masks!(mesh)
        indices = get_active_indices(mesh.elements[1])
        @test indices isa Vector{Int}
        @test length(indices) == 27
    end

    @testset "v_cycle" begin
        problem, f, _ = create_test_problem(3, :cartesian)
        base_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (4, 4, 4), 2)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)
        u = zeros(length(hierarchy[end].elements) * 27)

        initial_residual = compute_residual(hierarchy[end], f, u)
        for _ in 1:5
            u = v_cycle(hierarchy, f, u, length(hierarchy))
        end
        final_residual = compute_residual(hierarchy[end], f, u)

        @test norm(final_residual) < 0.1 * norm(initial_residual)
    end

    @testset "solve_multi_level" begin
        @testset "Accuracy" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, f, u_exact = create_test_problem(dim, coord_system)
                base_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, ntuple(_ -> 4, dim), 2)
                hierarchy = create_mesh_hierarchy(base_mesh, 3)
                solution = solve_multi_level(hierarchy, f, 3)

                error = sqrt(sum((solution[i*3^dim+j] - u_exact(element.points[j].coordinates))^2 * element.jacobian_determinant
                                 for (i, element) in enumerate(hierarchy[end].elements)
                                 for j in 1:length(element.points)))

                @test error < 1e-3
            end
        end

        @testset "3D Problems" begin
            for coord_system in [:cartesian, :spherical]
                problem, f, u_exact = create_test_problem(3, coord_system)
                base_mesh = Preprocessing.create_mesh(problem.domain, problem.coordinate_system, (4, 4, 4), 2)
                hierarchy = create_mesh_hierarchy(base_mesh, 3)
                solution = solve_multi_level(hierarchy, f, 3)

                @test length(solution) == length(hierarchy[end].elements) * 27

                center_element = hierarchy[end].elements[div(end, 2)]
                center_point = center_element.points[div(end, 2)]
                center_solution = solution[center_element.id*27+div(end, 2)]
                @test isapprox(center_solution, u_exact(center_point.coordinates), rtol=1e-2)
            end
        end
    end
end
