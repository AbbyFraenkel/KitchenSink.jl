using Test
using .MeshHierarchy


@testset "Mesh Hierarchy" begin
    # Test create_mesh_hierarchy
    @testset "create_mesh_hierarchy Tests" begin
        base_mesh = KSMesh([], [], [], [], 0.0, 2)  # Simplified base mesh for testing
        max_levels = 3
        hierarchy = create_mesh_hierarchy(base_mesh, max_levels)
        @test length(hierarchy) == max_levels
        @test_throws ArgumentError create_mesh_hierarchy(base_mesh, -1)
    end

    # Test refine_mesh_uniformly
    @testset "refine_mesh_uniformly Tests" begin
        mesh = KSMesh([], [], [], [], 0.0, 2)  # Simplified mesh for testing
        refined_mesh = refine_mesh_uniformly(mesh)
        # Assuming h_refine doubles the number of elements
        @test length(refined_mesh.elements) == 2 * length(mesh.elements)
    end

    # Test refine_mesh_hierarchy
    @testset "refine_mesh_hierarchy Tests" begin
        hierarchy = [KSMesh([], [], [], [], 0.0, 2) for _ in 1:3]  # Simplified hierarchy
        marked_elements = [1]  # Assuming at least one element to refine
        refined_hierarchy = refine_mesh_hierarchy(hierarchy, 2, marked_elements)
        @test length(refined_hierarchy) == length(hierarchy)
        @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 4, marked_elements)
    end

    # Test refine_marked_elements
    @testset "refine_marked_elements Tests" begin
        mesh = KSMesh([], [], [], [], 0.0, 2)  # Simplified mesh for testing
        marked_elements = [1]  # Assuming at least one element to refine
        refined_mesh = refine_marked_elements(mesh, marked_elements)
        # Assuming h_refine doubles the number of elements for marked ones
        @test length(refined_mesh.elements) == length(mesh.elements) + length(marked_elements)
        @test refine_marked_elements(mesh, []) == mesh
    end

    # Test adjust_finer_level
    @testset "adjust_finer_level Tests" begin
        coarse_mesh = KSMesh([], [], [], [], 0.0, 2)  # Simplified coarse mesh
        fine_mesh = KSMesh([], [], [], [], 0.0, 2)    # Simplified fine mesh
        adjusted_mesh = adjust_finer_level(coarse_mesh, fine_mesh)
        # Assuming h_refine doubles the number of elements for related elements
        @test length(adjusted_mesh.elements) == length(fine_mesh.elements)
    end



    @testset "create_mesh_hierarchy" begin
        @testset "Valid Cases" begin
            for dim in 1:3
                problem, _, _ = create_test_problem(dim, :cartesian)
                base_mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 4, dim), 2, problem.coordinate_system)
                hierarchy = create_mesh_hierarchy(base_mesh, 3)
                @test length(hierarchy) == 3
                @test all(mesh -> mesh isa KSMesh, hierarchy)
                @test all(i -> length(hierarchy[i].elements) >= length(hierarchy[i-1].elements), 2:3)
            end
        end

        @testset "Error Handling" begin
            problem, _, _ = create_test_problem(3, :cartesian)
            base_mesh = Preprocessing.create_mesh(problem.domain, (4, 4, 4), 2, problem.coordinate_system)
            @test_throws ArgumentError create_mesh_hierarchy(base_mesh, 0)
            @test_throws ArgumentError create_mesh_hierarchy(base_mesh, -1)
        end
    end
end

@testset "refine_mesh_uniformly" begin
    @testset "Valid Cases" begin
        problem, _ = create_test_problem(2, :cartesian)
        mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
        refined_mesh = refine_mesh_uniformly(mesh)
        @test length(refined_mesh.elements) == 4 * length(mesh.elements)
        @test all(e -> e.polynomial_degree == mesh.elements[1].polynomial_degree, refined_mesh.elements)
    end
end

@testset "refine_mesh_hierarchy" begin
    @testset "Valid Cases" begin
        problem, _ = create_test_problem(2, :cartesian)
        base_mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)
        marked_elements = [1]
        refined_hierarchy = refine_mesh_hierarchy(hierarchy, 2, marked_elements)
        @test length(refined_hierarchy) == length(hierarchy)
        @test length(refined_hierarchy[2].elements) > length(hierarchy[2].elements)
    end

    @testset "Error Handling" begin
        problem, _ = create_test_problem(2, :cartesian)
        base_mesh = Preprocessing.create_mesh(problem.domain, (4, 4), 2, problem.coordinate_system)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)
        @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 0, [1])
        @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 4, [1])
    end
end



@testset "Utility Functions" begin
    @testset "get_active_indices" begin
        problem, _, _ = create_test_problem(3, :cartesian)
        mesh = Preprocessing.create_mesh(problem.domain, (4, 4, 4), 2, problem.coordinate_system)
        Preprocessing.update_tensor_product_masks!(mesh)
        indices = get_active_indices(mesh.elements[1])
        @test indices isa Vector{Int}
        @test !isempty(indices)
        @test length(indices) == 27  # 3^3 for 3D quadratic elements
    end
end
@testset "v_cycle" begin
    @testset "Convergence Test" begin
        problem, f, _ = create_test_problem(3, :cartesian)
        base_mesh = Preprocessing.create_mesh(problem.domain, (4, 4, 4), 2, problem.coordinate_system)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)
        u = zeros(length(hierarchy[end].elements) * 27)

        initial_residual = compute_residual(hierarchy[end], f, u)
        for _ in 1:5
            u = v_cycle(hierarchy, f, u, length(hierarchy))
        end
        final_residual = compute_residual(hierarchy[end], f, u)

        @test norm(final_residual) < 0.1 * norm(initial_residual)
    end
end
@testset "solve_multi_level" begin
    @testset "Accuracy Tests" begin
        for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
            problem, f, u_exact = create_test_problem(dim, coord_system)
            base_mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 4, dim), 2, problem.coordinate_system)
            hierarchy = create_mesh_hierarchy(base_mesh, 3)
            solution = solve_multi_level(hierarchy, f, 3)

            # Compute L2 error
            error = 0.0
            for (i, element) in enumerate(hierarchy[end].elements)
                for (j, point) in enumerate(element.points)
                    error += (solution[i*3^dim+j] - u_exact(point.coordinates))^2 * element.jacobian_determinant
                end
            end
            error = sqrt(error)

            @test error < 1e-3  # Adjust this threshold based on your expected accuracy
        end
    end

    @testset "Performance Tests" begin
        problem, f, _ = create_test_problem(3, :cartesian)
        base_mesh = Preprocessing.create_mesh(problem.domain, (8, 8, 8), 2, problem.coordinate_system)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)

        b = @benchmark solve_multi_level($hierarchy, $f, 3)
        @test median(b.times) < 1e9  # Adjust this threshold based on your performance expectations
        @test b.memory < 1e9  # Adjust this threshold based on your memory usage expectations
    end

    @testset "3D Problems" begin
        for coord_system in [:cartesian, :spherical]
            problem, f, u_exact = create_test_problem(3, coord_system)
            base_mesh = Preprocessing.create_mesh(problem.domain, (4, 4, 4), 2, problem.coordinate_system)
            hierarchy = create_mesh_hierarchy(base_mesh, 3)
            solution = solve_multi_level(hierarchy, f, 3)
            @test length(solution) == length(hierarchy[end].elements) * 27  # 3^3 for 3D quadratic elements

            # Check solution at the center of the domain
            center_element = hierarchy[end].elements[div(end, 2)]
            center_point = center_element.points[div(end, 2)]
            center_solution = solution[center_element.id*27+div(end, 2)]
            @test isapprox(center_solution, u_exact(center_point.coordinates), rtol=1e-2)
        end
    end
end
