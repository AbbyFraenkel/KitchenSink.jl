@testset "Refinement by Superposition" begin
        @testset "hp_refine_superposition" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, _, _ = create_test_problem(dim, coord_system)
                mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(mesh)

                element = mesh.elements[1]

                # Test h-refinement
                refined_elements = hp_refine_superposition(element, 1.0, 0.1, 0.5)
                @test length(refined_elements) > 1

                # Test p-refinement
                refined_elements = hp_refine_superposition(element, 1.0, 0.9, 0.5)
                @test length(refined_elements) == 1
                @test refined_elements[1].polynomial_degree == element.polynomial_degree + 1
            end
        end

        @testset "superconvergence_refinement" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, f, u_exact = create_test_problem(dim, coord_system)
                mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(mesh)

                element = mesh.elements[1]
                solution = [u_exact(p.coordinates) for p in element.points]

                refined_elements = superconvergence_refinement(element, solution, problem)
                @test length(refined_elements) > 0
            end
        end

        @testset "adapt_mesh_superposition" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, f, u_exact = create_test_problem(dim, coord_system)
                mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(mesh)

                solution = [u_exact(p.coordinates) for element in mesh.elements for p in element.points]

                adapted_mesh = adapt_mesh_superposition(mesh, solution, problem, 0.1)
                @test length(adapted_mesh.elements) >= length(mesh.elements)
            end
        end
    end
