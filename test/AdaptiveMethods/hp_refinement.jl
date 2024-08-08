
@testset "hp-Refinement" begin
        @testset "h_refine" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, _, _ = create_test_problem(dim, coord_system)
                mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(mesh)

                element = mesh.elements[1]
                refined_elements = h_refine(element)
                @test length(refined_elements) == 2^dim
                @test all(e -> e.level == element.level + 1, refined_elements)
                @test all(e -> e.polynomial_degree == element.polynomial_degree, refined_elements)
            end
        end

        @testset "p_refine" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, _, _ = create_test_problem(dim, coord_system)
                mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(mesh)

                element = mesh.elements[1]
                refined_element = p_refine(element)
                @test refined_element.polynomial_degree == element.polynomial_degree + 1
                @test length(refined_element.points) == (refined_element.polynomial_degree + 1)^dim
            end
        end

        @testset "hp_refine" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, _, _ = create_test_problem(dim, coord_system)
                mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(mesh)

                element = mesh.elements[1]

                # Test h-refinement
                refined_elements = hp_refine(element, 1.0, 0.1, 0.5)
                @test length(refined_elements) == 2^dim

                # Test p-refinement
                refined_elements = hp_refine(element, 1.0, 0.9, 0.5)
                @test length(refined_elements) == 1
                @test refined_elements[1].polynomial_degree == element.polynomial_degree + 1
            end
        end
    end
