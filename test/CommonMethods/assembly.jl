# test/CommonMethods/assembly.jl
@testset "Assembly Tests" begin
    @testset "assemble_system_matrix" begin
        mesh = create_test_mesh_1D()

        @testset "Valid Inputs" begin
            A = assemble_system_matrix(mesh)
            expected_A = [2.0 1.0; 1.0 2.0]  # Example expected result for a simple mesh
            @test size(A, 1) == sum(length(el.basis_functions) for el in mesh.elements)
            @test size(A, 2) == sum(length(el.basis_functions) for el in mesh.elements)
            @test isapprox(A, expected_A, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError assemble_system_matrix("invalid_mesh")
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError assemble_system_matrix(empty_mesh)
        end
    end

    @testset "assemble_rhs_vector" begin
        mesh = create_test_mesh_1D()
        source_function = x -> sin(pi * x[1])

        @testset "Valid Inputs" begin
            b = assemble_rhs_vector(mesh, source_function)
            expected_b = [0.0, 0.0]  # Example expected result for a simple mesh and source function
            @test length(b) == sum(length(el.basis_functions) for el in mesh.elements)
            @test isapprox(b, expected_b, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError assemble_rhs_vector("invalid_mesh", source_function)
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test_throws BoundsError assemble_rhs_vector(empty_mesh, source_function)
        end
    end

    @testset "assemble_local_system_matrix" begin
        element = create_test_element_1D()

        @testset "Valid Inputs" begin
            A_local = assemble_local_system_matrix(element)
            expected_A_local = [1.0 0.5; 0.5 1.0]  # Example expected result for a simple element
            @test size(A_local) == (length(element.basis_functions), length(element.basis_functions))
            @test isapprox(A_local, expected_A_local, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError assemble_local_system_matrix("invalid_element")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError assemble_local_system_matrix(empty_element)
        end
    end

    @testset "element_size" begin
        element = create_test_element_1D()

        @testset "Valid Inputs" begin
            size = element_size(element)
            expected_size = 1.0  # Example expected result for a simple element
            @test isapprox(size, expected_size, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError element_size("invalid_element")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError element_size(empty_element)
        end
    end

    @testset "estimate_smoothness" begin
        element = create_test_element_1D()
        solution = [1.0, 2.0]

        @testset "Valid Inputs" begin
            smoothness = estimate_smoothness(element, solution)
            expected_smoothness = 0.0  # Example expected result for a simple element
            @test isapprox(smoothness, expected_smoothness, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError estimate_smoothness("invalid_element", solution)
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError estimate_smoothness(empty_element, solution)
        end
    end

    @testset "get_active_indices" begin
        mesh = create_test_mesh_1D()

        @testset "Valid Inputs" begin
            indices = get_active_indices(mesh)
            expected_indices = [1]  # Example expected result for a simple mesh
            @test length(indices) == length(mesh.elements)
            @test indices == expected_indices
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError get_active_indices("invalid_mesh")
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test length(get_active_indices(empty_mesh)) == 0
        end
    end

    @testset "inner_product" begin
        element = create_test_element_1D()
        f = element.basis_functions[1]
        g = element.basis_functions[2]

        @testset "Valid Inputs" begin
            product = inner_product(f, g, element)
            expected_product = 0.5  # Example expected result for the inner product
            @test isapprox(product, expected_product, atol=1e-6)
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError inner_product("invalid_function", g, element)
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.points = []
            @test_throws BoundsError inner_product(f, g, empty_element)
        end
    end

    @testset "total_dofs" begin
        mesh = create_test_mesh_1D()

        @testset "Valid Inputs" begin
            dofs = total_dofs(mesh)
            expected_dofs = 2  # Example expected result for a simple mesh
            @test dofs == expected_dofs
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError total_dofs("invalid_mesh")
        end

        @testset "Edge Cases" begin
            empty_mesh = create_test_mesh_1D()
            empty_mesh.elements = []
            @test total_dofs(empty_mesh) == 0
        end
    end

    @testset "element_dofs" begin
        element = create_test_element_1D()

        @testset "Valid Inputs" begin
            dofs = element_dofs(element)
            expected_dofs = 2  # Example expected result for a simple element
            @test dofs == expected_dofs
        end

        @testset "Invalid Inputs" begin
            @test_throws MethodError element_dofs("invalid_element")
        end

        @testset "Edge Cases" begin
            empty_element = create_test_element_1D()
            empty_element.basis_functions = []
            @test element_dofs(empty_element) == 0
        end
    end
end
