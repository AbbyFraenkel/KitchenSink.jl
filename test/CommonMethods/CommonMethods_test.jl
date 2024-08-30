using Test
using LinearAlgebra, SparseArrays, StaticArrays
# using ..KSTypes, ..CoordinateSystems, ..CommonMethods, ..SpectralMethods
using KitchenSink.KSTypes, KitchenSink.CoordinateSystems, KitchenSink.CommonMethods, KitchenSink.SpectralMethods

# Define common test data
domain_1d = [(0.0, 1.0)]
domain_2d = [(0.0, 1.0), (0.0, 1.0)]
domain_3d = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
num_elements_1d = (2,)
num_elements_2d = (2, 2)
num_elements_3d = (2, 2, 2)
polynomial_degree_1d = (3,)
polynomial_degree_2d = (3, 3)
polynomial_degree_3d = (3, 3, 3)

# Common Functions for Setting Up Test Scenarios
function create_test_problem_1d()
    KSProblem((x) -> x[1], (x) -> 0.0, domain_1d)
end

function create_test_problem_2d()
    KSProblem((x, y) -> x + y, (x, y) -> 0.0, domain_2d)
end

function create_test_problem_3d()
    KSProblem((x, y, z) -> x^2 + y^2 + z^2, (x, y, z) -> 0.0, domain_3d)
end

# Test Suite
@testset "CommonMethods Module Tests" begin
    @testset "create_mesh Function" begin
        @testset "Valid Cases" begin
            @testset "1D Mesh Creation" begin
                mesh_1d = create_mesh(domain_1d, num_elements_1d, polynomial_degree_1d)
                @test mesh_1d isa KSMesh{Float64, 1}
                @test length(mesh_1d.elements) == prod(num_elements_1d)
            end

            @testset "2D Mesh Creation" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                @test mesh_2d isa KSMesh{Float64, 2}
                @test length(mesh_2d.elements) == prod(num_elements_2d)
            end

            @testset "3D Mesh Creation" begin
                mesh_3d = create_mesh(domain_3d, num_elements_3d, polynomial_degree_3d)
                @test mesh_3d isa KSMesh{Float64, 3}
                @test length(mesh_3d.elements) == prod(num_elements_3d)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid Domain Dimensionality" begin
                @test_throws ArgumentError create_mesh([(0.0, 1.0)], num_elements_2d, polynomial_degree_2d)
            end

            @testset "Non-Positive Number of Elements" begin
                @test_throws ArgumentError create_mesh(domain_2d, (0, 2), polynomial_degree_2d)
            end

            @testset "Invalid Polynomial Degree" begin
                @test_throws ArgumentError create_mesh(domain_2d, num_elements_2d, (0,))
            end
        end

        @testset "Edge Cases" begin
            @testset "Single Element Mesh" begin
                mesh_single = create_mesh(domain_1d, (1,), polynomial_degree_1d)
                @test length(mesh_single.elements) == 1
            end

            @testset "High Polynomial Degree" begin
                mesh_high_degree = create_mesh(domain_1d, num_elements_1d, (10,))
                @test all(element -> element.polynomial_degree == (10,), mesh_high_degree.elements)
            end
        end
    end

    @testset "generate_mesh_nodes Function" begin
        @testset "Valid Cases" begin
            @testset "1D Nodes Generation" begin
                nodes_1d, node_map_1d = generate_mesh_nodes(domain_1d, num_elements_1d, polynomial_degree_1d)
                @test length(nodes_1d) == prod(num_elements_1d .* polynomial_degree_1d .+ 1)
                @test all(n -> n isa NTuple{1, Float64}, nodes_1d)
            end

            @testset "2D Nodes Generation" begin
                nodes_2d, node_map_2d = generate_mesh_nodes(domain_2d, num_elements_2d, polynomial_degree_2d)
                @test length(nodes_2d) == prod(num_elements_2d .* polynomial_degree_2d .+ 1)
                @test all(n -> n isa NTuple{2, Float64}, nodes_2d)
            end

            @testset "3D Nodes Generation" begin
                nodes_3d, node_map_3d = generate_mesh_nodes(domain_3d, num_elements_3d, polynomial_degree_3d)
                @test length(nodes_3d) == prod(num_elements_3d .* polynomial_degree_3d .+ 1)
                @test all(n -> n isa NTuple{3, Float64}, nodes_3d)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Domain Dimensionality Mismatch" begin
                @test_throws ArgumentError generate_mesh_nodes([(0.0, 1.0)], num_elements_2d, polynomial_degree_2d)
            end

            @testset "Non-Positive Polynomial Degree" begin
                @test_throws ArgumentError generate_mesh_nodes(domain_2d, num_elements_2d, (0,))
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal Domain Size" begin
                nodes_min, node_map_min = generate_mesh_nodes([(0.0, 0.1)], (1,), (1,))
                @test length(nodes_min) == 2
            end
        end
    end

    @testset "create_elements Function" begin
        @testset "Valid Cases" begin
            @testset "1D Elements Creation" begin
                nodes_1d, node_map_1d = generate_mesh_nodes(domain_1d, num_elements_1d, polynomial_degree_1d)
                elements_1d = create_elements(node_map_1d, num_elements_1d, polynomial_degree_1d)
                @test length(elements_1d) == prod(num_elements_1d)
                @test all(e -> e isa KSElement{Float64, 1}, elements_1d)
            end

            @testset "2D Elements Creation" begin
                nodes_2d, node_map_2d = generate_mesh_nodes(domain_2d, num_elements_2d, polynomial_degree_2d)
                elements_2d = create_elements(node_map_2d, num_elements_2d, polynomial_degree_2d)
                @test length(elements_2d) == prod(num_elements_2d)
                @test all(e -> e isa KSElement{Float64, 2}, elements_2d)
            end

            @testset "3D Elements Creation" begin
                nodes_3d, node_map_3d = generate_mesh_nodes(domain_3d, num_elements_3d, polynomial_degree_3d)
                elements_3d = create_elements(node_map_3d, num_elements_3d, polynomial_degree_3d)
                @test length(elements_3d) == prod(num_elements_3d)
                @test all(e -> e isa KSElement{Float64, 3}, elements_3d)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Empty Node Map" begin
                @test_throws ArgumentError create_elements(Dict(), num_elements_2d, polynomial_degree_2d)
            end
        end

        @testset "Edge Cases" begin
            @testset "Single Element Creation" begin
                nodes_single, node_map_single = generate_mesh_nodes(domain_1d, (1,), (1,))
                elements_single = create_elements(node_map_single, (1,), (1,))
                @test length(elements_single) == 1
            end
        end
    end

    @testset "update_location_matrices! Function" begin
        @testset "Valid Cases" begin
            @testset "Update Location Matrices for 2D Mesh" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                update_location_matrices!(mesh_2d)
                @test all(matrix -> matrix isa Dict{Int, Int}, mesh_2d.location_matrices)
                @test sum(length, mesh_2d.location_matrices) == total_dofs(mesh_2d)
            end

            @testset "Update Location Matrices for 3D Mesh" begin
                mesh_3d = create_mesh(domain_3d, num_elements_3d, polynomial_degree_3d)
                update_location_matrices!(mesh_3d)
                @test all(matrix -> matrix isa Dict{Int, Int}, mesh_3d.location_matrices)
                @test sum(length, mesh_3d.location_matrices) == total_dofs(mesh_3d)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Empty Mesh" begin
                mesh_empty = KSMesh{Float64, 2}([], [], [], 0.0)
                @test_throws ArgumentError update_location_matrices!(mesh_empty)
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal Location Matrix Update" begin
                mesh_min = create_mesh(domain_1d, (1,), (1,))
                update_location_matrices!(mesh_min)
                @test sum(length, mesh_min.location_matrices) == total_dofs(mesh_min)
            end
        end
    end

    @testset "assemble_system_matrix Function" begin
        @testset "Valid Cases" begin
            @testset "1D System Matrix Assembly" begin
                mesh_1d = create_mesh(domain_1d, num_elements_1d, polynomial_degree_1d)
                A_1d = assemble_system_matrix(mesh_1d)
                @test size(A_1d) == (total_dofs(mesh_1d), total_dofs(mesh_1d))
                @test A_1d isa SparseMatrixCSC{Float64, Int}
            end

            @testset "2D System Matrix Assembly" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                A_2d = assemble_system_matrix(mesh_2d)
                @test size(A_2d) == (total_dofs(mesh_2d), total_dofs(mesh_2d))
                @test A_2d isa SparseMatrixCSC{Float64, Int}
            end
        end

        @testset "Invalid Cases" begin
            @testset "Empty Mesh" begin
                mesh_empty = KSMesh{Float64, 1}([], [], [], 0.0)
                @test_throws ArgumentError assemble_system_matrix(mesh_empty)
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal System Matrix" begin
                mesh_min = create_mesh(domain_1d, (1,), (1,))
                A_min = assemble_system_matrix(mesh_min)
                @test size(A_min) == (2, 2)
            end
        end
    end

    @testset "assemble_local_system_matrix Function" begin
        @testset "Valid Cases" begin
            @testset "1D Local System Matrix Assembly" begin
                mesh_1d = create_mesh(domain_1d, num_elements_1d, polynomial_degree_1d)
                element_1d = mesh_1d.elements[1]
                A_local_1d = assemble_local_system_matrix(element_1d, mesh_1d)
                @test size(A_local_1d) == (polynomial_degree_1d[1] + 1, polynomial_degree_1d[1] + 1)
                @test A_local_1d isa Matrix{Float64}
            end
        end

        @testset "Invalid Cases" begin
            @testset "Empty Element" begin
                mesh_empty = KSMesh{Float64, 1}([], [], [], 0.0)
                @test_throws ArgumentError assemble_local_system_matrix(KSElement{Float64, 1}(0, 1, (1,)), mesh_empty)
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal Local Matrix" begin
                mesh_min = create_mesh(domain_1d, (1,), (1,))
                element_min = mesh_min.elements[1]
                A_local_min = assemble_local_system_matrix(element_min, mesh_min)
                @test size(A_local_min) == (2, 2)
            end
        end
    end

    @testset "assemble_rhs_vector Function" begin
        @testset "Valid Cases" begin
            @testset "1D RHS Vector Assembly" begin
                mesh_1d = create_mesh(domain_1d, num_elements_1d, polynomial_degree_1d)
                f_1d = x -> 1.0
                b_1d = assemble_rhs_vector(mesh_1d, f_1d)
                @test length(b_1d) == total_dofs(mesh_1d)
                @test b_1d isa Vector{Float64}
            end

            @testset "2D RHS Vector Assembly" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                f_2d = (x, y) -> x + y
                b_2d = assemble_rhs_vector(mesh_2d, f_2d)
                @test length(b_2d) == total_dofs(mesh_2d)
                @test b_2d isa Vector{Float64}
            end
        end

        @testset "Invalid Cases" begin
            @testset "Mismatched Vector Lengths" begin
                mesh_1d = create_mesh(domain_1d, num_elements_1d, polynomial_degree_1d)
                f_invalid = x -> 1 / (x[1] - 0.5)
                @test_throws DomainError assemble_rhs_vector(mesh_1d, f_invalid)
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal RHS Vector" begin
                mesh_min = create_mesh(domain_1d, (1,), (1,))
                f_min = x -> 0.0
                b_min = assemble_rhs_vector(mesh_min, f_min)
                @test all(b_min .== 0.0)
            end
        end
    end

    @testset "inner_product Function" begin
        @testset "Valid Cases" begin
            @testset "2D Inner Product Calculation" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                element_2d = mesh_2d.elements[1]
                std_elem = STANDARD_ELEMENT_CACHE[(element_2d.polynomial_degree, element_2d.level)]
                f_2d = ones(length(std_elem.collocation_points))
                g_2d = ones(length(std_elem.collocation_points))
                result_2d = inner_product(f_2d, g_2d, element_2d, mesh_2d)
                @test isapprox(result_2d, length(std_elem.collocation_points), atol = 1e-8)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Mismatched Input Vector Lengths" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                element_2d = mesh_2d.elements[1]
                std_elem = STANDARD_ELEMENT_CACHE[(element_2d.polynomial_degree, element_2d.level)]
                f_2d = ones(length(std_elem.collocation_points) - 1)
                g_2d = ones(length(std_elem.collocation_points))
                @test_throws DimensionMismatch inner_product(f_2d, g_2d, element_2d, mesh_2d)
            end
        end

        @testset "Edge Cases" begin
            @testset "Zero Length Vectors" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                element_2d = mesh_2d.elements[1]
                @test_throws DimensionMismatch inner_product([], [], element_2d, mesh_2d)
            end
        end
    end

    @testset "create_ocfe_discretization Function" begin
        @testset "Valid Cases" begin
            @testset "2D OCFE Discretization" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                problem_2d = create_test_problem_2d()
                A_2d, b_2d = create_ocfe_discretization(mesh_2d, problem_2d)
                @test size(A_2d) == (total_dofs(mesh_2d), total_dofs(mesh_2d))
                @test length(b_2d) == total_dofs(mesh_2d)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Empty Mesh Discretization" begin
                empty_mesh = KSMesh{Float64, 2}([], [], [], 0.0)
                problem_empty = create_test_problem_2d()
                @test_throws ArgumentError create_ocfe_discretization(empty_mesh, problem_empty)
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal Discretization" begin
                mesh_min = create_mesh(domain_1d, (1,), (1,))
                problem_min = create_test_problem_1d()
                A_min, b_min = create_ocfe_discretization(mesh_min, problem_min)
                @test size(A_min) == (2, 2)
                @test length(b_min) == 2
            end
        end
    end

    @testset "scale_standard_element Function" begin
        @testset "Valid Cases" begin
            @testset "Scale 2D Standard Element" begin
                std_elem_2d = create_standard_element((3, 3), 1)
                scaled_elem_2d = scale_standard_element(std_elem_2d, (0.5, 0.5))
                @test scaled_elem_2d isa StandardElement{Float64, 2}
                @test length(scaled_elem_2d.collocation_points) == (3 + 1)^2
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid Scaling Factor" begin
                std_elem_2d = create_standard_element((3, 3), 1)
                @test_throws DimensionMismatch scale_standard_element(std_elem_2d, (0.5,))
            end
        end

        @testset "Edge Cases" begin
            @testset "Scaling by Zero" begin
                std_elem_1d = create_standard_element((3,), 1)
                scaled_elem_1d = scale_standard_element(std_elem_1d, (0.0,))
                @test length(scaled_elem_1d.collocation_points) == 4
            end
        end
    end

    @testset "cache_scaled_element Function" begin
        @testset "Valid Cases" begin
            @testset "Cache Scaled 2D Element" begin
                std_elem_2d = create_standard_element((3, 3), 1)
                scaled_elem_2d = cache_scaled_element(std_elem_2d, (0.5, 0.5))
                @test scaled_elem_2d isa StandardElement{Float64, 2}
                @test haskey(STANDARD_ELEMENT_CACHE, ((0.5, 0.5), 1))
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid Cache Key" begin
                std_elem_1d = create_standard_element((3,), 1)
                @test_throws KeyError cache_scaled_element(std_elem_1d, (0.5, 0.5, 0.5))
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal Cache Scaling" begin
                std_elem_1d = create_standard_element((1,), 1)
                scaled_elem_1d = cache_scaled_element(std_elem_1d, (1.0,))
                @test scaled_elem_1d isa StandardElement{Float64, 1}
            end
        end
    end

    @testset "transform_standard_element Function" begin
        @testset "Valid Cases" begin
            @testset "Transform 2D Standard Element" begin
                std_elem_2d = create_standard_element((3, 7), 1)
                transformed_elem_2d = transform_standard_element(std_elem_2d, ((0.0, 1.0), (0.0, 2.0)), 1)
                @test transformed_elem_2d isa StandardElement{Float64, 2}
                @test all(0 .<= getindex.(transformed_elem_2d.collocation_points, 1) .<= 1)
                @test all(0 .<= getindex.(transformed_elem_2d.collocation_points, 2) .<= 2)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid Domain for Transformation" begin
                std_elem_1d = create_standard_element((3,), 1)
                @test_throws DimensionMismatch transform_standard_element(std_elem_1d, ((0.0, 1.0),), 1)
            end
        end

        @testset "Edge Cases" begin
            @testset "Transform with Minimal Domain" begin
                std_elem_1d = create_standard_element((3,), 1)
                transformed_elem_1d = transform_standard_element(std_elem_1d, ((0.0, 0.1),), 1)
                @test transformed_elem_1d isa StandardElement{Float64, 1}
            end
        end
    end

    @testset "element_dofs Function" begin
        @testset "Valid Cases" begin
            @testset "Calculate DOFs for 2D Element" begin
                mesh_2d = create_mesh(domain_2d, num_elements_2d, polynomial_degree_2d)
                element_2d = mesh_2d.elements[1]
                dofs_2d = element_dofs(element_2d, mesh_2d)
                @test dofs_2d > 0
                @test isa(dofs_2d, Int)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid Element in Mesh" begin
                mesh_empty = KSMesh{Float64, 2}([], [], [], 0.0)
                element_invalid = KSElement{Float64, 2}(1, 1, (1, 1))
                @test_throws ArgumentError element_dofs(element_invalid, mesh_empty)
            end
        end

        @testset "Edge Cases" begin
            @testset "Minimal DOFs Calculation" begin
                mesh_min = create_mesh(domain_1d, (1,), (1,))
                element_min = mesh_min.elements[1]
                dofs_min = element_dofs(element_min, mesh_min)
                @test dofs_min == 2
            end
        end
    end
end
