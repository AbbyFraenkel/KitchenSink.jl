using Test
using LinearAlgebra, SparseArrays, StaticArrays
using KitchenSink.KSTypes,
      KitchenSink.CoordinateSystems, KitchenSink.CommonMethods, KitchenSink.SpectralMethods

# Define common test data
domain_1d = ((0.0, 1.0),)
domain_2d = ((0.0, 1.0), (0.0, 1.0))
domain_3d = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
num_elements_1d = [2]
num_elements_2d = [2, 2]
num_elements_3d = [2, 2, 2]
polynomial_degree_1d = [3]
polynomial_degree_2d = [3, 3]
polynomial_degree_3d = [3, 3, 3]

# Common Functions for Setting Up Test Scenarios
function create_test_problem_1d()
    KSProblem((x) -> x[1], (x) -> 0.0, domain_1d, KSCartesianCoordinates(domain_1d))
end

function create_test_problem_2d()
    KSProblem((x, y) -> x + y, (x, y) -> 0.0, domain_2d, KSCartesianCoordinates(domain_2d))
end

function create_test_problem_3d()
    KSProblem(
        (x, y, z) -> x^2 + y^2 + z^2,
        (x, y, z) -> 0.0,
        domain_3d,
        KSCartesianCoordinates(domain_3d)
    )
end

# Test Suite
@testset "CommonMethods Module Tests" begin
    @testset "create_mesh Function" begin
        @testset "Valid Cases" begin
            @testset "1D Mesh Creation" begin
                mesh_1d = create_mesh(
                    domain_1d,
                    num_elements_1d,
                    polynomial_degree_1d,
                    KSCartesianCoordinates(domain_1d)
                )
                @test mesh_1d isa KSMesh{Float64, 1}
                @test length(mesh_1d.elements) == prod(num_elements_1d)
            end

            @testset "2D Mesh Creation" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                @test mesh_2d isa KSMesh{Float64, 2}
                @test length(mesh_2d.elements) == prod(num_elements_2d)
            end

            @testset "3D Mesh Creation" begin
                mesh_3d = create_mesh(
                    domain_3d,
                    num_elements_3d,
                    polynomial_degree_3d,
                    KSCartesianCoordinates(domain_3d)
                )
                @test mesh_3d isa KSMesh{Float64, 3}
                @test length(mesh_3d.elements) == prod(num_elements_3d)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid Domain Dimensionality" begin
                @test_throws AssertionError create_mesh(
                    domain_1d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
            end

            @testset "Non-Positive Number of Elements" begin
                @test_throws AssertionError create_mesh(
                    domain_2d,
                    [0, 2],
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
            end

            @testset "Invalid Polynomial Degree" begin
                @test_throws AssertionError create_mesh(
                    domain_2d,
                    num_elements_2d,
                    [0, 3],
                    KSCartesianCoordinates(domain_2d)
                )
            end
        end

        @testset "Edge Cases" begin
            @testset "Single Element Mesh" begin
                mesh_single = create_mesh(
                    domain_1d,
                    [1],
                    polynomial_degree_1d,
                    KSCartesianCoordinates(domain_1d)
                )
                @test length(mesh_single.elements) == 1
            end

            @testset "High Polynomial Degree" begin
                mesh_high_degree = create_mesh(
                    domain_1d,
                    num_elements_1d,
                    [10],
                    KSCartesianCoordinates(domain_1d)
                )
                @test all(
                    element -> element.polynomial_degree == (10,),
                    mesh_high_degree.elements
                )
            end
        end
    end

    @testset "update_tensor_product_masks! Function" begin
        @testset "Valid Cases" begin
            @testset "Update Tensor Product Masks for 2D Mesh" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                update_tensor_product_masks!(mesh_2d)
                @test all(mask -> mask isa BitArray{2}, mesh_2d.tensor_product_masks)
                @test length(mesh_2d.tensor_product_masks) == length(mesh_2d.elements)
            end
        end
    end

    @testset "update_location_matrices! Function" begin
        @testset "Valid Cases" begin
            @testset "Update Location Matrices for 2D Mesh" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                update_location_matrices!(mesh_2d)
                @test all(matrix -> matrix isa Dict{Int, Int}, mesh_2d.location_matrices)
                @test sum(length, mesh_2d.location_matrices) == total_dofs(mesh_2d)
            end
        end
    end

    @testset "assemble_system_matrix Function" begin
        @testset "Valid Cases" begin
            @testset "2D System Matrix Assembly" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                A_2d = assemble_system_matrix(mesh_2d, KSCartesianCoordinates(domain_2d))
                @test size(A_2d) == (total_dofs(mesh_2d), total_dofs(mesh_2d))
                @test A_2d isa SparseMatrixCSC{Float64, Int}
            end
        end
    end

    @testset "assemble_local_system_matrix Function" begin
        @testset "Valid Cases" begin
            @testset "2D Local System Matrix Assembly" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                element_2d = mesh_2d.elements[1]
                A_local_2d = assemble_local_system_matrix(
                    element_2d,
                    mesh_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                @test size(A_local_2d) ==
                      (prod(polynomial_degree_2d .+ 1), prod(polynomial_degree_2d .+ 1))
                @test A_local_2d isa SparseMatrixCSC{Float64, Int}
            end
        end
    end

    @testset "assemble_rhs_vector Function" begin
        @testset "Valid Cases" begin
            @testset "2D RHS Vector Assembly" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                f_2d = (x, y) -> x + y
                b_2d = assemble_rhs_vector(mesh_2d, f_2d, KSCartesianCoordinates(domain_2d))
                @test length(b_2d) == total_dofs(mesh_2d)
                @test b_2d isa Vector{Float64}
            end
        end
    end

    @testset "inner_product Function" begin
        @testset "Valid Cases" begin
            @testset "2D Inner Product Calculation" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                element_2d = mesh_2d.elements[1]
                std_elem = create_or_get_standard_element(
                    element_2d.polynomial_degree,
                    element_2d.level
                )
                f_2d = ones(length(std_elem.collocation_points))
                g_2d = ones(length(std_elem.collocation_points))
                result_2d = inner_product(f_2d, g_2d, element_2d, mesh_2d)
                @test isapprox(result_2d, length(std_elem.collocation_points), atol = 1e-8)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Mismatched Input Vector Lengths" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                element_2d = mesh_2d.elements[1]
                std_elem = create_or_get_standard_element(
                    element_2d.polynomial_degree,
                    element_2d.level
                )
                f_2d = ones(length(std_elem.collocation_points) - 1)
                g_2d = ones(length(std_elem.collocation_points))
                @test_throws DimensionMismatch inner_product(
                    f_2d,
                    g_2d,
                    element_2d,
                    mesh_2d
                )
            end
        end
    end

    @testset "create_ocfc_discretization Function" begin
        @testset "Valid Cases" begin
            @testset "2D OCFC Discretization" begin
                mesh_2d = create_mesh(
                    domain_2d,
                    num_elements_2d,
                    polynomial_degree_2d,
                    KSCartesianCoordinates(domain_2d)
                )
                problem_2d = create_test_problem_2d()
                A_2d, b_2d = create_ocfc_discretization(mesh_2d, problem_2d)
                @test size(A_2d) == (total_dofs(mesh_2d), total_dofs(mesh_2d))
                @test length(b_2d) == total_dofs(mesh_2d)
            end
        end
    end

    @testset "create_or_get_standard_element Function" begin
        @testset "Valid Cases" begin
            @testset "Create and Cache 2D Standard Element" begin
                polynomial_degree = (3, 3)
                level = 1
                std_elem = create_or_get_standard_element(polynomial_degree, level)
                @test std_elem isa StandardKSCell{Float64, 2}
                @test haskey(STANDARD_ELEMENT_CACHE, (polynomial_degree, level))
            end
        end
    end

    @testset "deactivate_internal_boundary! Function" begin
        @testset "Valid Cases" begin
            @testset "Deactivate Internal Boundary in 2D Mask" begin
                mask = trues(4, 4)
                deactivate_internal_boundary!(mask, 1, 1)
                @test all(mask[1:2, :] .== false)
                @test all(mask[3:4, :] .== true)
            end
        end
    end
end
