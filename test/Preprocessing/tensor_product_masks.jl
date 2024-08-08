@testset "Tensor Product Masks" begin
    @testset "update_tensor_product_masks!" begin
        problem = create_test_problem(2, :cartesian)
        mesh = generate_initial_mesh(problem.domain, problem.coordinate_system, (3, 3), 2)
        update_tensor_product_masks!(mesh)

        @test all(mask -> mask isa BitArray, mesh.tensor_product_masks)
        @test all(mask -> size(mask) == (3, 3), mesh.tensor_product_masks)

        empty_mesh = KSMesh{Float64,2}([], [], [], [], 0.0, 2)
        @test_throws ArgumentError update_tensor_product_masks!(empty_mesh)
    end

    @testset "restore_interface_compatibility!" begin
        mask1 = trues(3, 3)
        mask2 = trues(3, 3)
        mask2[1, :] .= false

        result = restore_interface_compatibility!(copy(mask1), mask2, 1, 1)
        @test all(result[1, :] .== false)

        @test_throws ArgumentError restore_interface_compatibility!(trues(3, 3), trues(4, 4), 1, 1)
        @test_throws ArgumentError restore_interface_compatibility!(trues(3, 3), trues(3, 3), 3, 1)
        @test_throws ArgumentError restore_interface_compatibility!(trues(3, 3), trues(3, 3), 1, 2)
    end

    @testset "deactivate_internal_boundary!" begin
        mask = trues(3, 3)
        result = deactivate_internal_boundary!(copy(mask), 1, 0)
        @test all(result[1, :] .== false)

        @test_throws ArgumentError deactivate_internal_boundary!(trues(3, 3), 3, 0)
        @test_throws ArgumentError deactivate_internal_boundary!(trues(3, 3), 1, 2)
    end

    @testset "update_location_matrices!" begin
        problem = create_test_problem(2)
        mesh = generate_initial_mesh(problem.domain, problem.coordinate_system, (3, 3), 2)
        update_tensor_product_masks!(mesh)
        update_location_matrices!(mesh)

        @test all(m -> m isa Dict{Int,Int}, mesh.location_matrices)
        @test all(m -> !isempty(m), mesh.location_matrices)

        empty_mesh = KSMesh{Float64,2}([], [], [], [], 0.0, 2)
        @test_throws ArgumentError update_location_matrices!(empty_mesh)
    end

    @testset "linear_index" begin
        @test linear_index(CartesianIndex(2, 2), (3, 3)) == 5
        @test linear_index(CartesianIndex(1, 1, 1), (2, 2, 2)) == 1
        @test linear_index(CartesianIndex(2, 2, 2), (2, 2, 2)) == 8

        @test_throws ArgumentError linear_index(CartesianIndex(3, 3), (2, 2))
    end

    @testset "update_tensor_product_masks_with_trunk!" begin
        problem = create_test_problem(2)
        mesh = generate_initial_mesh(problem.domain, problem.coordinate_system, (3, 3), 2)
        update_tensor_product_masks_with_trunk!(mesh)

        @test all(mask -> mask isa BitArray, mesh.tensor_product_masks)
        @test all(mask -> size(mask) == (3, 3), mesh.tensor_product_masks)
        @test all(mask -> mask[1, 1], mesh.tensor_product_masks)

        empty_mesh = KSMesh{Float64,2}([], [], [], [], 0.0, 2)
        @test_throws ArgumentError update_tensor_product_masks_with_trunk!(empty_mesh)
    end

end
