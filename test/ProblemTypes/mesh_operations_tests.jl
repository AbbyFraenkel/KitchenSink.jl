using Test
using LinearAlgebra
using KitchenSink.KSTypes
using KitchenSink.SpectralMethods
using KitchenSink.CoordinateSystems
using KitchenSink.ProblemTypes
using KitchenSink.Transforms
using KitchenSink.BoundaryConditions
using KitchenSink.NumericUtilities

@testset "Mesh Operations" begin
    @testset "Basic Mesh Connectivity" begin
        mesh = create_test_mesh(2)
        @test !isempty(mesh.cells)

        ProblemTypes.update_mesh_connectivity!(mesh)

        for cell in mesh.cells
            @test haskey(cell.neighbors, :dim1_pos)
            @test haskey(cell.neighbors, :dim1_neg)
            @test haskey(cell.neighbors, :dim2_pos)
            @test haskey(cell.neighbors, :dim2_neg)
        end
    end

    @testset "Coordinate Mapping" begin
        mesh = create_test_mesh(2)
        nodes = ProblemTypes.map_nodes_to_computational_domain(mesh)

        @test !isempty(nodes)
        for coords in values(nodes)
            @test length(coords) == 2
            @test all(-1.0 .<= coords .<= 1.0)
        end
    end

    @testset "Cell Neighbors" begin
        mesh = create_test_mesh(2)
        ProblemTypes.update_cell_neighbors!(mesh)

        # Test neighbor relationships
        for cell in mesh.cells
            for (dir, neighbor_id) in cell.neighbors
                if neighbor_id != -1
                    neighbor = mesh.cells[neighbor_id]
                    opp_dir = ProblemTypes.get_opposite_direction(dir)
                    @test neighbor.neighbors[opp_dir] == cell.id
                end
            end
        end
    end

    @testset "Boundary Detection" begin
        mesh = create_test_mesh(2)
        ProblemTypes.update_boundary_cells!(mesh)

        @test !isempty(mesh.boundary_cells)

        # Test boundary node identification
        for (dir, cell_ids) in mesh.boundary_cells
            for cell_id in cell_ids
                cell = mesh.cells[cell_id]
                @test cell.neighbors[dir] == -1

                # Test boundary coordinates
                std_cell = SpectralMethods.get_or_create_standard_cell(
                    cell.p, cell.level)
                dim = parse(Int, string(dir)[4])
                side = endswith(string(dir), "pos")

                nodes = std_cell.nodes_with_boundary[dim]
                coord = side ? nodes[end] : nodes[1]
                @test ProblemTypes.is_on_boundary(coord)
            end
        end
    end

    @testset "Transform Integration" begin
        # Create mesh with coordinate transformation
        cart_sys = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
        polar_sys = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))

        transform = NonlinearTransform(
            forward_map = x -> collect(from_cartesian(x, polar_sys)),
            inverse_map = x -> collect(to_cartesian(x, polar_sys))
        )

        mesh = create_test_mesh(2)
        mesh.transformation_data = transform

        comp_nodes = ProblemTypes.map_nodes_to_computational_domain(mesh)
        node_cell_map = ProblemTypes.create_node_cell_map(mesh)

        # Test transform application
        ProblemTypes.update_node_map_with_transform!(mesh, comp_nodes, node_cell_map)

        for cell in mesh.cells
            for (_, global_idx) in cell.node_map
                @test haskey(comp_nodes, global_idx)
                coords = comp_nodes[global_idx]
                trans_coords = transform.forward_map(coords)
                @test 0.0 <= trans_coords[1] <= 1.0  # r
                @test 0.0 <= trans_coords[2] <= 2π   # θ
            end
        end
    end

    @testset "Node Cell Mapping" begin
        mesh = create_test_mesh(2)
        node_cell_map = ProblemTypes.create_node_cell_map(mesh)

        @test !isempty(node_cell_map)
        for (node, cells) in node_cell_map
            @test !isempty(cells)
            for cell_id in cells
                cell = mesh.cells[cell_id]
                @test any(global_idx -> global_idx == node, values(cell.node_map))
            end
        end
    end

    @testset "Cell Centers and Directions" begin
        mesh = create_test_mesh(2)
        cell1, cell2 = mesh.cells[1:2]
        std_cell = SpectralMethods.get_or_create_standard_cell(cell1.p, cell1.level)

        center = ProblemTypes.get_cell_center(cell1, std_cell)
        @test length(center) == 2
        @test all(isfinite, center)

        dir = ProblemTypes.get_neighbor_direction(cell1, cell2)
        @test dir isa Symbol
        @test occursin(r"dim\d+_(pos|neg)", string(dir))
    end

    @testset "Edge Cases" begin
        # Single cell mesh
        single_mesh = create_test_mesh(1, num_cells=(1,))
        ProblemTypes.update_mesh_connectivity!(single_mesh)
        @test length(single_mesh.cells) == 1
        @test all(v -> v == -1, values(first(single_mesh.cells).neighbors))

        # Invalid direction test
        @test_throws ArgumentError ProblemTypes.get_opposite_direction(:invalid)

        # Empty mesh test
        empty_mesh = create_test_mesh(1)
        empty_mesh.cells = KSCell{Float64,1}[]
        @test_throws ArgumentError ProblemTypes.update_mesh_connectivity!(empty_mesh)
    end

    @testset "Integration with BoundaryConditions" begin
        mesh = create_test_mesh(2)
        ProblemTypes.update_boundary_cells!(mesh)

        boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)
        @test !isempty(boundary_nodes)

        # Verify boundary nodes are actually on boundaries
        for node in boundary_nodes
            found_on_boundary = false
            for cell in mesh.cells
                if haskey(cell.node_map, node)
                    std_cell = SpectralMethods.get_or_create_standard_cell(cell.p, cell.level)
                    coords = ntuple(d -> std_cell.nodes_with_boundary[d][node[d]], 2)
                    if any(ProblemTypes.is_on_boundary.(coords))
                        found_on_boundary = true
                        break
                    end
                end
            end
            @test found_on_boundary
        end
    end

    @testset "Performance" begin
        # Test scaling with mesh size
        function measure_update_time(dim::Int, num_cells::Int)
            mesh = create_test_mesh(dim, num_cells=ntuple(_ -> num_cells, dim))
            return @elapsed ProblemTypes.update_mesh_connectivity!(mesh)
        end

        t1 = measure_update_time(2, 2)
        t2 = measure_update_time(2, 4)
        # Should scale approximately linearly with number of cells
        @test t2/t1 < 5.0

        # Test memory allocation
        mesh = create_test_mesh(2)
        allocs = @allocated ProblemTypes.update_mesh_connectivity!(mesh)
        # Set reasonable allocation limits
        max_allowed_allocs = 10^6
        @test allocs < max_allowed_allocs
    end
end

