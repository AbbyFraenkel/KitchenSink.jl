using Test
using Plots
using KitchenSink.KSTypes
using KitchenSink.CoordinateSystems
using KitchenSink.Visualization

@testset "Visualization" begin
    # Create a mock mesh for testing
    function create_mock_mesh(dim::Int, num_elements::Int)
        cells = [KSCell(
                     id = i,
                     p = ntuple(_ -> 2, dim),
                     level = 1,
                     continuity_order = ntuple(_ -> 1, dim),
                     standard_cell_key = (ntuple(_ -> 2, dim), 1),
                     neighbors = Dict{Symbol, Int}(),
                     node_map = Dict(ntuple(d -> i, dim) => i
                     for i in 1:(num_elements + 1)),
                     tensor_product_mask = ntuple(_ -> trues(3), dim),
                     boundary_connectivity = Dict{Symbol, Int}(),
                     error_estimate = 0.0,
                     legendre_decay_rate = 0.0,
                     is_leaf = true,
                     is_fictitious = false,
                     refinement_options = nothing,
                     parent_id = nothing,
                     child_ids = nothing
                 ) for i in 1:num_elements]

        KSMesh(
            cells = cells,
            global_error_estimate = 0.0,
            boundary_cells = Dict{Symbol, Vector{Int}}(),
            physical_domain = x -> true
        )
    end

    mock_mesh = create_mock_mesh(1, 2)

    @testset "plot_solution" begin
        u = [0.0, 0.5, 1.0]
        p = Visualization.plot_solution(u, mock_mesh)

        @test p isa Plots.Plot
        @test length(p.series_list) == 1
        @test p.series_list[1][:x]≈[0.0, 0.5, 1.0] atol=1e-6
        @test p.series_list[1][:y] == u
    end

    @testset "plot_error_distribution" begin
        error = [1e-3, 1e-4, 1e-5]
        p = Visualization.plot_error_distribution(error, mock_mesh)

        @test p isa Plots.Plot
        @test length(p.series_list) == 1
        @test p.series_list[1][:x]≈[0.0, 0.5, 1.0] atol=1e-6
        @test p.series_list[1][:y] == error
        @test p.series_list[1][:yscale] == :log10
    end

    @testset "plot_mesh" begin
        p = Visualization.plot_mesh(mock_mesh)

        @test p isa Plots.Plot
        @test length(p.series_list) == 2  # One for scatter points, one for vertical lines
        @test p.series_list[1][:x]≈[0.0, 0.5, 1.0] atol=1e-6
        @test p.series_list[1][:y] == [0.0, 0.0, 0.0]
        @test p.series_list[1][:seriestype] == :scatter
        @test p.series_list[2][:seriestype] == :straightline  # Changed from :vline to :straightline
    end

    @testset "get_node_coordinates" begin
        coords = Visualization.get_node_coordinates(mock_mesh)

        @test coords isa Vector{Float64}
        @test length(coords) == 3
        @test coords≈[0.0, 0.5, 1.0] atol=1e-6
    end
end
