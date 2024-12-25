using Test
using LinearAlgebra, StaticArrays, SparseArrays, Statistics, BenchmarkTools
using KitchenSink.KSTypes, KitchenSink.NumericUtilities, KitchenSink.ProblemTypes
using KitchenSink.CoordinateSystems, KitchenSink.BoundaryConditions
using KitchenSink.SpectralMethods, KitchenSink.CacheManagement
using KitchenSink.Transforms, KitchenSink.Preconditioners, KitchenSink.LinearSolvers

include("test_utils_ProblemTypes.jl")

# @testset "ProblemTypes" begin

#     @testset "Problem Creation and Validation" begin
#         @testset "Problem Types" begin
#             # Test each problem type
#             for dim in TEST_DIMENSIONS
#                 for prob_type in TEST_PROBLEM_TYPES
#                     @testset "$(prob_type) - $(dim)D" begin
#                         problem = create_test_problem(prob_type, dim)
#                         @test problem isa AbstractKSProblem
#                         @test get_problem_dimension(problem) == dim

#                         # Test coordinate system
#                         coord_sys = get_coordinate_system(problem)
#                         @test coord_sys isa AbstractKSCoordinateSystem

#                         # Test time span if applicable
#                         if hasfield(typeof(problem), :tspan)
#                             @test validate_time_span(problem)
#                         end
#                     end
#                 end
#             end
#         end

#         @testset "Problem-Mesh Compatibility" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)
#                     for prob_type in TEST_PROBLEM_TYPES
#                         problem = create_test_problem(prob_type, dim)
#                         @test validate_problem_mesh_compatibility(problem, mesh)
#                         @test validate_problem_setup(problem, mesh, coord_sys)
#                     end
#                 end
#             end
#         end
#     end

#     @testset "Mesh Operations" begin
#         @testset "Connectivity" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)

#                     # Test mesh update
#                     update_mesh_connectivity!(mesh, coord_sys)

#                     for cell in mesh.cells
#                         # Test neighbor relationships
#                         for (dir, neighbor_id) in cell.neighbors
#                             if neighbor_id != -1  # Not a boundary
#                                 neighbor = mesh.cells[neighbor_id]
#                                 opp_dir = get_opposite_direction(dir)
#                                 @test haskey(neighbor.neighbors, opp_dir)
#                                 @test neighbor.neighbors[opp_dir] == cell.id
#                             end
#                         end

#                         # Test node map
#                         @test !isempty(cell.node_map)
#                         @test all(v -> v > 0, values(cell.node_map))
#                     end

#                     # Test boundary cells
#                     @test !isempty(mesh.boundary_cells)
#                     for (dir, cells) in mesh.boundary_cells
#                         @test all(id -> id in 1:length(mesh.cells), cells)
#                     end
#                 end
#             end
#         end

#         @testset "Node Mapping" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)

#                     # Test node-cell map creation
#                     node_cell_map = create_node_cell_map(mesh)
#                     @test !isempty(node_cell_map)

#                     # Verify map consistency
#                     for (node, cells) in node_cell_map
#                         for cell_id in cells
#                             cell = mesh.cells[cell_id]
#                             @test any(global_idx -> global_idx == node, values(cell.node_map))
#                         end
#                     end
#                 end
#             end
#         end
#     end

#     @testset "Assembly" begin
#         @testset "Matrix Assembly" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)

#                     for prob_type in TEST_PROBLEM_TYPES
#                         problem = create_test_problem(prob_type, dim)

#                         # Test system creation
#                         A, b = create_system_matrix_and_vector(problem, mesh, coord_sys)

#                         # Basic matrix properties
#                         @test size(A, 1) == size(A, 2)  # Square matrix
#                         @test length(b) == size(A, 1)   # Compatible sizes
#                         @test issparse(A)               # Sparse storage

#                         # Verify structure
#                         @test verify_matrix_structure(problem, A)
#                     end
#                 end
#             end
#         end

#         @testset "Local Assembly" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)

#                     for prob_type in TEST_PROBLEM_TYPES
#                         problem = create_test_problem(prob_type, dim)

#                         for cell in mesh.cells
#                             # Test local matrix initialization
#                             local_A, local_b = initialize_local_matrices(problem, cell)

#                             @test size(local_A, 1) == size(local_A, 2)
#                             @test length(local_b) == size(local_A, 1)

#                             # Test quadrature contributions
#                             std_cell = get_standard_cell_data(cell)
#                             diff_ops = get_cell_operators(std_cell)

#                             @test add_problem_contributions!(
#                                 local_A, local_b, problem, cell, mesh, std_cell, diff_ops)
#                         end
#                     end
#                 end
#             end
#         end
#     end

#     @testset "Solution" begin
#         @testset "Solution Properties" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)

#                     for prob_type in TEST_PROBLEM_TYPES
#                         problem = create_test_problem(prob_type, dim)
#                         solution = create_test_solution(problem, get_total_dof(mesh))

#                         @test verify_solution_properties(solution, problem)
#                         @test verify_solution_specific(solution, problem)

#                         if hasfield(typeof(problem), :boundary_conditions)
#                             @test verify_boundary_conditions(solution, problem)
#                         end
#                     end
#                 end
#             end
#         end

#         @testset "Solution Process" begin
#             for dim in TEST_DIMENSIONS
#                 @testset "$(dim)D" begin
#                     mesh, coord_sys = create_test_mesh(dim)

#                     for prob_type in TEST_PROBLEM_TYPES
#                         problem = create_test_problem(prob_type, dim)

#                         # Test full solution process
#                         try
#                             solution = solve_problem(problem, mesh)

#                             # Basic solution checks
#                             @test length(solution) == get_total_dof(mesh) * get_total_dof(problem)
#                             @test all(isfinite, solution)

#                             # Transform checks if needed
#                             if needs_transform(problem)
#                                 transformed = transform_to_physical_space(
#                                     solution, coord_sys, mesh)
#                                 @test length(transformed) == length(solution)
#                                 @test all(isfinite, transformed)
#                             end
#                         catch e
#                             @test_broken e === nothing
#                             @warn "Solution failed for $prob_type in $(dim)D" exception=e
#                         end
#                     end
#                 end
#             end
#         end
#     end

#     @testset "Cache Management" begin
#         @testset "Cache Operations" begin
#             # Test solver cache
#             @test ProblemTypes.SOLVER_CACHE.capacity > 0

#             # Test standard cell cache
#             @test ProblemTypes.STANDARD_CELL_CACHE.capacity > 0

#             for dim in TEST_DIMENSIONS
#                 mesh, coord_sys = create_test_mesh(dim)

#                 # Test cell data caching
#                 for cell in mesh.cells
#                     std_cell = get_standard_cell_data(cell)
#                     @test std_cell isa StandardKSCell

#                     # Test cache hit
#                     cached_cell = get_standard_cell_data(cell)
#                     @test cached_cell === std_cell
#                 end
#             end
#         end
#     end
# end

@testset "ProblemTypes" begin
    # Passing
	@testset "DOF Tests" begin
		include("dof_tests.jl")
	end

	@testset "Validation" begin
		include("validation_tests.jl")
	end

	# @testset "Assemble Problem" begin
	# 	include("assemble_problems_tests.jl")
	# end

	# @testset "Mesh Operations" begin
	# 	include("mesh_operations_tests.jl")
	# end

	# @testset "Basic Solve" begin
	# 	include("basic_solve_tests.jl")
	# end

	# @testset "Problem Implementation" begin
	# 	include("problem_implementation_tests.jl")
	# end

	# @testset "Integration" begin
	# 	include("integration_tests.jl")
	# end
end
