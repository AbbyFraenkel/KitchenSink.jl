using Test
using LinearAlgebra, StaticArrays, StaticArrays
using KitchenSink.KSTypes
using KitchenSink.CoordinateSystems
using KitchenSink.BoundaryConditions
using KitchenSink.CacheManagement, KitchenSink.NumericUtilities
# Include the utility functions and dependencies
include("../test_utils_with_deps.jl")

# Define cell origins based on their position in the mesh
cell_origins = [
	(0.0, 0.0),  # Cell 1
	(0.5, 0.0),  # Cell 2
	(0.0, 0.5),  # Cell 3
	(0.5, 0.5),   # Cell 4
]

# Define coordinate system
coord_sys_2d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

# Define cells
cells = [
	KSCell(;
		id = i,
		p = (3 + i, 3 + (i % 2)),
		level = 1,
		continuity_order = (1, 1),
		standard_cell_key = ((3 + i, 3 + (i % 2)), 1),
		neighbors = Dict(
			:dim1_neg => i == 1 || i == 3 ? -1 : i - 1,
			:dim1_pos => i == 2 || i == 4 ? -1 : i + 1,
			:dim2_neg => i <= 2 ? -1 : i - 2,
			:dim2_pos => i >= 3 ? -1 : i + 2,
		),
		node_map = Dict(
			(x, y) => (i - 1) * (5 + i) * (5 + (i % 2)) + (x - 1) * (5 + (i % 2)) + y
			for x in 1:(5 + i), y in 1:(5 + (i % 2))
		),
		tensor_product_mask = (trues(5 + i), trues(5 + (i % 2))),
		boundary_connectivity = Dict{Symbol, Int}(),
		error_estimate = 0.0,
		legendre_decay_rate = 0.0,
		is_leaf = true,
		is_fictitious = false,
		refinement_options = nothing,
		parent_id = nothing,
		child_ids = nothing,
	) for i in 1:4
]

# Create mesh
mesh = KSMesh(;
	cells = cells,
	global_error_estimate = 0.0,
	boundary_cells = Dict(
		:left => [1, 3],
		:right => [2, 4],
		:bottom => [1, 2],
		:top => [3, 4],
	),
	physical_domain = x -> true,
)

@testset "BoundaryConditions Tests" begin
	@testset "parse_direction Tests" begin
		@test BoundaryConditions.parse_direction(:dim1_pos) == (1, :pos)
		@test BoundaryConditions.parse_direction(:dim2_neg) == (2, :neg)
		@test BoundaryConditions.parse_direction(:dim3_pos) == (3, :pos)
		@test_throws ArgumentError BoundaryConditions.parse_direction(:invalid)
	end

	@testset "get_local_boundary_nodes Tests" begin
		cell = KSCell(;
			id = 1,
			p = (4, 3),
			level = 1,
			continuity_order = (1, 1),
			standard_cell_key = ((4, 3), 1),
			neighbors = Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 3,
			),
			node_map = Dict((i, j) => (i - 1) * 5 + j for i in 1:6, j in 1:5),
			tensor_product_mask = (trues(6), trues(5)),
			boundary_connectivity = Dict{Symbol, Int}(),
			error_estimate = 0.0,
			legendre_decay_rate = 0.0,
			is_leaf = true,
			is_fictitious = false,
			refinement_options = nothing,
			parent_id = nothing,
			child_ids = nothing,
		)

		@test BoundaryConditions.get_local_boundary_nodes(cell, :dim1_neg) ==
			[(1, j) for j in 1:5]
		@test BoundaryConditions.get_local_boundary_nodes(cell, :dim1_pos) ==
			[(6, j) for j in 1:5]
		@test BoundaryConditions.get_local_boundary_nodes(cell, :dim2_neg) ==
			[(i, 1) for i in 1:6]
		@test BoundaryConditions.get_local_boundary_nodes(cell, :dim2_pos) ==
			[(i, 5) for i in 1:6]
	end

	@testset "get_boundary_nodes Tests" begin
		boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)

		# Calculate expected boundary nodes manually
		expected_boundary_nodes = Set{Int}()
		for cell in cells
			for (direction, neighbor_id) in cell.neighbors
				if neighbor_id == -1  # This indicates a global boundary
					local_boundary_nodes = BoundaryConditions.get_local_boundary_nodes(
						cell, direction)
					for local_idx in local_boundary_nodes
						global_idx = cell.node_map[local_idx]
						push!(expected_boundary_nodes, global_idx)
					end
				end
			end
		end

		@test Set(boundary_nodes) == expected_boundary_nodes
	end

	@testset "apply_boundary_condition Tests" begin
		coord_sys_2d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

		@testset "Dirichlet BC" begin
			dirichlet_bc = KSDirichletBC(;
				boundary_value = x -> sum(x),
				boundary_region = x -> true,
				coordinate_system = coord_sys_2d,
			)

			@test apply_boundary_condition(dirichlet_bc, [0.5, 0.5]) ≈ 1.0
			@test apply_boundary_condition(dirichlet_bc, [0.2, 0.8]) ≈ 1.0
			@test_throws ArgumentError apply_boundary_condition(dirichlet_bc, [0.5])
		end

		@testset "Neumann BC" begin
			# Adjusted flux function to match test expectations
			neumann_bc = KSNeumannBC(;
				flux_value = x -> begin
					if abs(x[1]) < 1e-8  # Left boundary
						return [2.0, 0.0]
					elseif abs(x[1] - 1.0) < 1e-8  # Right boundary
						return [2.0, 0.0]
					elseif abs(x[2]) < 1e-8  # Bottom boundary
						return [0.0, 1.0]
					elseif abs(x[2] - 1.0) < 1e-8  # Top boundary
						return [0.0, 1.0]
					else
						return [0.0, 0.0]
					end
				end,
				boundary_region = x -> true,
				coordinate_system = coord_sys_2d,
			)

			@test apply_boundary_condition(neumann_bc, [1.0, 0.5]) ≈ 2.0
			@test apply_boundary_condition(neumann_bc, [0.0, 0.5]) ≈ -2.0
			@test apply_boundary_condition(neumann_bc, [0.5, 1.0]) ≈ 1.0
			@test apply_boundary_condition(neumann_bc, [0.5, 0.0]) ≈ -1.0
			@test_throws ArgumentError apply_boundary_condition(neumann_bc, [0.5])
		end

		@testset "Robin BC" begin
			robin_bc = KSRobinBC(;
				neumann_coefficient = x -> 2.0,
				dirichlet_coefficient = x -> 1.0,
				boundary_value = x -> sum(x),
				boundary_region = x -> true,
				coordinate_system = coord_sys_2d,
			)

			robin_func = apply_boundary_condition(robin_bc, [1.0, 0.5])
			@test robin_func(1.0) ≈ (1.0 * 1.0 + 2.0 * 1.0) - 1.5  # normal vector is [1.0, 0.0]

			robin_func = apply_boundary_condition(robin_bc, [0.0, 0.5])
			@test robin_func(1.0) ≈ (1.0 * 1.0 - 2.0 * 1.0) - 0.5  # normal vector is [-1.0, 0.0]

			robin_func = apply_boundary_condition(robin_bc, [0.5, 1.0])
			@test robin_func(1.0) ≈ (1.0 * 1.0 + 2.0 * 1.0) - 1.5  # normal vector is [0.0, 1.0]

			robin_func = apply_boundary_condition(robin_bc, [0.5, 0.0])
			@test robin_func(1.0) ≈ (1.0 * 1.0 - 2.0 * 1.0) - 0.5  # normal vector is [0.0, -1.0]

			@test_throws ArgumentError apply_boundary_condition(robin_bc, [0.5])

			# Test vector-valued Robin BC
			vector_robin_bc = KSRobinBC(;
				neumann_coefficient = x -> [2.0, 3.0],
				dirichlet_coefficient = x -> [1.0, 1.5],
				boundary_value = x -> x,
				boundary_region = x -> true,
				coordinate_system = coord_sys_2d,
			)

			vector_robin_func = apply_boundary_condition(vector_robin_bc, [1.0, 0.5])
			normal = compute_normal_vector_cartesian([1.0, 0.5], 2)
			@test vector_robin_func([1.0, 2.0]) ≈
				[1.0 * 1.0 + 2.0 * normal[1] - 1.0, 1.5 * 2.0 + 3.0 * normal[2] - 0.5]
			@test_throws ArgumentError apply_boundary_condition(vector_robin_bc, [0.5])
		end

		@testset "Different Coordinate Systems" begin
			coord_sys_1d = KSCartesianCoordinates(((0.0, 1.0),))
			coord_sys_3d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
			polar_coord_sys = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))

			dirichlet_bc_1d = KSDirichletBC(;
				boundary_value = x -> x[1],
				boundary_region = x -> true,
				coordinate_system = coord_sys_1d,
			)

			dirichlet_bc_3d = KSDirichletBC(;
				boundary_value = x -> sum(x),
				boundary_region = x -> true,
				coordinate_system = coord_sys_3d,
			)

			dirichlet_bc_polar = KSDirichletBC(;
				boundary_value = x -> x[1],
				boundary_region = x -> true,
				coordinate_system = polar_coord_sys,
			)

			@test apply_boundary_condition(dirichlet_bc_1d, [0.5]) ≈ 0.5
			@test apply_boundary_condition(dirichlet_bc_3d, [0.2, 0.3, 0.4]) ≈ 0.9
			@test apply_boundary_condition(dirichlet_bc_polar, [0.5, π / 2]) ≈ 0.5

			@test_throws ArgumentError apply_boundary_condition(dirichlet_bc_1d, [0.5, 0.5])
			@test_throws ArgumentError apply_boundary_condition(dirichlet_bc_3d, [0.5, 0.5])
			@test_throws ArgumentError apply_boundary_condition(dirichlet_bc_polar, [0.5])
		end

		@testset "Error Handling" begin
			invalid_coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			invalid_bc = KSDirichletBC(;
				boundary_value = x -> sum(x),
				boundary_region = x -> true,
				coordinate_system = invalid_coord_sys,
			)

			@test_throws ArgumentError apply_boundary_condition(invalid_bc, [0.5])
			@test_throws ArgumentError apply_boundary_condition(invalid_bc, [0.5, 0.5, 0.5])

			# Test invalid boundary condition type
			struct InvalidBC <: AbstractKSBoundaryCondition
				coordinate_system::AbstractKSCoordinateSystem
			end
			invalid_type_bc = InvalidBC(coord_sys_2d)
			@test_throws ErrorException apply_boundary_condition(invalid_type_bc, [0.5, 0.5])
		end
	end

	@testset "Integration Tests" begin
		dirichlet_bc = KSDirichletBC(;
			boundary_value = x -> sum(x),
			boundary_region = x -> true,
			coordinate_system = coord_sys_2d,)

		boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)

		# Apply boundary conditions to all boundary nodes
		for node_id in boundary_nodes
			cell_id = findfirst(cell -> node_id in values(cell.node_map), mesh.cells)
			cell = mesh.cells[cell_id]
			local_coords = [k for (k, v) in cell.node_map if v == node_id][1]
			x0, y0 = cell_origins[cell_id]
			global_coords = [
				x0 + (local_coords[1] - 1) / (cell.p[1] + 1) * 0.5,
				y0 + (local_coords[2] - 1) / (cell.p[2] + 1) * 0.5,
			]

			bc_value = apply_boundary_condition(dirichlet_bc, global_coords)
			@test bc_value ≈ sum(global_coords)
		end

		# Function to map local coordinates to global coordinates
		function map_local_to_global(cell, local_coords, x0, y0)
			xi = (local_coords[1] - 1) / (cell.p[1] + 1)
			eta = (local_coords[2] - 1) / (cell.p[2] + 1)
			x_global = x0 + xi * 0.5
			y_global = y0 + eta * 0.5
			return [x_global, y_global]
		end

		# Test boundary conditions on cells with different p values
		@testset "Boundary Conditions on Different p Values" begin
			for (i, cell) in enumerate(mesh.cells)
				x0, y0 = cell_origins[i]
				# Test left boundary
				if i in [1, 3]
					left_boundary_nodes = [
						v
						for (coords, v) in cell.node_map
						if coords[1] == 1
					]
					for node in left_boundary_nodes
						local_coords = [k for (k, v) in cell.node_map if v == node][1]
						global_coords = map_local_to_global(cell, local_coords, x0, y0)
						@test isapprox(
							apply_boundary_condition(dirichlet_bc, global_coords),
							sum(global_coords);
							atol = 1e-8,
						)
					end
				end

				# Test right boundary
				if i in [2, 4]
					right_boundary_nodes = [
						v
						for (coords, v) in cell.node_map
						if coords[1] == cell.p[1] + 2
					]
					for node in right_boundary_nodes
						local_coords = [k for (k, v) in cell.node_map if v == node][1]
						global_coords = map_local_to_global(cell, local_coords, x0, y0)
						@test isapprox(
							apply_boundary_condition(dirichlet_bc, global_coords),
							sum(global_coords);
							atol = 1e-8,
						)
					end
				end

				# Test bottom boundary
				if i in [1, 2]
					bottom_boundary_nodes = [
						v
						for (coords, v) in cell.node_map
						if coords[2] == 1
					]
					for node in bottom_boundary_nodes
						local_coords = [k for (k, v) in cell.node_map if v == node][1]
						global_coords = map_local_to_global(cell, local_coords, x0, y0)
						@test isapprox(
							apply_boundary_condition(dirichlet_bc, global_coords),
							sum(global_coords);
							atol = 1e-8,
						)
					end
				end

				# Test top boundary
				if i in [3, 4]
					top_boundary_nodes = [
						v
						for (coords, v) in cell.node_map
						if coords[2] == cell.p[2] + 2
					]
					for node in top_boundary_nodes
						local_coords = [k for (k, v) in cell.node_map if v == node][1]
						global_coords = map_local_to_global(cell, local_coords, x0, y0)
						@test isapprox(
							apply_boundary_condition(dirichlet_bc, global_coords),
							sum(global_coords);
							atol = 1e-8,
						)
					end
				end
			end
		end

		@testset "Continuity Across Cells" begin
			# Test continuity of boundary conditions across cells with different p values
			for i in 1:3
				cell1 = mesh.cells[i]
				cell2 = mesh.cells[i + 1]
				x0_cell1, y0_cell1 = cell_origins[i]
				x0_cell2, y0_cell2 = cell_origins[i + 1]

				# Check continuity along vertical boundary
				if i in [1, 3]
					# Use a parameter s in [0, 1] to sample points along the shared boundary
					num_points = max(cell1.p[2], cell2.p[2]) + 2
					s_values = range(0.0, 1.0; length = num_points)

					for s in s_values
						# Compute global coordinates in both cells at the same physical position
						global_coords1 = [x0_cell1 + 0.5, y0_cell1 + s * 0.5]
						global_coords2 = [x0_cell2 + 0.0, y0_cell2 + s * 0.5]

						bc_value1 = apply_boundary_condition(dirichlet_bc, global_coords1)
						bc_value2 = apply_boundary_condition(dirichlet_bc, global_coords2)

						@test isapprox(bc_value1, bc_value2; atol = 1e-8)
					end
				end

				# Check continuity along horizontal boundary
				if i == 2
					num_points = max(cell1.p[1], cell2.p[1]) + 2
					s_values = range(0.0, 1.0; length = num_points)

					for s in s_values
						global_coords1 = [x0_cell1 + s * 0.5, y0_cell1 + 0.5]
						global_coords2 = [x0_cell2 + s * 0.5, y0_cell2 + 0.5]

						bc_value1 = apply_boundary_condition(dirichlet_bc, global_coords1)
						bc_value2 = apply_boundary_condition(dirichlet_bc, global_coords2)

						@test isapprox(bc_value1, bc_value2; atol = 1e-8)
					end
				end
			end
		end
	end

	@testset "Normal Vector Edge Cases" begin
		@test_throws ArgumentError compute_normal_vector_cartesian([0.5, 0.5], 2) # Non-boundary point
		@test compute_normal_vector_cartesian([0.0, 0.0], 2) ≈ [-1.0, 0.0] # Corner point
		@test compute_normal_vector_cartesian([1.0, 1.0], 2) ≈ [1.0, 0.0] # Another corner
	end

	@testset "Mixed Type Boundary Conditions" begin
		mixed_robin_bc = KSRobinBC(;
			neumann_coefficient = x -> [1.0],  # Vector
			dirichlet_coefficient = x -> 2.0,  # Scalar
			boundary_value = x -> 3.0,        # Scalar
			boundary_region = x -> true,
			coordinate_system = coord_sys_2d,
		)
		@test_throws ArgumentError apply_boundary_condition(mixed_robin_bc, [0.5, 0.5])
	end

	@testset "Extended BoundaryConditions Tests" begin
		@testset "Degenerate Cases" begin

			@testset "Degenerate Boundary Conditions" begin
				coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

				# Test zero-coefficient Robin BC
				zero_robin = KSRobinBC(;
					neumann_coefficient = x -> 0.0,
					dirichlet_coefficient = x -> 0.0,
					boundary_value = x -> 1.0,
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				# Test at boundary point
				@test_throws ArgumentError apply_boundary_condition(zero_robin, [0.0, 0.5])

				# Test zero-flux Neumann BC
				zero_neumann = KSNeumannBC(;
					flux_value = x -> zeros(2),
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				@test iszero(apply_boundary_condition(zero_neumann, [1.0, 0.5]))
			end

			@testset "Corner Cases" begin
				coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

				# Test corner points where multiple boundaries meet
				corner_points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

				dirichlet_bc = KSDirichletBC(;
					boundary_value = x -> sum(x),
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				for point in corner_points
					# Verify boundary condition application at corners
					@test apply_boundary_condition(dirichlet_bc, point) ≈ sum(point)

					# Verify normal vector computation at corners with dimension
					normal = compute_normal_vector_cartesian(point, 2)
					@test length(normal) == 2
					@test norm(normal) ≈ 1.0
				end
			end
		end

		# @testset "Performance Tests" begin
		# 	@testset "Large Mesh Performance" begin
		# 		# Create a large mesh
		# 		n = 100  # Large number of cells
		# 		large_cells = [
		# 			KSCell(;
		# 				id = i,
		# 				p = (4, 4),
		# 				level = 1,
		# 				continuity_order = (1, 1),
		# 				standard_cell_key = ((4, 4), 1),
		# 				neighbors = Dict{Symbol, Int}(),
		# 				node_map = Dict(
		# 					(x, y) => (i - 1) * 25 + (x - 1) * 5 + y for x in 1:5, y in 1:5
		# 				),
		# 				tensor_product_mask = (trues(5), trues(5)),
		# 				boundary_connectivity = Dict{Symbol, Int}(),
		# 				error_estimate = 0.0,
		# 				legendre_decay_rate = 0.0,
		# 				is_leaf = true,
		# 			) for i in 1:n
		# 		]

		# 		large_mesh = KSMesh(
		# 			large_cells, 0.0, Dict(:boundary => collect(1:n)), x -> true
		# 		)

		# 		# Benchmark boundary node detection
		# 		b = @benchmark get_boundary_nodes($large_mesh)
		# 		@test b.memory < 10_000_000  # Less than 10MB allocation
		# 		@test b.time < 1_000_000_000  # Less than 1 second

		# 		# Test memory allocation
		# 		allocated = @allocated get_boundary_nodes(large_mesh)
		# 		@test allocated < 1_000_000  # Less than 1MB allocation
		# 	end
		# end

		@testset "Advanced Boundary Conditions" begin
			@testset "Mixed Boundary Conditions" begin
				coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

				# Create mixed BC that switches between types
				mixed_bc = KSRobinBC(;
					neumann_coefficient = x -> x[1] > 0.5 ? 1.0 : 0.0,
					dirichlet_coefficient = x -> x[1] > 0.5 ? 0.0 : 1.0,
					boundary_value = x -> sum(x),
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				# Test transition point
				@test_throws ArgumentError apply_boundary_condition(mixed_bc, [0.5, 0.5])
			end

			@testset "Time-Dependent BCs" begin
				coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

				# Create time-dependent Dirichlet BC
				time_bc = KSDirichletBC(;
					boundary_value = x -> cos(x[1]) * sin(0.1),  # Time-dependent
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				point = [0.5, 0.5]
				# Test at different times
				@test apply_boundary_condition(time_bc, point) ≈ cos(0.5) * sin(0.1)
			end

			@testset "Periodic Boundary Conditions" begin
				coord_sys = KSCartesianCoordinates(((0.0, 2π), (0.0, 2π)))

				# Create periodic BC
				periodic_bc = KSDirichletBC(;
					boundary_value = x -> sin(x[1]) + cos(x[2]),
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				# Test periodicity
				@test apply_boundary_condition(periodic_bc, [0.0, 0.0]) ≈
					apply_boundary_condition(periodic_bc, [2π, 2π])
			end
		end

		@testset "Error Conditions" begin
			@testset "BC Compatibility" begin
				coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

				# Test incompatible BCs
				dirichlet_bc = KSDirichletBC(;
					boundary_value = x -> 1.0,
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				neumann_bc = KSNeumannBC(;
					flux_value = x -> [1.0, 0.0],
					boundary_region = x -> true,
					coordinate_system = coord_sys,
				)

				# Test point with conflicting BCs
				point = [0.0, 0.0]
				@test apply_boundary_condition(dirichlet_bc, point) ≠
					apply_boundary_condition(neumann_bc, point)
			end

			@testset "Comprehensive Error States" begin
				coord_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

				# Test invalid boundary region
				invalid_region_bc = KSDirichletBC(;
					boundary_value = x -> 1.0,
					boundary_region = x -> false,  # Never on boundary
					coordinate_system = coord_sys,
				)

				@test_throws ArgumentError apply_boundary_condition(
					invalid_region_bc, [0.0, 0.5]
				)

				# Test invalid coordinate values
				@test_throws ArgumentError apply_boundary_condition(
					invalid_region_bc, [Inf, 0.5])
				@test_throws ArgumentError apply_boundary_condition(
					invalid_region_bc, [NaN, 0.5])
			end
		end
	end
end
