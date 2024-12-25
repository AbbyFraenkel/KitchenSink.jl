using Test
using LinearAlgebra, SparseArrays, StaticArrays
using KitchenSink.KSTypes, KitchenSink.CoordinateSystems, KitchenSink.SpectralMethods
using KitchenSink.BoundaryConditions, KitchenSink.ProblemTypes, KitchenSink.LinearSolvers
using KitchenSink.CacheManagement, KitchenSink.NumericUtilities
using BSplineKit: BSplineBasis, BSplineOrder, KnotVector, evaluate
using BSplineKit

include("../test_utils_with_deps.jl")
# Helper section at top

using StaticArrays

# Combine face node functions into one
function get_face_nodes(cell::KSCell, dim::Int, side::Symbol)
	is_neg = side == :neg
	return [
		idx for idx in keys(cell.node_map) if
		(is_neg && idx[dim] == 1) || (!is_neg && idx[dim] == cell.p[dim] + 2)
	]
end

# More efficient coordinate calculation
function get_node_coordinates_test(
	coord_system::KSCartesianCoordinates,
	cell::KSCell,
	node_index::Tuple,
)
	N = length(coord_system.ranges)

	# Calculate cell position in each dimension
	cell_pos = ntuple(d -> begin
			denominator = 1
			for i in 1:(d - 1)
				denominator *= (cell.p[i] + 2)
			end
			div((cell.id - 1), denominator) % (cell.p[d] + 2)
		end, N)

	# Calculate cell start coordinates
	cell_start = ntuple(
		d -> begin
			domain_size = coord_system.ranges[d][2] - coord_system.ranges[d][1]
			coord_system.ranges[d][1] + cell_pos[d] * domain_size / (cell.p[d] + 2)
		end, N)

	# Calculate cell size in each dimension
	cell_size = ntuple(
		d -> begin
			domain_size = coord_system.ranges[d][2] - coord_system.ranges[d][1]
			domain_size / (cell.p[d] + 2)
		end, N)

	# Calculate final node coordinates
	node_coords = ntuple(
		d -> cell_start[d] + (node_index[d] - 1) * cell_size[d] / (cell.p[d] + 1),
		N,
	)

	return SVector{N}(node_coords)
end

# Parse direction into dimension and side
function parse_direction(direction::Symbol)
	str = string(direction)
	dim = parse(Int, match(r"dim(\d+)", str).captures[1])
	side = endswith(str, "pos") ? :pos : :neg
	return (dim, side)
end

function verify_spectral_output(nodes_b, weights_b, nodes, weights, p)
	exp_nodes_b, exp_weights_b, exp_nodes, exp_weights = expected_values(p)
	return all(isapprox.(nodes_b, exp_nodes_b, atol = TEST_TOLERANCES[1])) &&
		   all(isapprox.(weights_b, exp_weights_b, atol = TEST_TOLERANCES[1])) &&
		   all(isapprox.(nodes, exp_nodes, atol = TEST_TOLERANCES[1])) &&
		   all(isapprox.(weights, exp_weights, atol = TEST_TOLERANCES[1]))
end

function verify_derivative_matrices(
	D_b::Matrix, D_i::Matrix, nodes::Vector{Float64}, p::Int, tol::Float64
)
	f_vals = test_f.(nodes)
	df_vals = test_df.(nodes)
	return all(isapprox.(D_b * f_vals, df_vals, atol = tol, rtol = tol)) &&
		   all(
		isapprox.(D_i * f_vals[2:(end - 1)], df_vals[2:(end - 1)], atol = tol, rtol = tol)
	)
end

@testset "SpectralMethods Tests" begin
	@testset "Legendre nodes and quadrature" begin
		@testset for p in TEST_POLYNOMIAL_ORDERS
			nodes_b, weights_b, nodes, weights = create_legendre_nodes_and_weights(p)
			@test verify_spectral_output(nodes_b, weights_b, nodes, weights, p)
		end
	end

	@testset "Gauss-Legendre N-D" begin
		@testset for (dim, p) in Iterators.product(TEST_DIMENSIONS, TEST_POLYNOMIAL_ORDERS)
			p_vec = ntuple(d -> p + d - 1, dim > 1 ? dim : 1)
			nodes, weights, nodes_i, weights_i = gauss_legendre_with_boundary_nd(
				p_vec; dim = dim
			)

			# Verify dimensions and lengths with broadcasting
			@test all(length.(nodes) .== p_vec .+ 2)
			@test all(length.(weights) .== p_vec .+ 2)
			@test all(length.(nodes_i) .== p_vec)
			@test all(length.(weights_i) .== p_vec)
		end
	end

	@testset "Derivative matrices" begin
		@testset for p in TEST_POLYNOMIAL_ORDERS[2:end]
			D_b, D_i = derivative_matrix!(p)
			nodes_b, _, nodes_i, _ = create_legendre_nodes_and_weights(p)
			tol = 1.5e-1 * (p - 1)
			@test verify_derivative_matrices(D_b, D_i, nodes_b, p, tol)

			# Test kth derivatives
			if p >= 3
				D_mats_b, D_mats_i = kth_derivative_matrix!(p, min(3, p - 1))
				test_functions = [test_f, test_df, test_d2f, test_d3f]

				@test all(
					map(1:min(3, p - 1)) do k
						dkf_vals = test_functions[k].(nodes_i)  # Use interior nodes
						isapprox(D_mats_i[k] * test_f.(nodes_i), dkf_vals;  # Use interior nodes
							atol = 5e-1 * 2p^k, rtol = 5e-1 * 2p^k)
					end,
				)
			end
		end
	end

	# Combine quadrature and interpolation tests
	@testset "Quadrature and Interpolation" begin
		@testset for p in TEST_POLYNOMIAL_ORDERS[2:end]
			nodes, weights = create_legendre_nodes_and_weights(p)[1:2]
			Q = quadrature_matrix(weights)

			# Test quadrature matrix properties
			@test size(Q) == (p + 2, p + 2) && Q ≈ Diagonal(weights)

			# Test integral computation
			f_vals = test_f.(nodes)
			@test isapprox(compute_integral(weights, f_vals), 2.0,
				atol = 5e-1 * 2p, rtol = 5e-1 * 2p)

			# Test interpolation at multiple points
			if p >= 5
				values = test_f.(nodes)
				test_points = [-0.5, 0.0, 0.5]  # Multiple test points
				interpolated = [
					barycentric_interpolation(nodes, values, x) for x in test_points
				]
				exact = test_f.(test_points)
				@test all(
					isapprox.(interpolated, exact,
						atol = 5e-1 * 2p, rtol = 5e-1 * 2p),
				)
			end
		end
	end

	@testset "get_or_create_spectral_properties" begin
		for p in TEST_POLYNOMIAL_ORDERS
			for continuity_order in TEST_CONTINUITY_ORDERS
				props = get_or_create_spectral_properties(p, continuity_order)

				@test props.p == p
				@test props.continuity_order == continuity_order
				@test length(props.nodes_with_boundary) == p + 2
				@test length(props.nodes_interior) == p
				@test size(props.differentiation_matrix_with_boundary) == (p + 2, p + 2)
				@test size(props.differentiation_matrix_interior) == (p, p)
				@test length(props.higher_order_diff_matrices_with_boundary) ==
					continuity_order
				@test length(props.higher_order_diff_matrices_interior) == continuity_order
			end
		end
	end

	@testset "create_standard_ks_cell" begin
		for p in TEST_POLYNOMIAL_ORDERS
			for level in TEST_REFINEMENT_LEVELS
				for dim in TEST_DIMENSIONS
					continuity_order = ntuple(_ -> 2, dim)
					cell = create_standard_ks_cell(
						ntuple(_ -> p, dim),
						level,
						dim;
						continuity_order = continuity_order,
					)
					scale_factor = 2.0^(-level + 1)

					expected_nodes_with_boundary,
					expected_weights_with_boundary,
					expected_nodes,
					expected_weights = expected_values(p)

					expected_nodes_with_boundary *= scale_factor
					expected_nodes *= scale_factor

					@test length(cell.nodes_with_boundary[1]) ==
						length(expected_nodes_with_boundary)
					@test all(
						isapprox.(
							cell.nodes_with_boundary[1],
							expected_nodes_with_boundary,
							atol = TEST_TOLERANCES[1],
						),
					)
				end
			end
		end
	end

	@testset "barycentric_weights" begin
		for p in TEST_POLYNOMIAL_ORDERS
			nodes, _, _, _ = create_legendre_nodes_and_weights(p)
			weights = barycentric_weights(nodes)

			@test length(weights) == length(nodes)
			@test all(!isinf, weights)
			@test all(!isnan, weights)
			@test all(abs(w) >= 1 for w in weights)
			expected_weights = [
				1 /
				prod(nodes[j] - nodes[k]
					 for k in 1:length(nodes) if k != j)
				for j in eachindex(nodes)]
			@test all(isapprox(weights, expected_weights; atol = TEST_TOLERANCES[1]))
		end
	end

	@testset "enforce_ck_continuity_spectral!" begin
		for p in TEST_POLYNOMIAL_ORDERS[2:end]
			for k in 1:min(2, p)
				nodes, _, _, _ = create_legendre_nodes_and_weights(p)
				D_matrices_with_boundary, _ = kth_derivative_matrix!(p, k)

				for order in 1:k
					D_original = copy(D_matrices_with_boundary[order])
					D_enforced = enforce_ck_continuity_spectral!(
						copy(D_original), order, nodes
					)

					for i in 1:order
						expected_left = [
							nodes[i]^(j - 1) / factorial(j - 1) for j in 1:(order + 1)
						]
						expected_right = [
							(nodes[end] - nodes[end - i + 1])^(j - 1) / factorial(j - 1)
							for
							j in 1:(order + 1)
						]

						@test isapprox(
							D_enforced[i, 1:(order + 1)],
							expected_left,
							atol = TEST_TOLERANCES[1],
						)
						@test isapprox(
							D_enforced[end - i + 1, (end - order):end],
							expected_right,
							atol = TEST_TOLERANCES[1],
						)
					end

					if p + 2 > 2 * order
						@test isapprox(
							D_enforced[(order + 1):(end - order), :],
							D_original[(order + 1):(end - order), :],
							atol = TEST_TOLERANCES[1],
						)
					end
				end
			end
		end
	end

	@testset "create_tensor_product_mask" begin
		for dim in TEST_DIMENSIONS
			p_vec = ntuple(_ -> 5, dim)
			continuity_order = ntuple(_ -> 2, dim)

			for is_boundary in (false, true)
				for is_fictitious in (false, true)
					mask = create_tensor_product_mask(
						p_vec,
						continuity_order,
						is_boundary,
						is_fictitious,
						dim,
					)

					@test length(mask) == dim
					@test all(length(m) == p + 2 for (m, p) in zip(mask, p_vec))

					for d in 1:dim
						if is_boundary
							@test !any(mask[d][1:continuity_order[d]])
							@test !any(mask[d][(end - continuity_order[d] + 1):end])
						else
							@test !any(mask[d][1:continuity_order[d]])
							@test !any(mask[d][(end - continuity_order[d] + 1):end])
						end

						if is_fictitious
							@test !any(mask[d][2:(end - 1)])
						end
					end
				end
			end
		end
	end

	@testset "create_ocfc_mesh" begin
		for dim in TEST_DIMENSIONS
			num_cells = ntuple(_ -> 2, dim)
			p_vec = ntuple(_ -> 3, dim)
			continuity_order = ntuple(_ -> 1, dim)
			max_level = 2
			coord_system = generate_test_coordinate_system(dim)

			mesh = create_ocfc_mesh(
				coord_system,
				num_cells,
				p_vec,
				continuity_order,
				dim,
				max_level,
			)

			@test length(mesh.cells) == prod(num_cells) * max_level
			@test all(cell.p == p_vec for cell in mesh.cells)
			@test all(cell.level <= max_level for cell in mesh.cells)
			@test all(cell.continuity_order == continuity_order for cell in mesh.cells)

			for d in 1:dim
				@test haskey(mesh.boundary_cells, Symbol("left_dim_$d"))
				@test haskey(mesh.boundary_cells, Symbol("right_dim_$d"))
			end
		end
	end

	@testset "derivative_matrix_nd" begin
		for dim in TEST_DIMENSIONS
			p_vec = ntuple(_ -> 5, dim)
			k_max = 3
			derivative_matrices = derivative_matrix_nd(collect(p_vec), dim, k_max)

			@test length(derivative_matrices) == dim
			for d in 1:dim
				@test length(derivative_matrices[d]) == k_max
				for k in 1:k_max
					@test size(derivative_matrices[d][k]) == (p_vec[d] + 2, p_vec[d] + 2)
				end
			end
		end
	end

	@testset "quadrature_matrix_nd" begin
		for dim in TEST_DIMENSIONS
			p_vec = ntuple(_ -> 5, dim)
			weights_tuple = ntuple(d -> rand(p_vec[d]), dim)
			quad_matrices = quadrature_matrix_nd(weights_tuple)

			@test length(quad_matrices) == dim
			for d in 1:dim
				@test size(quad_matrices[d]) == (p_vec[d], p_vec[d])
				@test quad_matrices[d] ≈ Diagonal(weights_tuple[d])
			end
		end
	end

	@testset "compute_integral_nd" begin
		for dim in TEST_DIMENSIONS
			p_vec = ntuple(_ -> 8, dim)
			nodes_tuple = ntuple(d -> create_legendre_nodes_and_weights(p_vec[d])[3], dim)
			weights_tuple = ntuple(d -> create_legendre_nodes_and_weights(p_vec[d])[4], dim)
			quad_matrices = quadrature_matrix_nd(weights_tuple)

			grid = generate_nd_grid(dim, p_vec[1])

			values = [test_multidim(collect(point)...) for point in grid]

			computed_integral = compute_integral_nd(values, quad_matrices)

			limits = ntuple(d -> (-1.0, 1.0), dim)
			analytical_integral = analytical_integral_multidim(limits)
			@test isapprox(
				computed_integral,
				analytical_integral,
				atol = TEST_TOLERANCES[2],
			)
		end
	end
	@testset "create_ocfc_mesh Tests" begin
		@testset "1D Mesh Tests" begin
			coord_system = KSCartesianCoordinates(((0.0, 1.0),))
			num_cells = (4,)
			p_vec = (3,)
			continuity_order = (1,)
			dim = 1
			max_level = 1

			mesh = create_ocfc_mesh(
				coord_system,
				num_cells,
				p_vec,
				continuity_order,
				dim,
				max_level,
			)

			@test length(mesh.cells) == 4
			@test all(cell.p == p_vec for cell in mesh.cells)
			@test all(cell.level == 1 for cell in mesh.cells)
			@test all(cell.continuity_order == continuity_order for cell in mesh.cells)

			# Test boundary cells
			@test length(mesh.boundary_cells[:left_dim_1]) == 1
			@test length(mesh.boundary_cells[:right_dim_1]) == 1
			@test mesh.boundary_cells[:left_dim_1][1] == 1
			@test mesh.boundary_cells[:right_dim_1][1] == 4

			# Test neighbors
			@test mesh.cells[1].neighbors[:dim1_neg] == -1
			@test mesh.cells[1].neighbors[:dim1_pos] == 2
			@test mesh.cells[4].neighbors[:dim1_neg] == 3
			@test mesh.cells[4].neighbors[:dim1_pos] == -1

			# Test node map
			@test all(length(cell.node_map) == p_vec[1] + 2 for cell in mesh.cells)

			# Test tensor product mask
			@test all(length(cell.tensor_product_mask) == 1 for cell in mesh.cells)
			@test all(
				length(cell.tensor_product_mask[1]) == p_vec[1] + 2 for cell in mesh.cells
			)
		end

		@testset "2D Mesh Tests" begin
			coord_system = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			num_cells = (3, 3)
			p_vec = (3, 3)
			continuity_order = (1, 1)
			dim = 2
			max_level = 1

			mesh = create_ocfc_mesh(
				coord_system,
				num_cells,
				p_vec,
				continuity_order,
				dim,
				max_level,
			)

			@test length(mesh.cells) == 9
			@test all(cell.p == p_vec for cell in mesh.cells)
			@test all(cell.level == 1 for cell in mesh.cells)
			@test all(cell.continuity_order == continuity_order for cell in mesh.cells)

			# Test boundary cells
			@test length(mesh.boundary_cells[:left_dim_1]) == 3
			@test length(mesh.boundary_cells[:right_dim_1]) == 3
			@test length(mesh.boundary_cells[:left_dim_2]) == 3
			@test length(mesh.boundary_cells[:right_dim_2]) == 3

			# Test neighbors for corner cell (bottom-left)
			corner_cell = mesh.cells[1]
			@test corner_cell.neighbors[:dim1_neg] == -1
			@test corner_cell.neighbors[:dim1_pos] == 2
			@test corner_cell.neighbors[:dim2_neg] == -1
			@test corner_cell.neighbors[:dim2_pos] == 4

			# Test neighbors for central cell
			central_cell = mesh.cells[5]
			@test central_cell.neighbors[:dim1_neg] == 4
			@test central_cell.neighbors[:dim1_pos] == 6
			@test central_cell.neighbors[:dim2_neg] == 2
			@test central_cell.neighbors[:dim2_pos] == 8

			# Test node map
			@test all(
				length(cell.node_map) == (p_vec[1] + 2) * (p_vec[2] + 2)
				for
				cell in mesh.cells
			)

			# Test tensor product mask
			@test all(length(cell.tensor_product_mask) == 2 for cell in mesh.cells)
			@test all(
				all(
					length(mask) == p + 2
					for
					(mask, p) in zip(cell.tensor_product_mask, p_vec)
				) for cell in mesh.cells
			)
		end

		@testset "3D Mesh Tests" begin
			coord_system = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
			num_cells = (2, 2, 2)
			p_vec = (3, 3, 3)
			continuity_order = (1, 1, 1)
			dim = 3
			max_level = 1

			mesh = create_ocfc_mesh(
				coord_system,
				num_cells,
				p_vec,
				continuity_order,
				dim,
				max_level,
			)

			@test length(mesh.cells) == 8
			@test all(cell.p == p_vec for cell in mesh.cells)
			@test all(cell.level == 1 for cell in mesh.cells)
			@test all(cell.continuity_order == continuity_order for cell in mesh.cells)

			# Test boundary cells
			for d in 1:3
				@test length(mesh.boundary_cells[Symbol("left_dim_$d")]) == 4
				@test length(mesh.boundary_cells[Symbol("right_dim_$d")]) == 4
			end

			# Test neighbors for corner cell (bottom-left-front)
			corner_cell = mesh.cells[1]
			@test corner_cell.neighbors[:dim1_neg] == -1
			@test corner_cell.neighbors[:dim1_pos] == 2
			@test corner_cell.neighbors[:dim2_neg] == -1
			@test corner_cell.neighbors[:dim2_pos] == 3
			@test corner_cell.neighbors[:dim3_neg] == -1
			@test corner_cell.neighbors[:dim3_pos] == 5

			# Test neighbors for central cell
			central_cell = mesh.cells[8]
			@test central_cell.neighbors[:dim1_neg] == 7
			@test central_cell.neighbors[:dim1_pos] == -1
			@test central_cell.neighbors[:dim2_neg] == 6
			@test central_cell.neighbors[:dim2_pos] == -1
			@test central_cell.neighbors[:dim3_neg] == 4
			@test central_cell.neighbors[:dim3_pos] == -1

			# Test node map
			@test all(
				length(cell.node_map) == (p_vec[1] + 2) * (p_vec[2] + 2) * (p_vec[3] + 2)
				for cell in mesh.cells
			)

			# Test tensor product mask
			@test all(length(cell.tensor_product_mask) == 3 for cell in mesh.cells)
			@test all(
				all(
					length(mask) == p + 2
					for
					(mask, p) in zip(cell.tensor_product_mask, p_vec)
				) for cell in mesh.cells
			)
		end
		@testset "Multi-level Mesh Tests" begin
			coord_system = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			num_cells = (2, 2)
			p_vec = (3, 3)
			continuity_order = (1, 1)
			dim = 2
			max_level = 2

			mesh = create_ocfc_mesh(
				coord_system,
				num_cells,
				p_vec,
				continuity_order,
				dim,
				max_level,
			)

			@test length(mesh.cells) == 8  # 4 cells at level 1, 4 cells at level 2
			@test count(cell.level == 1 for cell in mesh.cells) == 4
			@test count(cell.level == 2 for cell in mesh.cells) == 4

			# Test refinement
			level_1_cells = filter(cell -> cell.level == 1, mesh.cells)
			level_2_cells = filter(cell -> cell.level == 2, mesh.cells)

			@test all(cell.p == p_vec for cell in level_1_cells)
			@test all(cell.p == p_vec for cell in level_2_cells)

			# Test cell sizes (this may need adjustment based on your implementation)
			level_1_cell_size = 0.5
			level_2_cell_size = 0.25

			# Test neighbor relationships for level 1 cells
			@test all(
				haskey(cell.neighbors, :dim1_pos) &&
				(cell.neighbors[:dim1_pos] == -1 || cell.neighbors[:dim1_pos] in 1:8)
				for
				cell in level_1_cells[1:(end - 1)]
			)
			@test all(
				haskey(cell.neighbors, :dim2_pos) &&
				(cell.neighbors[:dim2_pos] == -1 || cell.neighbors[:dim2_pos] in 1:8)
				for
				cell in level_1_cells[1:2]
			)

			# Test neighbor relationships for level 2 cells
			@test all(
				haskey(cell.neighbors, :dim1_pos) &&
				(cell.neighbors[:dim1_pos] == -1 || cell.neighbors[:dim1_pos] in 1:8)
				for
				cell in level_2_cells
			)
			@test all(
				haskey(cell.neighbors, :dim2_pos) &&
				(cell.neighbors[:dim2_pos] == -1 || cell.neighbors[:dim2_pos] in 1:8)
				for
				cell in level_2_cells
			)
		end
	end
	@testset "Additional SpectralMethods Tests" begin
		@testset "map_to_reference_cell" begin
			for dim in TEST_DIMENSIONS
				coord_system = generate_test_coordinate_system(dim)
				num_cells = ntuple(_ -> 2, dim)
				p_vec = ntuple(_ -> 3, dim)
				continuity_order = ntuple(_ -> 1, dim)

				# Create a mesh with proper cells
				mesh = create_ocfc_mesh(
					coord_system,
					num_cells,
					p_vec,
					continuity_order,
					dim,
					1,
				)

				for cell in mesh.cells
					# Test points inside cell
					point = ntuple(_ -> 0.5, dim)
					mapped_point = map_to_reference_cell(point, coord_system)
					@test all(-1.0 .<= mapped_point .<= 1.0)

					# Test boundary points
					point = ntuple(_ -> 0.0, dim)
					mapped_point = map_to_reference_cell(point, coord_system)
					@test all(isapprox.(mapped_point, -1.0, atol = 1e-10))
				end
			end
		end

		@testset "get_node_coordinates" begin
			for dim in TEST_DIMENSIONS
				coord_system = generate_test_coordinate_system(dim)
				num_cells = ntuple(_ -> 2, dim)
				mesh = create_ocfc_mesh(
					coord_system,
					num_cells,
					ntuple(_ -> 3, dim),
					ntuple(_ -> 1, dim),
					dim,
					1,
				)

				@test all(
					verify_node_coordinates(
						get_node_coordinates_test(coord_system, cell, node_idx),
						cell,
						mesh,
					)
					for cell in mesh.cells
					for node_idx in keys(cell.node_map)
				)
			end
		end

		@testset "is_point_in_cell" begin
			for dim in TEST_DIMENSIONS
				coord_system = generate_test_coordinate_system(dim)
				num_cells = ntuple(_ -> 2, dim)
				mesh = create_ocfc_mesh(coord_system, num_cells, ntuple(_ -> 3, dim),
					ntuple(_ -> 1, dim), dim, 1)

				for cell in mesh.cells
					# Test center point
					center_point = [0.5 for _ in 1:dim]
					@test is_point_in_cell(cell, center_point, mesh) ==
						verify_node_coordinates(tuple(center_point...), cell, mesh)

					# Test boundary point
					boundary_point = [1.0 for _ in 1:dim]
					@test !is_point_in_cell(cell, boundary_point, mesh) ||
						verify_node_coordinates(tuple(boundary_point...), cell, mesh)
				end
			end
		end

		@testset "create_basis_functions" begin
			for p in TEST_POLYNOMIAL_ORDERS
				basis_functions = create_basis_functions(p, 2)
				nodes, _, _, _ = create_legendre_nodes_and_weights(p)

				# Test Kronecker delta property using broadcasting
				values = [basis_functions[i](nodes[j]) for i in 1:(p + 2), j in 1:(p + 2)]
				expected = Matrix{Float64}(I, p + 2, p + 2)
				@test all(isapprox.(values, expected, atol = TEST_TOLERANCES[1]))
			end
		end

		@testset "determine_scaling_factor and apply_scaling_factors!" begin
			# Test determine_scaling_factor
			@test_throws ArgumentError determine_scaling_factor(1.0, 0.0)
			diff_scale, quad_scale = determine_scaling_factor(0.0, 1.0)
			@test isapprox(diff_scale, 2.0)
			@test isapprox(quad_scale, 0.5)

			# Test apply_scaling_factors!
			D = ones(3, 3)
			Q = ones(3, 3)

			# Test apply_scaling_factors!
			D = ones(3, 3)
			Q = ones(3, 3)
			D_scaled, Q_scaled = apply_scaling_factors!(
				copy(D), copy(Q), diff_scale, quad_scale
			)
			@test all(isapprox.(D_scaled, D .* diff_scale))
			@test all(isapprox.(Q_scaled, Q .* quad_scale))
		end
		# @testset "enforce_ck_continuity_bspline!" begin
		# 	for p in TEST_POLYNOMIAL_ORDERS[2:end]
		# 		nodes, _, _, _ = create_legendre_nodes_and_weights(p)
		# 		D = derivative_matrix!(p)[1]
		# 		D_enforced = enforce_ck_continuity_bspline!(copy(D), 2, nodes, p)
		# 		@test size(D_enforced) == size(D)
		# 		@test_throws ArgumentError enforce_ck_continuity_bspline!(D, -1, nodes, p)
		# 	end
		# end
		@testset "add_standard_cell!" begin
			# Clear the cache before testing
			cell = create_standard_ks_cell((3,), 1, 1)
			@test_nowarn add_standard_cell!(cell)
			@test_throws ArgumentError add_standard_cell!(cell)  # Adding same cell twice
		end
		@testset "compute_quadrature_weight" begin
			# Test for mesh version
			mesh = create_ocfc_mesh(
				generate_test_coordinate_system(2), (2, 2), (3, 3), (1, 1), 2, 1)
			global_idx = first(first(mesh.cells).node_map)[2]
			@test compute_quadrature_weight(mesh, global_idx) isa Number

			# Test for cell_key version
			cell_key = ((3, 3), 1)
			quad_point = CartesianIndex(2, 2)
			@test compute_quadrature_weight(cell_key, quad_point) isa Number
		end
		@testset "Lagrange_polynomials" begin
			nodes = [-1.0, 0.0, 1.0]
			x = collect(-1.0:0.5:1.0)

			# Test for each basis functionI
			for i in 1:length(nodes)
				values = Lagrange_polynomials(x, nodes, i)
				@test length(values) == length(x)
				@test all(isapprox.(values[2 * i - 1], 1.0; atol = 1e-10))  # Should be 1 at its node
				@test all(
					isapprox.(values[filter(j -> j != 2 * i - 1, 1:2:5)], 0.0; atol = 1e-10)
				)  # 0 at other nodes

				# Test values at nodes other than the i-th node
				other_indices = filter(j -> j != 2 * i - 1, 1:2:5)
				@test all(isapprox.(values[other_indices], 0.0, atol = 1e-10))  # 0 at other nodes
			end

			# Test invalid index
			@test_throws ArgumentError Lagrange_polynomials(x, nodes, 4)
		end
	end
end
