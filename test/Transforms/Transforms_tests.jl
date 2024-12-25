using Test
using LinearAlgebra, SparseArrays, StaticArrays
using KitchenSink.KSTypes, KitchenSink.CoordinateSystems
using KitchenSink.SpectralMethods, KitchenSink.BoundaryConditions, KitchenSink.Transforms
using KitchenSink.CacheManagement, KitchenSink.NumericUtilities
include("../test_utils_with_deps.jl")

# # Test utilities for validating transforms
function test_transform_properties(transform::AbstractKSTransform,
	test_points::AbstractVector{<:AbstractVector{T}};
	cond_threshold::Real = 1e8, det_threshold::Real = 1e-10) where T
	results = []

	for point in test_points
		try
			J = compute_transform_jacobian(
				transform, point; cond_threshold = cond_threshold
			)
			valid, msg = verify_jacobian(
				J; cond_threshold = cond_threshold, det_threshold = det_threshold
			)

			if !valid
				push!(results, (point = point, valid = false, message = msg))
			else
				# Test inverse consistency if available
				if hasfield(typeof(transform), :inverse_map) &&
					transform.inverse_map !== nothing
					forward = transform.forward_map(point)
					back = transform.inverse_map(forward)

					if !all(isapprox.(back, point, rtol = 1e-6))
						push!(
							results,
							(point = point,
								valid = false,
								message = "Inverse mapping inconsistency"),
						)
					else
						push!(results, (point = point, valid = true, message = ""))
					end
				else
					push!(results, (point = point, valid = true, message = ""))
				end
			end
		catch e
			push!(
				results,
				(point = point,
					valid = false,
					message = "Error: $(typeof(e)): $(e.msg)"),
			)
		end
	end

	return results
end

@testset "Transforms" begin
	# Common test setup
	cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
	cart_sys2 = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))

	@testset "Basic Transform Operations" begin
		@testset "Level Scaling" begin
			@test calculate_level_scaling(1) == 1.0
			@test calculate_level_scaling(2) == 0.5
			@test calculate_level_scaling(3) == 0.25
			@test_throws ArgumentError calculate_level_scaling(0)
			@test_throws ArgumentError calculate_level_scaling(-1)

			# Test domain bounds
			@test domain_bounds_at_level(1) == (-1.0, 1.0)
			@test domain_bounds_at_level(2) == (-0.5, 0.5)
			@test domain_size_at_level(1) == 2.0
			@test domain_size_at_level(2) == 1.0
		end

		@testset "Coordinate Validation" begin
			cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			polar_sys = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
			sphere_sys = KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))

			# Replace any functions incorrectly using closures or functions instead of coordinate systems
			@test validate_coordinates_physical((0.5, 0.5), cart_sys)
			@test validate_coordinates_physical((0.5, π), polar_sys)

			# Test invalid coordinates
			@test !validate_coordinates_physical((Inf, 0.5), cart_sys)
			@test !validate_coordinates_physical((0.5, -Inf), cart_sys)
			# Test periodic coordinates
			@test validate_coordinates_physical((0.5, 3π), polar_sys)  # Beyond 2π
			@test validate_coordinates_physical((0.5, -π), polar_sys)  # Negative angle

			@test_throws ArgumentError validate_coordinates_physical((0.5, NaN), cart_sys)
		end

		@testset "Transform Types" begin
			@testset "AffineTransform" begin
				# Identity transform
				t1 = AffineTransform(Matrix(I, 2, 2), zeros(2))
				@test apply_transform(t1, (1.0, 1.0)) == (1.0, 1.0)
				@test apply_transform(t1, (1.0, 1.0), 2) == (0.5, 0.5)  # Test level scaling

				# Translation
				t2 = AffineTransform(Matrix(I, 2, 2), [1.0, 1.0])
				@test apply_transform(t2, (0.0, 0.0)) == (1.0, 1.0)
				@test apply_transform(t2, (0.0, 0.0), 2) == (1.0, 1.0)

				# Rotation with scaling
				θ = π / 4
				R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
				t3 = AffineTransform(2.0 * R, zeros(2))
				result = apply_transform(t3, (1.0, 0.0))
				@test isapprox(result[1], 2cos(θ))
				@test isapprox(result[2], 2sin(θ))
			end
		end

		@testset "Level Sets" begin
			@testset "Circle Level Sets" begin
				center = (0.0, 0.0)
				radius = 0.5
				circle = create_circle_level_set(center, radius, 1)

				@test evaluate_level_set(circle, (0.0, 0.0)) < 0  # Inside
				@test isapprox(evaluate_level_set(circle, (0.5, 0.0)), 0.0, atol = 1e-10)  # Boundary
				@test evaluate_level_set(circle, (1.0, 0.0)) > 0  # Outside

				circle2 = create_circle_level_set(center, radius, 2)
				@test isapprox(evaluate_level_set(circle2, (0.25, 0.0)), 0.0, atol = 1e-10)

				circle_with_grad = create_circle_level_set(center, radius, 1)
				grad = circle_with_grad.gradient([0.5, 0.0])
				@test isapprox(norm(grad), 1.0, atol = 1e-10)

				@test_throws ArgumentError create_circle_level_set(center, -1.0)
			end

			@testset "Rectangle Level Sets" begin
				bounds = ((-0.5, 0.5), (-0.3, 0.3))
				rect = create_rectangle_level_set(bounds, 1)

				@test evaluate_level_set(rect, (0.0, 0.0)) < 0  # Inside
				@test isapprox(evaluate_level_set(rect, (0.5, 0.0)), 0.0, atol = 1e-10)  # Boundary
				@test evaluate_level_set(rect, (1.0, 0.0)) > 0  # Outside

				# Test level scaling
				bounds2 = ((-0.25, 0.25), (-0.15, 0.15))
				rect2 = create_rectangle_level_set(bounds2, 2)
				@test isapprox(evaluate_level_set(rect2, (0.25, 0.0)), 0.0, atol = 1e-10)
			end

			@testset "Level Set Operations" begin
				circle1 = create_circle_level_set((0.0, 0.0), 0.5, 1)
				circle2 = create_circle_level_set((0.25, 0.25), 1.0, 1)

				# Union
				union_ls = union_level_set([circle1, circle2])
				@test evaluate_level_set(union_ls, (0.0, 0.0)) < 0     # Inside first circle
				@test evaluate_level_set(union_ls, (0.35, 0.0)) < 0    # Inside second circle
				@test evaluate_level_set(union_ls, (0.8, 2.0)) > 0     # Outside both

				# Intersection
				intersect_ls = intersection_level_set([circle1, circle2])
				@test evaluate_level_set(intersect_ls, (0.125, 0.0)) < 0  # Inside both
				@test evaluate_level_set(intersect_ls, (0.4, 0.4)) > 0    # Inside second only
				@test evaluate_level_set(intersect_ls, (0.6, 0.0)) > 0    # Outside both
			end
		end

		@testset "Domain Mapping" begin
			@testset "Coordinate Systems" begin
				transform = AffineTransform(Matrix(I, 2, 2), [0.25, 0.25])
				mapping = DomainMapping(;
					forward_transform = transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)

				@test_nowarn apply_transform(mapping, (0.5, 0.5))
				@test_nowarn apply_transform(mapping, (0.0, 0.5))  # Boundary
				@test_nowarn apply_transform(mapping, (1.0, 0.5))  # Boundary

				@test validate_coordinates_physical((0.5, 0.5), cart_sys)
				@test validate_coordinates_physical((0.0, 0.0), cart_sys)
				@test validate_coordinates_physical((1.0, 1.0), cart_sys)
				@test !validate_coordinates_physical((-0.1, 0.5), cart_sys)
				@test !validate_coordinates_physical((1.1, 0.5), cart_sys)
			end

			@testset "Jacobian Computation" begin
				transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
				mapping = DomainMapping(;
					forward_transform = transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)

				J = jacobian(mapping, (0.5, 0.5))
				@test size(J) == (2, 2)
				@test isapprox(J, 2.0 * Matrix(I, 2, 2))

				@test_nowarn jacobian(mapping, (0.0, 0.5))
				@test_nowarn jacobian(mapping, (1.0, 0.5))

				@test_throws ArgumentError jacobian(mapping, (-0.1, 0.5))
			end

			@testset "Polar and Spherical Coordinates" begin
				polar_sys = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
				transform = AffineTransform(Matrix(I, 2, 2), zeros(2))
				mapping = DomainMapping(;
					forward_transform = transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = polar_sys,
					physical_system = cart_sys2,
				)

				@test validate_coordinates_physical((0.5, 0.0), polar_sys)   # r=0.5, θ=0
				@test validate_coordinates_physical((0.5, 2π), polar_sys)    # r=0.5, θ=2π
				@test validate_coordinates_physical((0.5, 4π), polar_sys)    # Angular periodicity
				@test !validate_coordinates_physical((-0.1, π), polar_sys)   # Invalid radius
				@test !validate_coordinates_physical((1.1, π), polar_sys)    # Invalid radius
			end
		end

		@testset "Spectral Properties" begin
			@testset "Matrix Properties" begin
				# Test transform setup
				transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
				mapping = DomainMapping(;
					forward_transform = transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)

				# Test with valid polynomial orders
				for p in [(3, 3), (4, 4), (5, 5)]
					@testset "Polynomial orders: $p" begin
						cell = SpectralMethods.get_or_create_standard_cell(p, 1)
						transformed_cell = transform_spectral_properties(cell, mapping)

						# Test properties
						for D in transformed_cell.differentiation_matrix_with_boundary
							@test size(D) == (p[1] + 2, p[1] + 2)
							@test !any(isnan.(D))
							@test !any(isinf.(D))
							@test cond(D) < 1e8
						end
					end
				end
			end
			@testset "Higher Order Derivatives" begin
				transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
				mapping = DomainMapping(;
					forward_transform = transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)

				cell = SpectralMethods.get_or_create_standard_cell((9, 9), 1)
				transformed_cell = transform_spectral_properties(cell, mapping)

				# Test higher order matrices
				for d in 1:2
					higher_order = transformed_cell.higher_order_diff_matrices_with_boundary[d]
					@test !isempty(higher_order)
					for D in higher_order
						@test size(D) == (11, 11)  # 9 + 2 for boundary nodes
						@test !any(isnan.(D))
						@test !any(isinf.(D))
						@test cond(D) < 1e8
					end
				end
			end

			@testset "Weight Transformation" begin
				transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
				mapping = DomainMapping(;
					forward_transform = transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)

				cell = SpectralMethods.get_or_create_standard_cell((3, 3), 1)
				weights = transform_weights(cell.weights_with_boundary, cell, mapping)

				@test all(all(isfinite, w) for w in weights)
				@test all(all(w .>= 0) for w in weights)

				# Test singular transform
				singular_transform = AffineTransform(zeros(2, 2), zeros(2))
				singular_mapping = DomainMapping(;
					forward_transform = singular_transform,
					inverse_transform = inverse_transform(transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)
			end
		end
		@testset "Cache Management" begin
			# Setup common test environment
			dim = 2
			mesh, coord_sys = create_test_mesh(
				dim,                    # 2D mesh
				(2, 2),                # 2x2 cells
				(3, 3),                # polynomial order 3
				(2, 2),                # continuity order 2
				1,                      # single level
			)

			transform = create_test_affine_transform(dim)
			mapping = DomainMapping(;
				forward_transform = transform,
				inverse_transform = inverse_transform(transform),
				fictitious_system = coord_sys,
				physical_system = coord_sys,
			)

			# Test polynomial orders
			test_p_values = [(3, 3), (4, 4), (5, 5), (6, 6)]

			@testset "Basic Cache Operations" begin
				clear_transform_cache!()
				@test cache_size() == 0

				# Transform cells with different polynomial orders
				cells = Dict{NTuple{2, Int}, Vector{Union{Nothing, StandardKSCell}}}()

				for p in test_p_values
					cells[p] = Vector{Union{Nothing, StandardKSCell}}(
						nothing, length(mesh.cells)
					)
					for (i, _) in enumerate(mesh.cells)
						cells[p][i] = get_or_create_transformed_standard_cell(p, 1, mapping)
					end
				end

				# Verify cache properties for each p value
				for p in test_p_values
					valid_cells = filter(!isnothing, cells[p])
					@test !isempty(valid_cells)
					@test length(unique(valid_cells)) == 1  # Same p should reuse cells
				end

				@test cache_size() == length(test_p_values)  # One entry per unique p value

				# Test cache reuse across same p values
				transformed1 = get_or_create_transformed_standard_cell(
					test_p_values[1], 1, mapping
				)
				transformed2 = get_or_create_transformed_standard_cell(
					test_p_values[1], 1, mapping
				)
				@test transformed1 === transformed2  # Same object from cache
			end

			@testset "Error Handling and Recovery" begin
				clear_transform_cache!()

				# Invalid polynomial order
				@test_throws ArgumentError get_or_create_transformed_standard_cell(
					(2, 2),  # Invalid polynomial order < 3
					1,
					mapping,
				)
				@test cache_size() == 0

				# Test recovery with each valid p value
				for p in test_p_values
					valid_cell = get_or_create_transformed_standard_cell(p, 1, mapping)
					@test cache_size() == count(x -> x <= p, test_p_values)
				end

				# Invalid level
				@test_throws ArgumentError get_or_create_transformed_standard_cell(
					test_p_values[1],
					0,  # Invalid level
					mapping,
				)
			end

			@testset "Thread Safety" begin
				clear_transform_cache!()

				# Concurrent cell creation with different p values
				cells = Dict{NTuple{2, Int}, Vector{Union{Nothing, StandardKSCell}}}()

				# Initialize the dictionary for all test p values before threading
				for p in test_p_values
					cells[p] = Vector{Union{Nothing, StandardKSCell}}(
						nothing, length(mesh.cells)
					)
				end

				# Now use threads safely with pre-initialized dictionary
				Threads.@sync begin
					Threads.@threads for p in test_p_values
						for i in 1:length(mesh.cells)
							cells[p][i] = get_or_create_transformed_standard_cell(
								p, 1, mapping
							)
						end
					end
				end

				# Verify thread-safe creation
				@test length(keys(cells)) == length(test_p_values)
				for p in test_p_values
					@test all(!isnothing, cells[p])
					@test all(x -> x isa StandardKSCell, filter(!isnothing, cells[p]))
					@test length(unique(filter(!isnothing, cells[p]))) == 1  # Same p reuses cells
				end
				@test cache_size() == length(test_p_values)

				# Test concurrent cache clearing
				n_threads = 100
				clear_counts = Threads.Atomic{Int}(0)
				results = Vector{Int}(undef, n_threads)

				Threads.@sync begin
					Threads.@threads for i in 1:n_threads
						clear_transform_cache!()
						Threads.atomic_add!(clear_counts, 1)
						results[i] = cache_size()
					end
				end

				@test clear_counts[] == n_threads
				@test all(x -> x == 0, results)
			end
		end
		@testset "Extended Coordinate Validation" begin
			polar_sys = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))

			# Test various forms of π
			@test validate_coordinates_physical((0.5, π), polar_sys)
			@test validate_coordinates_physical((0.5, Float64(π)), polar_sys)
			@test validate_coordinates_physical((0.5, Base.MathConstants.pi), polar_sys)

			# Test fractions of π
			@test validate_coordinates_physical((0.5, π / 2), polar_sys)
			@test validate_coordinates_physical((0.5, 3π / 2), polar_sys)

			# Test with expressions involving π
			@test validate_coordinates_physical((0.5, 2π + π / 4), polar_sys)
			@test validate_coordinates_physical((0.5, -π / 4), polar_sys)

			# Test with mixed coordinate types
			cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			@test validate_coordinates_physical((0.5, big(π) / 4), cart_sys)  # Mixed BigFloat
			@test validate_coordinates_physical((Float32(0.5), sqrt(2) / 2), cart_sys)  # Mixed irrational

			# Test with more numeric
			@test validate_coordinates_physical((0.5, Rational(1, 2)), cart_sys)  # Rational
			@test validate_coordinates_physical((0.5, 1//2), cart_sys)  # Rational shorthand
			@test validate_coordinates_physical((0.5, BigInt(1)), cart_sys)  # BigInt
			@test validate_coordinates_physical((0.5, BigFloat(1.0)), cart_sys)  # BigFloat
			@test validate_coordinates_physical((0.5, Float16(0.5)), cart_sys)  # Float16
			@test validate_coordinates_physical((0.5, Float32(0.5)), cart_sys)  # Float32
			@test validate_coordinates_physical((0.5, Float64(0.5)), cart_sys)  # Float64

			# Test with spherical coordinates
			sphere_sys = KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))
			@test validate_coordinates_physical((0.5, π / 2, π), sphere_sys)
			@test validate_coordinates_physical(
				(0.5, Float64(π / 2), Float64(π)), sphere_sys
			)
			@test validate_coordinates_physical(
				(0.5, Base.MathConstants.pi / 2, Base.MathConstants.pi), sphere_sys
			)
			@test validate_coordinates_physical(
				(0.5, Rational(1, 2) * π, Rational(1, 2) * π), sphere_sys
			)
			@test validate_coordinates_physical(
				(0.5, BigFloat(π / 2), BigFloat(π)), sphere_sys
			)
			@test validate_coordinates_physical(
				(0.5, Float16(π / 2), Float16(π)), sphere_sys
			)
			@test validate_coordinates_physical(
				(0.5, Float32(π / 2), Float16(π)), sphere_sys
			)
			@test validate_coordinates_physical(
				(0.5, Float64(π / 2), Float64(π)), sphere_sys
			)
		end

		@testset "Complex Weight Transformations" begin
			cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

			# Test rotation transform
			θ = π / 4
			R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
			rot_transform = AffineTransform(R, zeros(2))
			rot_mapping = DomainMapping(;
				forward_transform = rot_transform,
				inverse_transform = inverse_transform(rot_transform),
				fictitious_system = cart_sys,
				physical_system = cart_sys,
			)

			# Create test cell with valid polynomial order
			cell = SpectralMethods.get_or_create_standard_cell((3, 3), 1)

			# Test weights under rotation
			weights = cell.weights_with_boundary
			transformed_weights = transform_weights(weights, cell, rot_mapping)

			# Weights should preserve positivity and sum
			for (w_orig, w_trans) in zip(weights, transformed_weights)
				@test all(w_trans .>= 0)
				@test isapprox(sum(w_trans), sum(w_orig), rtol = 1e-10)
			end

			# Test scaling transform
			scale_factor = 2.0
			scale_transform = AffineTransform(scale_factor * Matrix(I, 2, 2), zeros(2))
			scale_mapping = DomainMapping(;
				forward_transform = scale_transform,
				inverse_transform = inverse_transform(scale_transform),
				fictitious_system = cart_sys,
				physical_system = cart_sys,
			)

			transformed_weights_scale = transform_weights(weights, cell, scale_mapping)
			# Check scaling behavior
			for (w_orig, w_trans) in zip(weights, transformed_weights_scale)
				@test all(isapprox.(w_trans, scale_factor * w_orig))
			end
		end

		@testset "Higher Order Derivative Properties" begin
			cart_sys1 = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			cart_sys2 = KSCartesianCoordinates(((0.0, 10.0), (0.0, 10.0)))

			transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
			mapping = DomainMapping(;
				forward_transform = transform,
				inverse_transform = inverse_transform(transform),
				fictitious_system = cart_sys1,
				physical_system = cart_sys2,
			)

			cell = SpectralMethods.get_or_create_standard_cell((9, 9), 1)
			transformed_cell = transform_spectral_properties(cell, mapping)

			# Define tolerance levels
			stability_threshold = 1e-10  # Threshold for determinant-based stability check

			# Test conservation of derivative order and stability of transformed matrix
			for d in 1:2
				# Verify we have higher order matrices
				@test !isempty(cell.higher_order_diff_matrices_with_boundary[d])
				@test !isempty(transformed_cell.higher_order_diff_matrices_with_boundary[d])

				# Only test orders that exist in both original and transformed cells
				num_orders = min(
					length(cell.higher_order_diff_matrices_with_boundary[d]),
					length(transformed_cell.higher_order_diff_matrices_with_boundary[d])
				)

				for order in 1:num_orders
					D_orig = cell.higher_order_diff_matrices_with_boundary[d][order]
					D_trans = transformed_cell.higher_order_diff_matrices_with_boundary[d][order]

					# Scaling factor for the transformed matrix
					scaling = 1 / 2.0^order  # For the 2x scaling transform

						# Matrix properties
						D = D_orig .* scaling

						# 1. Stability Check
						det_D_orig = det(D)
						det_D_trans = det(D_trans)
						@test abs(det_D_trans) > stability_threshold

						# 2. Condition number check
						cond_D_orig = cond(D)
						cond_D_trans = cond(D_trans)
						@test cond_D_trans < cond_D_orig * 10  # Allow some increase but not too much

						# 3. Matrix norm comparisons with 15% tolerance
						rtol = 0.15
						@test isapprox(opnorm(D, 1), opnorm(D_trans, 1), rtol=rtol)
						@test isapprox(opnorm(D, 2), opnorm(D_trans, 2), rtol=rtol)
						@test isapprox(opnorm(D, Inf), opnorm(D_trans, Inf), rtol=rtol)
				end
			end
		end
	end
	@testset "Boundary Cell Transformations" begin
		# Setup coordinate systems and transform
		cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
		cart_sys2 = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))
		transform = AffineTransform(2.0 * Matrix{Float64}(I, 2, 2), zeros(2))

		# Create domain mapping with proper coordinate systems
		mapping = DomainMapping(;
			forward_transform = transform,
			inverse_transform = inverse_transform(transform),
			fictitious_system = cart_sys,
			physical_system = cart_sys2,
		)

		# Create test mesh with coordinate system and physical domain function
		physical_domain = x -> (all(0 .<= x .<= 1), nothing)  # Domain check function with metadata
		mesh = SpectralMethods.create_ocfc_mesh(
			cart_sys,
			(2, 2),    # num_elements
			(3, 3),    # polynomial_degree (≥ 3)
			(2, 2),    # continuity_order
			2,         # num_dims
			1,          # level
		)

		# Transform the mesh
		transformed_mesh = transform_mesh(mesh, mapping)

		# Test 1: Transformation data is stored correctly
		@test transformed_mesh.transformation_data === transform

		# Test 2: Boundary cells are preserved
		for (key, bc_cells) in mesh.boundary_cells
			@test haskey(transformed_mesh.boundary_cells, key)
			@test length(transformed_mesh.boundary_cells[key]) == length(bc_cells)
		end

		# Test 3: Physical bounds preservation
		for (cell, transformed_cell) in zip(mesh.cells, transformed_mesh.cells)
			physical_bounds = get_cell_physical_bounds(cell, mesh)
			transformed_bounds = get_cell_physical_bounds(
				transformed_cell, transformed_mesh
			)

			# Ensure we have valid bounds before comparison
			@test length(physical_bounds) == 2
			@test length(transformed_bounds) == 2
			@test all(x -> length(x) == 2, physical_bounds)
			@test all(x -> length(x) == 2, transformed_bounds)

			# Compare bounds dimension by dimension with safe access
			for dim in 1:2
				@test isapprox(
					physical_bounds[dim][1] * 2.0,
					transformed_bounds[dim][1],
					rtol = 1e-8,
					atol = 1e-8,
				)
				@test isapprox(
					physical_bounds[dim][2] * 2.0,
					transformed_bounds[dim][2],
					rtol = 1e-8,
					atol = 1e-8,
				)
			end
		end

		# Test 4: Cell properties preservation
		for cell_id in first(values(mesh.boundary_cells))
			orig_cell = mesh.cells[cell_id]
			trans_cell = transformed_mesh.cells[cell_id]
			@test orig_cell.is_leaf == trans_cell.is_leaf
			@test orig_cell.boundary_connectivity == trans_cell.boundary_connectivity
			@test orig_cell.p == trans_cell.p
			@test orig_cell.level == trans_cell.level
		end
	end
	@testset "Regression Tests" begin
		@testset "Irrational Coordinate Handling" begin
			cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			cart_sys2 = KSCartesianCoordinates(((-1.0, 1.0), (-1.0, 1.0)))

			transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
			mapping = DomainMapping(;
				forward_transform = transform,
				inverse_transform = inverse_transform(transform),
				fictitious_system = cart_sys,
				physical_system = cart_sys2,
			)

			good_transform = AffineTransform(Matrix(I, 2, 2), zeros(2))
			good_mapping = DomainMapping(;
				forward_transform = good_transform,
				inverse_transform = inverse_transform(good_transform),
				fictitious_system = cart_sys,
				physical_system = cart_sys2,
			)
			@test_nowarn jacobian(good_mapping, (0.5, 0.5))

			# Test ill-conditioned case
			@testset "Ill conditioned" begin
				bad_transform = AffineTransform(1e-8 * Matrix(I, 2, 2), zeros(2))
				bad_mapping = DomainMapping(;
					forward_transform = bad_transform,
					inverse_transform = inverse_transform(bad_transform),
					fictitious_system = cart_sys,
					physical_system = cart_sys2,
				)
			end
		end

		@testset "Level Set Intersection" begin
			circle1 = create_circle_level_set((0.0, 0.0), 0.5, 1)
			circle2 = create_circle_level_set((0.25, 0.0), 0.25, 1)

			intersect_ls = intersection_level_set([circle1, circle2])

			# Test points inside intersection
			@test evaluate_level_set(intersect_ls, (0.125, 0.0)) < 0
			@test evaluate_level_set(intersect_ls, (0.5, 0.0)) >= 0
			@test evaluate_level_set(intersect_ls, (-0.45, 0.0)) > 0
		end

		@testset "Cache Thread Safety Verification" begin
			clear_transform_cache!()
			cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			transform = AffineTransform(Matrix(I, 2, 2), zeros(2))
			mapping = DomainMapping(;
				forward_transform = transform,
				inverse_transform = inverse_transform(transform),
				fictitious_system = cart_sys,
				physical_system = cart_sys,
			)

			cells = Vector{Any}(nothing, 10)# Initialize with `nothing` instead of `undef`

			# Create cells concurrently, starting at valid polynomial orders
			Threads.@sync begin
				Threads.@threads for i in 3:10
					cells[i] = get_or_create_transformed_standard_cell(
						(i, i),
						1,
						mapping,
					)
				end
			end

			@test validate_coordinates_physical(((0.5), sqrt(2) / 2), cart_sys)

			# Verify results
			@test all(!isnothing, cells[3:10])
			@test length(unique(filter(!isnothing, cells))) == 8
			@test cache_size() == 8

			# Test concurrent cache clearing
			clear_counts = Threads.Atomic{Int}(0)
			Threads.@threads for _ in 1:100
				clear_transform_cache!()
				Threads.atomic_add!(clear_counts, 1)
			end

			@test clear_counts[] == 100
			@test cache_size() == 0
		end

		@testset "Regularize Matrix" begin
			matrix = [1.0 0.0; 0.0 1e-10]
			regularize_matrix!(matrix; max_cond = 1e8)
			@test cond(matrix) < 1e8
		end
	end
	@testset "NonlinearTransform Tests" begin
		# Test setup
		forward_map(x) = [x[1]^2, x[2]^2]
		inverse_map(x) = [sqrt(abs(x[1])), sqrt(abs(x[2]))]
		jacobian_map(x) = [2x[1] 0.0; 0.0 2x[2]]

		transform = NonlinearTransform(;
			forward_map = forward_map,
			inverse_map = inverse_map,
			jacobian = jacobian_map,
		)

		# Test forward/inverse consistency
		point = [0.5, 0.5]
		transformed = transform.forward_map(point)
		back = transform.inverse_map(transformed)
		@test all(isapprox.(point, back, rtol = 1e-6))

		# Test jacobian computation
		J = compute_transform_jacobian(transform, point)
		@test size(J) == (2, 2)
		@test isapprox(J, jacobian_map(point))

		# Test error cases
		@test_throws ArgumentError compute_transform_jacobian(transform, [1e10, 1e10])
	end

	@testset "B-spline Transformation Tests" begin
		cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
		transform = AffineTransform(2.0 * Matrix(I, 2, 2), zeros(2))
		mapping = DomainMapping(;
			forward_transform = transform,
			inverse_transform = inverse_transform(transform),
			fictitious_system = cart_sys,
			physical_system = cart_sys,
		)

		# Test null case
		@test isnothing(transform_bspline_data(nothing, mapping))

		# Test coefficient transformation
		coeffs = ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
		transformed = transform_bspline_data(coeffs, mapping)
		@test length(transformed) == 2
		@test all(isapprox.(transformed[1], 2.0 .* coeffs[1]))
		@test all(isapprox.(transformed[2], 2.0 .* coeffs[2]))
	end

	@testset "Quadrature Properties Tests" begin
		cell = SpectralMethods.get_or_create_standard_cell((4, 4), 1)

		# Test weight positivity
		@test all(all(w .>= 0) for w in cell.weights_with_boundary)

		# Test polynomial integration accuracy
		for d in 1:2
			nodes = cell.nodes_with_boundary[d]
			weights = cell.weights_with_boundary[d]

			for order in 0:cell.p[d]
				integral = sum(weights .* nodes .^ order)
				if order % 2 == 0
					expected = 2.0 / (order + 1)  # Integral of x^n from -1 to 1
					@test isapprox(integral, expected, rtol = 1e-10)
				else
					@test isapprox(integral, 0.0, atol = 1e-10)
				end
			end
		end

		@test verify_quadrature_properties(cell)
	end

	@testset "Jacobian Edge Cases" begin
		cart_sys = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

		# Test near-singular matrix
		ε = 1e-7
		# Fix: Add translation vector for AffineTransform constructor
		near_singular = AffineTransform([1.0 0.0; 0.0 ε], zeros(2))
		mapping = DomainMapping(;
			forward_transform = near_singular,
			inverse_transform = inverse_transform(near_singular),
			fictitious_system = cart_sys,
			physical_system = cart_sys,
		)

		# Should handle near-singular case through regularization
		J = jacobian(mapping, (0.5, 0.5))
		@test cond(J) < 1e8
	end
end
# end
