using Test, LinearAlgebra
using KitchenSink.KSTypes, KitchenSink.CoordinateSystems

@testset "CoordinateSystems" begin
	cartesian_3d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
	cartesian_2d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
	polar = KSPolarCoordinates((0.0, 1.0), (0.0, 2.0 * π))
	spherical = KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2.0 * π))
	cylindrical = KSCylindricalCoordinates((0.0, 1.0), (0.0, 2.0 * π), (-1.0, 1.0))
	cart_point = (0.5, 0.5, 0.5)
	polar_point = (0.7, π / 4.0)
	spherical_point = (0.8, π / 3.0, π / 4.0)
	cylindrical_point = (0.6, π / 3.0, 0.5)

	@testset "Coordinate Transformations" begin

		# Cartesian to/from Cartesian (identity check)
		@test all(
			isapprox.(to_cartesian(cart_point, cartesian_3d), cart_point, atol = 1e-10)
		)
		@test all(
			isapprox.(from_cartesian(cart_point, cartesian_3d), cart_point, atol = 1e-10)
		)

		# Polar to/from Cartesian
		cart_polar = to_cartesian(polar_point, polar)
		expected_polar = (
			polar_point[1] * cos(polar_point[2]), polar_point[1] * sin(polar_point[2]))

		@test all(isapprox.(cart_polar, expected_polar, atol = 1e-9))
		@test all(isapprox.(from_cartesian(cart_polar, polar), polar_point, atol = 1e-9))

		# Spherical to/from Cartesian
		cart_spherical = to_cartesian(spherical_point, spherical)
		expected_spherical = (
			spherical_point[1] * sin(spherical_point[2]) * cos(spherical_point[3]),
			spherical_point[1] * sin(spherical_point[2]) * sin(spherical_point[3]),
			spherical_point[1] * cos(spherical_point[2]),
		)
		@test all(isapprox.(cart_spherical, expected_spherical, atol = 1e-9))
		@test all(
			isapprox.(
				from_cartesian(cart_spherical, spherical),
				spherical_point,
				atol = 1e-9,
			),
		)

		# Cylindrical to/from Cartesian
		cart_cylindrical = to_cartesian(cylindrical_point, cylindrical)
		expected_cylindrical = (
			cylindrical_point[1] * cos(cylindrical_point[2]),
			cylindrical_point[1] * sin(cylindrical_point[2]),
			cylindrical_point[3],
		)
		@test all(isapprox.(cart_cylindrical, expected_cylindrical, atol = 1e-9))
		@test all(
			isapprox.(
				from_cartesian(cart_cylindrical, cylindrical),
				cylindrical_point,
				atol = 1e-9,
			),
		)
	end

	@testset "Mapping to/from Reference Element" begin
		# Cartesian coordinates
		cartesian_ref = map_to_reference_cell(cart_point, cartesian_3d)
		@test all(
			isapprox.(
				map_from_reference_cell(cartesian_ref, cartesian_3d),
				cart_point,
				atol = 1e-10,
			),
		)

		# Polar coordinates
		polar_ref = map_to_reference_cell(polar_point, polar)
		@test all(
			isapprox.(map_from_reference_cell(polar_ref, polar), polar_point, atol = 1e-10)
		)

		# Spherical coordinates
		spherical_ref = map_to_reference_cell(spherical_point, spherical)
		@test all(
			isapprox.(
				map_from_reference_cell(spherical_ref, spherical),
				spherical_point,
				atol = 1e-10,
			),
		)

		# Cylindrical coordinates
		cylindrical_ref = map_to_reference_cell(cylindrical_point, cylindrical)
		@test all(
			isapprox.(
				map_from_reference_cell(cylindrical_ref, cylindrical),
				cylindrical_point,
				atol = 1e-10,
			),
		)
	end

	@testset "Handling Inactive Domains" begin
		inactive_polar = KSPolarCoordinates((0.0, 1.0), nothing)
		inactive_point_polar = (0.5, π / 4.0)
		@test isnan(to_cartesian(inactive_point_polar, inactive_polar)[2])

		inactive_spherical = KSSphericalCoordinates((0.0, 1.0), nothing, (0.0, 2.0 * π))
		inactive_point_spherical = (0.5, π / 4.0, π / 2.0)
		@test isnan(to_cartesian(inactive_point_spherical, inactive_spherical)[2])

		inactive_cylindrical = KSCylindricalCoordinates((0.0, 1.0), (0.0, 2.0 * π), nothing)
		inactive_point_cylindrical = (0.5, π / 4.0, 1.0)
		@test isnan(to_cartesian(inactive_point_cylindrical, inactive_cylindrical)[3])
	end

	@testset "Edge Cases" begin
		# North pole (phi is undefined, typically set to 0)
		@test all(
			isapprox.(
				to_cartesian((1.0, 0.0, 0.0), spherical),
				(0.0, 0.0, 1.0),
				atol = 1e-9,
			),
		)

		# South pole (phi is undefined, typically set to 0)
		@test all(
			isapprox.(
				to_cartesian((1.0, π, 0.0), spherical),
				(0.0, 0.0, -1.0),
				atol = 1e-9,
			),
		)

		# Origin in cylindrical coordinates
		@test all(
			isapprox.(
				to_cartesian((0.0, 0.0, 0.5), cylindrical),
				(0.0, 0.0, 0.5),
				atol = 1e-9,
			),
		)

		# Angle wrapping in polar coordinates
		@test all(
			isapprox.(
				to_cartesian((0.5, 2.0 * π), polar),
				to_cartesian((0.5, 0.0), polar),
				atol = 1e-9,
			),
		)

		# Negative radii in polar and spherical coordinates
		@test all(
			isapprox.(
				to_cartesian((-1.0, π), polar),
				to_cartesian((1.0, 0.0), polar),
				atol = 1e-9,
			),
		)
		@test all(
			isapprox.(
				to_cartesian((-1.0, π, 0.0), spherical),
				to_cartesian((1.0, 0.0, 0.0), spherical),
				atol = 1e-9,
			),
		)
	end

	@testset "Error Handling" begin
		@test_throws ArgumentError from_cartesian((0.5, 0.5), spherical)  # Insufficient coordinates
		@test_throws ArgumentError to_cartesian((0.5,), cartesian_3d)  # Invalid number of coordinates
		@test_throws ArgumentError to_cartesian((0.5, 0.5, 0.5), polar)  # Invalid number of coordinates for polar
		@test_throws ArgumentError to_cartesian((0.5, 0.5, 0.5, 0.5), spherical)  # Too many coordinates for spherical
		@test_throws ArgumentError to_cartesian((0.5, 0.5, 0.5, 0.5), cylindrical)  # Too many coordinates for cylindrical
	end

	# Test for KSPolarCoordinates
	@testset "KSPolarCoordinates to Cartesian" begin
		polar_coords = (1.0, π / 2.0)  # r = 1, theta = pi/2 (90 degrees)
		expected_cartesian = (0.0, 1.0)  # (0, 1)
		result = to_cartesian(polar_coords, KSPolarCoordinates((0.0, 1.0), (0.0, 2.0 * π)))
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

		polar_coords = (2.0, π / 4.0)  # r = 2, theta = pi/4 (45 degrees)
		expected_cartesian = (sqrt(2.0), sqrt(2.0))  # (~1.414, ~1.414)
		result = to_cartesian(polar_coords, KSPolarCoordinates((0.0, 2.0), (0.0, 2.0 * π)))
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))
	end

	# Test for KSCylindricalCoordinates
	@testset "KSCylindricalCoordinates to Cartesian" begin
		cylindrical_coords = (1.0, π / 2.0, 3.0)  # r = 1, theta = pi/2 (90 degrees), z = 3
		expected_cartesian = (0.0, 1.0, 3.0)  # (0, 1, 3)
		result = to_cartesian(
			cylindrical_coords,
			KSCylindricalCoordinates((0.0, 1.0), (0.0, 2.0 * π), (0.0, 3.0)),
		)
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

		cylindrical_coords = (2.0, π / 4.0, -1.0)  # r = 2, theta = pi/4 (45 degrees), z = -1
		expected_cartesian = (sqrt(2.0), sqrt(2.0), -1.0)  # (~1.414, ~1.414, -1)
		result = to_cartesian(
			cylindrical_coords,
			KSCylindricalCoordinates((0.0, 2.0), (0.0, 2.0 * π), (-1.0, 1.0)),
		)
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))
	end

	# Test for KSSphericalCoordinates
	@testset "KSSphericalCoordinates to Cartesian" begin
		spherical_coords = (1.0, π / 2.0, 0.0)  # r = 1, theta = pi/2 (90 degrees), phi = 0
		expected_cartesian = (1.0, 0.0, 0.0)  # (1, 0, 0)
		result = to_cartesian(
			spherical_coords,
			KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2.0 * π)),
		)
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

		spherical_coords = (1.0, π / 2.0, π / 2.0)  # r = 1, theta = pi/2 (90 degrees), phi = pi/2 (90 degrees)
		expected_cartesian = (0.0, 1.0, 0.0)  # (0, 1, 0)
		result = to_cartesian(
			spherical_coords,
			KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2.0 * π)),
		)
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

		spherical_coords = (1.0, π, 0.0)  # r = 1, theta = pi (180 degrees), phi = 0
		expected_cartesian = (0.0, 0.0, -1.0)  # (0, 0, -1)
		result = to_cartesian(
			spherical_coords,
			KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2.0 * π)),
		)
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

		spherical_coords = (1.0, π / 4.0, π / 4.0)  # r = 1, theta = pi/4 (45 degrees), phi = pi/4 (45 degrees)
		expected_cartesian = (0.5, 0.5, sqrt(2.0) / 2.0)  # (~0.5, ~0.5, ~0.707)
		result = to_cartesian(
			spherical_coords,
			KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2.0 * π)),
		)
		@test all(isapprox.(result, expected_cartesian, atol = 1e-9))
	end

	@testset "scale_value and unscale_value" begin
		# Cartesian
		scaled_value = scale_value(0.5, (0.0, 1.0))
		@test isapprox(scaled_value, 0.5, atol = 1e-10)

		unscaled_value = unscale_value(scaled_value, (0.0, 1.0))
		@test isapprox(unscaled_value, 0.5, atol = 1e-10)

		# Polar (testing with angular coordinates)
		scaled_theta = scale_value(π / 4, (0.0, 2.0 * π))
		@test isapprox(scaled_theta, 1 / 8, atol = 1e-10)

		unscaled_theta = unscale_value(scaled_theta, (0.0, 2.0 * π))
		@test isapprox(unscaled_theta, π / 4, atol = 1e-10)
	end

	@testset "map_coordinate and map_coordinate_back" begin
		# Cartesian
		mapped_coord = map_coordinate(0.5, (0.0, 1.0), false)
		@test isapprox(mapped_coord, 0.0, atol = 1e-10)

		unmapped_coord = map_coordinate_back(mapped_coord, (0.0, 1.0), false)
		@test isapprox(unmapped_coord, 0.5, atol = 1e-10)

		# Polar (testing with angular coordinates)
		mapped_theta = map_coordinate(π / 4, (0.0, 2.0 * π), true)
		@test isapprox(mapped_theta, -0.75, atol = 1e-10)

		unmapped_theta = map_coordinate_back(mapped_theta, (0.0, 2.0 * π), true)
		@test isapprox(unmapped_theta, π / 4, atol = 1e-10)
	end

	# Additional Robustness Tests

	@testset "Additional Robustness Tests" begin

		# Range Boundaries
		@testset "Range Boundaries" begin
			# Test boundary values
			@test all(isapprox.(to_cartesian((0.0, 0.0), polar), (0.0, 0.0), atol = 1e-9))
			@test all(isapprox.(to_cartesian((1.0, 0.0), polar), (1.0, 0.0), atol = 1e-9))
			@test all(
				isapprox.(to_cartesian((1.0, 2.0 * π), polar), (1.0, 0.0), atol = 1e-9)
			)  # Testing angle wrapping

			# Testing boundary values for spherical coordinates
			@test all(
				isapprox.(
					to_cartesian((1.0, 0.0, 0.0), spherical),
					(0.0, 0.0, 1.0),
					atol = 1e-9,
				),
			)
			@test all(
				isapprox.(
					to_cartesian((1.0, π, 0.0), spherical),
					(0.0, 0.0, -1.0),
					atol = 1e-9,
				),
			)
		end

		# Degenerate Cases
		@testset "Invalid Cases" begin
			# Zero-length ranges
			@test_throws ArgumentError KSPolarCoordinates((1.0, 1.0), (0.0, π))  # Invalid range
			@test_throws ArgumentError KSPolarCoordinates((0.0, 1.0), (π, π))  # Invalid angular range

			@test_throws ArgumentError to_cartesian(
				(1.0, π / 2.0),
				KSPolarCoordinates((-1.0, -0.5), (0.0, 2.0 * π)),
			)  # Invalid negative radius
			@test_throws ArgumentError KSSphericalCoordinates(
				(0.0, 0.0),
				(0.0, π),
				(0.0, 2.0 * π),
			)  # Invalid zero radius
		end

		# NaN Handling in Inactive Domains
		@testset "NaN Handling in Inactive Domains" begin
			# Ensure NaN propagation in inactive domains
			polar_inactive = KSPolarCoordinates((0.0, 1.0), nothing)
			spherical_inactive_theta = KSSphericalCoordinates(
				(0.0, 1.0), nothing, (0.0, 2.0 * π))

			@test_throws ArgumentError to_cartesian((0.5, NaN), polar_inactive)

			@test_throws ArgumentError to_cartesian((0.5, NaN, π / 4.0), spherical_inactive_theta)
		end

		# Type Stability and Promotion
		@testset "Type Stability and Promotion" begin
			# Mixed-type inputs
			polar_mixed_types = KSPolarCoordinates((1, 2.0), (π / 4, π))
			cylindrical_mixed_types = KSCylindricalCoordinates(
				(0.0, 2.0), (π / 2, 2π), (1, 5.0))

			@test typeof(polar_mixed_types.r[1]) == Float64
			@test typeof(cylindrical_mixed_types.z[2]) == Float64
		end
	end
	@testset "Normal Vector Computation" begin
		# Cartesian normal vectors
		@test compute_normal_vector(cartesian_2d, (0.0, 0.5)) == (-1.0, 0.0)  # Left boundary
		@test compute_normal_vector(cartesian_2d, (1.0, 0.5)) == (1.0, 0.0)   # Right boundary
		@test compute_normal_vector(cartesian_2d, (0.5, 0.0)) == (0.0, -1.0)  # Bottom boundary
		@test compute_normal_vector(cartesian_2d, (0.5, 1.0)) == (0.0, 1.0)   # Top boundary

		# Polar normal vectors
		@test all(
			isapprox.(compute_normal_vector(polar, (1.0, π / 4)), (cos(π / 4), sin(π / 4)))
		)  # Outer radius
		@test all(
			isapprox.(compute_normal_vector(polar, (0.5, 0.0)), (-sin(0.0), cos(0.0)))
		)  # θ=0 boundary

		# Spherical normal vectors
		@test all(
			isapprox.(
				compute_normal_vector(spherical, (1.0, π / 2, π / 4)),
				(sin(π / 2) * cos(π / 4), sin(π / 2) * sin(π / 4), cos(π / 2)),
			),
		)  # Outer radius

		# Cylindrical normal vectors
		@test all(
			isapprox.(
				compute_normal_vector(cylindrical, (1.0, π / 4, 0.5)),
				(cos(π / 4), sin(π / 4), 0.0),
			),
		)  # Outer radius
		@test all(
			isapprox.(
				compute_normal_vector(cylindrical, (0.5, 0.0, 1.0)),
				(0.0, 0.0, 1.0),
			),
		)  # Top boundary
	end
	@testset "Coordinate Transform Functions" begin
		# Test cartesian transform
		transform_to_ref, transform_from_ref = get_coordinate_transform(cartesian_2d)
		point = [0.5, 0.5]
		@test all(isapprox.(transform_to_ref(point), [0.0, 0.0], atol = 1e-10))
		@test all(isapprox.(transform_from_ref([0.0, 0.0]), point, atol = 1e-10))

		# Test polar transform
		transform_to_ref, transform_from_ref = get_coordinate_transform(polar)
		point = [0.5 * cos(π / 4), 0.5 * sin(π / 4)]
		ref_point = transform_to_ref(point)
		@test all(isapprox.(transform_from_ref(ref_point), point, atol = 1e-10))

		# Test with invalid inputs
		@test_throws ArgumentError get_coordinate_transform("invalid")
	end
	@testset "Additional Error Cases" begin
		# Invalid coordinate system configurations
		@test_throws ArgumentError KSPolarCoordinates((1.0, 0.0), (0.0, π))  # Invalid radius range
		@test_throws ArgumentError KSSphericalCoordinates((0.0, 1.0), (π, 0.0), (0.0, 2π))  # Invalid theta range

		# NaN handling
		@test_throws ArgumentError to_cartesian((NaN, 0.0), polar)
		@test_throws ArgumentError from_cartesian((0.0, NaN), polar)

		# Inf handling
		@test_throws ArgumentError map_to_reference_cell((Inf, 0.0), cartesian_2d)
		@test_throws ArgumentError map_from_reference_cell((0.0, Inf), cartesian_2d)
	end
end
