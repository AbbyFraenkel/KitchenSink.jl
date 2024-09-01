using Test
using KitchenSink.KSTypes, KitchenSink.CoordinateSystems

@testset "CoordinateSystems" begin
	cartesian_3d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
	polar = KSPolarCoordinates((0.0, 1.0), (0.0, 2.0 * 3.141592653589793))
	spherical = KSSphericalCoordinates((0.0, 1.0), (0.0, 3.141592653589793), (0.0, 2.0 * 3.141592653589793))
	cylindrical = KSCylindricalCoordinates((0.0, 1.0), (0.0, 2.0 * 3.141592653589793), (-1.0, 1.0))
	cart_point = (0.5, 0.5, 0.5)
	polar_point = (0.7, 3.141592653589793 / 4.0)
	spherical_point = (0.8, 3.141592653589793 / 3.0, 3.141592653589793 / 4.0)
	cylindrical_point = (0.6, 3.141592653589793 / 3.0, 0.5)

	@testset "Coordinate Transformations" begin

		# Cartesian to/from Cartesian (identity check)
		@test all(isapprox.(to_cartesian(cart_point, cartesian_3d), cart_point, atol = 1e-10))
		@test all(isapprox.(from_cartesian(cart_point, cartesian_3d), cart_point, atol = 1e-10))

		# Polar to/from Cartesian
		cart_polar = to_cartesian(polar_point, polar)
		expected_polar = (polar_point[1] * cos(polar_point[2]), polar_point[1] * sin(polar_point[2]))

		@test all(isapprox.(cart_polar, expected_polar, atol = 1e-9))  # Increase tolerance
		@test all(isapprox.(from_cartesian(cart_polar, polar), polar_point, atol = 1e-9))  # Increase tolerance

		# Spherical to/from Cartesian
		cart_spherical = to_cartesian(spherical_point, spherical)
		expected_spherical = (spherical_point[1] * sin(spherical_point[2]) * cos(spherical_point[3]),
							  spherical_point[1] * sin(spherical_point[2]) * sin(spherical_point[3]),
							  spherical_point[1] * cos(spherical_point[2]))
		@test all(isapprox.(cart_spherical, expected_spherical, atol = 1e-9))  # Increase tolerance
		@test all(isapprox.(from_cartesian(cart_spherical, spherical), spherical_point, atol = 1e-9))  # Increase tolerance

		# Cylindrical to/from Cartesian
		cart_cylindrical = to_cartesian(cylindrical_point, cylindrical)
		expected_cylindrical = (cylindrical_point[1] * cos(cylindrical_point[2]),
								cylindrical_point[1] * sin(cylindrical_point[2]),
								cylindrical_point[3])
		@test all(isapprox.(cart_cylindrical, expected_cylindrical, atol = 1e-9))  # Increase tolerance
		@test all(isapprox.(from_cartesian(cart_cylindrical, cylindrical), cylindrical_point, atol = 1e-9))  # Increase tolerance
	end

	@testset "Mapping to/from Reference Element" begin
		# Cartesian coordinates
		cartesian_ref = map_to_reference_element(cart_point, cartesian_3d)
		@test all(isapprox.(map_from_reference_element(cartesian_ref, cartesian_3d), cart_point, atol = 1e-10))

		# Polar coordinates
		polar_ref = map_to_reference_element(polar_point, polar)
		@test all(isapprox.(map_from_reference_element(polar_ref, polar), polar_point, atol = 1e-10))

		# Spherical coordinates
		spherical_ref = map_to_reference_element(spherical_point, spherical)
		@test all(isapprox.(map_from_reference_element(spherical_ref, spherical), spherical_point, atol = 1e-10))

		# Cylindrical coordinates
		cylindrical_ref = map_to_reference_element(cylindrical_point, cylindrical)
		@test all(isapprox.(map_from_reference_element(cylindrical_ref, cylindrical), cylindrical_point, atol = 1e-10))
	end

	@testset "Handling Inactive Domains" begin
		inactive_cartesian = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (true, false, true))
		inactive_point = (0.5, 0.5, 0.5)

		@test all(isnan(to_cartesian(inactive_point, inactive_cartesian)[2]))

		inactive_polar = KSPolarCoordinates((0.0, 1.0), nothing)
		inactive_point_polar = (0.5, 3.141592653589793 / 4.0)
		@test isnan(to_cartesian(inactive_point_polar, inactive_polar)[2])

		inactive_spherical = KSSphericalCoordinates((0.0, 1.0), nothing, (0.0, 2.0 * 3.141592653589793))
		inactive_point_spherical = (0.5, 3.141592653589793 / 4.0, 3.141592653589793 / 2.0)
		@test isnan(to_cartesian(inactive_point_spherical, inactive_spherical)[2])

		inactive_cylindrical = KSCylindricalCoordinates((0.0, 1.0), (0.0, 2.0 * 3.141592653589793), nothing)
		inactive_point_cylindrical = (0.5, 3.141592653589793 / 4.0, 1.0)
		@test isnan(to_cartesian(inactive_point_cylindrical, inactive_cylindrical)[3])
	end

	@testset "Edge Cases" begin
		# North pole (phi is undefined, typically set to 0)
		@test all(isapprox.(to_cartesian((1.0, 0.0, 0.0), spherical), (0.0, 0.0, 1.0), atol = 1e-9))

		# South pole (phi is undefined, typically set to 0)
		@test all(isapprox.(to_cartesian((1.0, 3.141592653589793, 0.0), spherical), (0.0, 0.0, -1.0), atol = 1e-9))

		# Origin in cylindrical coordinates
		@test all(isapprox.(to_cartesian((0.0, 0.0, 0.5), cylindrical), (0.0, 0.0, 0.5), atol = 1e-9))

		# Angle wrapping in polar coordinates
		@test all(isapprox.(to_cartesian((0.5, 2.0 * 3.141592653589793), polar), to_cartesian((0.5, 0.0), polar), atol = 1e-9))

		# Negative radii in polar and spherical coordinates
		@test all(isapprox.(to_cartesian((-1.0, 3.141592653589793), polar), to_cartesian((1.0, 0.0), polar), atol = 1e-9))
		@test all(isapprox.(to_cartesian((-1.0, 3.141592653589793, 0.0), spherical), to_cartesian((1.0, 0.0, 0.0), spherical), atol = 1e-9))
	end

	@testset "Error Handling" begin
		@test_throws ArgumentError from_cartesian((0.5, 0.5), spherical)  # Insufficient coordinates
		@test_throws ArgumentError to_cartesian((0.5,), cartesian_3d)  # Invalid number of coordinates
	end
end

# Test for KSPolarCoordinates
@testset "KSPolarCoordinates to Cartesian" begin
	polar_coords = (1.0, 3.141592653589793 / 2.0)  # r = 1, theta = pi/2 (90 degrees)
	expected_cartesian = (0.0, 1.0)  # (0, 1)
	result = to_cartesian(polar_coords, KSPolarCoordinates((0.0, 1.0), (0.0, 2.0 * 3.141592653589793)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

	polar_coords = (2.0, 3.141592653589793 / 4.0)  # r = 2, theta = pi/4 (45 degrees)
	expected_cartesian = (sqrt(2.0), sqrt(2.0))  # (~1.414, ~1.414)
	result = to_cartesian(polar_coords, KSPolarCoordinates((0.0, 2.0), (0.0, 2.0 * 3.141592653589793)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))
end

# Test for KSCylindricalCoordinates
@testset "KSCylindricalCoordinates to Cartesian" begin
	cylindrical_coords = (1.0, 3.141592653589793 / 2.0, 3.0)  # r = 1, theta = pi/2 (90 degrees), z = 3
	expected_cartesian = (0.0, 1.0, 3.0)  # (0, 1, 3)
	result = to_cartesian(cylindrical_coords, KSCylindricalCoordinates((0.0, 1.0), (0.0, 2.0 * 3.141592653589793), (0.0, 3.0)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

	cylindrical_coords = (2.0, 3.141592653589793 / 4.0, -1.0)  # r = 2, theta = pi/4 (45 degrees), z = -1
	expected_cartesian = (sqrt(2.0), sqrt(2.0), -1.0)  # (~1.414, ~1.414, -1)
	result = to_cartesian(cylindrical_coords, KSCylindricalCoordinates((0.0, 2.0), (0.0, 2.0 * 3.141592653589793), (-1.0, 1.0)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))
end

# Test for KSSphericalCoordinates
@testset "KSSphericalCoordinates to Cartesian" begin
	spherical_coords = (1.0, 3.141592653589793 / 2.0, 0.0)  # r = 1, theta = pi/2 (90 degrees), phi = 0
	expected_cartesian = (1.0, 0.0, 0.0)  # (1, 0, 0)
	result = to_cartesian(spherical_coords, KSSphericalCoordinates((0.0, 1.0), (0.0, 3.141592653589793), (0.0, 2.0 * 3.141592653589793)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

	spherical_coords = (1.0, 3.141592653589793 / 2.0, 3.141592653589793 / 2.0)  # r = 1, theta = pi/2 (90 degrees), phi = pi/2 (90 degrees)
	expected_cartesian = (0.0, 1.0, 0.0)  # (0, 1, 0)
	result = to_cartesian(spherical_coords, KSSphericalCoordinates((0.0, 1.0), (0.0, 3.141592653589793), (0.0, 2.0 * 3.141592653589793)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

	spherical_coords = (1.0, 3.141592653589793, 0.0)  # r = 1, theta = pi (180 degrees), phi = 0
	expected_cartesian = (0.0, 0.0, -1.0)  # (0, 0, -1)
	result = to_cartesian(spherical_coords, KSSphericalCoordinates((0.0, 1.0), (0.0, 3.141592653589793), (0.0, 2.0 * 3.141592653589793)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))

	spherical_coords = (1.0, 3.141592653589793 / 4.0, 3.141592653589793 / 4.0)  # r = 1, theta = pi/4 (45 degrees), phi = pi/4 (45 degrees)
	expected_cartesian = (0.5, 0.5, sqrt(2.0) / 2.0)  # (~0.5, ~0.5, ~0.707)
	result = to_cartesian(spherical_coords, KSSphericalCoordinates((0.0, 1.0), (0.0, 3.141592653589793), (0.0, 2.0 * 3.141592653589793)))
	@test all(isapprox.(result, expected_cartesian, atol = 1e-9))
end
