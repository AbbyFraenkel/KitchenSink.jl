
using Test, LinearAlgebra
# using ..KSTypes, ..CoordinateSystems
using KitchenSink.KSTypes, KitchenSink.CoordinateSystems

@testset "CoordinateSystems" begin

    @testset "KSCartesianCoordinates" begin
        system_2d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
        system_3d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))

        @testset "to_cartesian and from_cartesian" begin
            @test to_cartesian((0.5, 0.5), system_2d) == (0.5, 0.5)
            @test from_cartesian((0.5, 0.5), system_2d) == (0.5, 0.5)
            @test to_cartesian((0.5, 0.5, 0.5), system_3d) == (0.5, 0.5, 0.5)
            @test from_cartesian((0.5, 0.5, 0.5), system_3d) == (0.5, 0.5, 0.5)
            @test_throws MethodError to_cartesian((0.5,), system_2d)
            @test_throws MethodError from_cartesian((0.5,), system_2d)
        end

        @testset "compute_jacobian" begin
            @test compute_jacobian((0.5, 0.5), system_2d) == I(2)
            @test compute_jacobian((0.5, 0.5, 0.5), system_3d) == I(3)
        end

        @testset "map_to_reference_element and map_from_reference_element" begin
            @test map_to_reference_element((0.5, 0.5), system_2d) == (0.0, 0.0)
            @test map_from_reference_element((0.0, 0.0), system_2d) == (0.5, 0.5)
            @test map_to_reference_element((0.5, 0.5, 0.5), system_3d) == (0.0, 0.0, 0.0)
            @test map_from_reference_element((0.0, 0.0, 0.0), system_3d) == (0.5, 0.5, 0.5)
            @test_throws ArgumentError map_to_reference_element((0.5,), system_2d)
            @test_throws ArgumentError map_from_reference_element((0.0,), system_2d)
        end
    end

    @testset "KSPolarCoordinates" begin
        system_full = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))
        system_partial = KSPolarCoordinates((0.0, 1.0), nothing)

        @testset "to_cartesian and from_cartesian" begin
            @test all(isapprox.(to_cartesian((0.5, π/4), system_full), (0.5/√2, 0.5/√2), atol=1e-10))
            @test all(isapprox.(from_cartesian((0.5/√2, 0.5/√2), system_full), (0.5, π/4), atol=1e-10))
            @test all(isapprox.(to_cartesian((0.5, π/4), system_partial), (0.5*cos(π/4), 0.5*sin(π/4)), atol=1e-10))
            @test all(isapprox.(from_cartesian((0.5*cos(π/4), 0.5*sin(π/4)), system_partial), (0.5, π/4), atol=1e-10))
            @test_throws ArgumentError to_cartesian((0.5,), system_full)
            @test_throws ArgumentError from_cartesian((0.5,), system_full)
        end

        @testset "compute_jacobian" begin
            @test isapprox(compute_jacobian((1.0, π/4), system_full), [√2/2 -√2/2; √2/2 √2/2], atol=1e-10)
            @test isapprox(compute_jacobian((1.0, π/4), system_partial), [√2/2 -√2/2; √2/2 √2/2], atol=1e-10)
        end

        @testset "map_to_reference_element and map_from_reference_element" begin
            @test all(isapprox.(map_to_reference_element((0.5, π), system_full), (0.0, 0.0), atol=1e-10))
            @test all(isapprox.(map_from_reference_element((0.0, 0.0), system_full), (0.5, π), atol=1e-10))
            @test all(isapprox.(map_to_reference_element((0.5, π), system_partial), (0.0, π), atol=1e-10))
            @test all(isapprox.(map_from_reference_element((0.0, π), system_partial), (0.5, π), atol=1e-10))
            @test_throws ArgumentError map_to_reference_element((0.5,), system_full)
            @test_throws ArgumentError map_from_reference_element((0.0,), system_full)
        end
    end

    @testset "KSSphericalCoordinates" begin
        system_full = KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))
        system_partial = KSSphericalCoordinates((0.0, 1.0), (0.0, π), nothing)

        @testset "to_cartesian and from_cartesian" begin
            @test all(isapprox.(to_cartesian((1.0, π/2, 0.0), system_full), (1.0, 0.0, 0.0), atol=1e-10))
            @test all(isapprox.(from_cartesian((1.0, 0.0, 0.0), system_full), (1.0, π/2, 0.0), atol=1e-10))
            @test all(isapprox.(to_cartesian((1.0, π/2, π/4), system_partial), (√2/2, √2/2, 0.0), atol=1e-10))
            @test all(isapprox.(from_cartesian((√2/2, √2/2, 0.0), system_partial), (1.0, π/2, π/4), atol=1e-10))
            @test_throws ArgumentError to_cartesian((1.0, π/2), system_full)
            @test_throws ArgumentError from_cartesian((1.0, 0.0), system_full)
        end

        @testset "compute_jacobian" begin
            expected_jacobian = [sin(π/2)*cos(0.0)  1.0*cos(π/2)*cos(0.0)  -1.0*sin(π/2)*sin(0.0);
                                 sin(π/2)*sin(0.0)  1.0*cos(π/2)*sin(0.0)   1.0*sin(π/2)*cos(0.0);
                                 cos(π/2)          -1.0*sin(π/2)            0]
            @test isapprox(compute_jacobian((1.0, π/2, 0.0), system_full), expected_jacobian, atol=1e-10)
            @test isapprox(compute_jacobian((1.0, π/2, 0.0), system_partial), expected_jacobian, atol=1e-10)
        end

        @testset "map_to_reference_element and map_from_reference_element" begin
            @test all(isapprox.(map_to_reference_element((0.5, π/2, π), system_full), (0.0, 0.0, 0.0), atol=1e-10))
            @test all(isapprox.(map_from_reference_element((0.0, 0.0, 0.0), system_full), (0.5, π/2, π), atol=1e-10))
            @test all(isapprox.(map_to_reference_element((0.5, π/2, π), system_partial), (0.0, 0.0, π), atol=1e-10))
            @test all(isapprox.(map_from_reference_element((0.0, 0.0, π), system_partial), (0.5, π/2, π), atol=1e-10))
            @test_throws ArgumentError map_to_reference_element((0.5, π/2), system_full)
            @test_throws ArgumentError map_from_reference_element((0.0, 0.0), system_full)
        end
    end

    @testset "KSCylindricalCoordinates" begin
        system_full = KSCylindricalCoordinates((0.0, 1.0), (0.0, 2π), (-1.0, 1.0))
        system_partial = KSCylindricalCoordinates((0.0, 1.0), nothing, (-1.0, 1.0))

        @testset "to_cartesian and from_cartesian" begin
            @test all(isapprox.(to_cartesian((1.0, 0.0, 0.5), system_full), (1.0, 0.0, 0.5), atol=1e-10))
            @test all(isapprox.(from_cartesian((1.0, 0.0, 0.5), system_full), (1.0, 0.0, 0.5), atol=1e-10))
            @test all(isapprox.(to_cartesian((0.5, π/4, 0.0), system_partial), (0.5*cos(π/4), 0.5*sin(π/4), 0.0), atol=1e-10))
            @test all(isapprox.(from_cartesian((0.5*cos(π/4), 0.5*sin(π/4), 0.0), system_partial), (0.5, π/4, 0.0), atol=1e-10))
            @test_throws ArgumentError to_cartesian((1.0, 0.0), system_full)
            @test_throws ArgumentError from_cartesian((1.0, 0.0), system_full)
        end

        @testset "compute_jacobian" begin
            expected_jacobian = [cos(0.0) -1.0*sin(0.0) 0;
                                 sin(0.0)  1.0*cos(0.0) 0;
                                 0           0            1]
            @test isapprox(compute_jacobian((1.0, 0.0, 0.5), system_full), expected_jacobian, atol=1e-10)
            @test isapprox(compute_jacobian((1.0, 0.0, 0.5), system_partial), expected_jacobian, atol=1e-10)
        end

        @testset "map_to_reference_element and map_from_reference_element" begin
            @test all(isapprox.(map_to_reference_element((0.5, π, 0.0), system_full), (0.0, 0.0, 0.0), atol=1e-10))
            @test all(isapprox.(map_from_reference_element((0.0, 0.0, 0.0), system_full), (0.5, π, 0.0), atol=1e-10))
            @test all(isapprox.(map_to_reference_element((0.5, π, 0.0), system_partial), (0.0, π, 0.0), atol=1e-10))
            @test all(isapprox.(map_from_reference_element((0.0, π, 0.0), system_partial), (0.5, π, 0.0), atol=1e-10))
            @test_throws ArgumentError map_to_reference_element((0.5, π), system_full)
            @test_throws ArgumentError map_from_reference_element((0.0, 0.0), system_full)
        end
    end
end
