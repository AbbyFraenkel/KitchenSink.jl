


@testset "Coordinate Conversion" begin
    # Tests for to_cartesian
    @testset "to_cartesian tests" begin
        @testset "Valid Cases" begin
            @testset "Cartesian" begin
                @test to_cartesian([1.0, 2.0, 3.0], KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0))) == [1.0, 2.0, 3.0]
            end

            @testset "Polar" begin
                r, θ = 1.0, π / 4
                expected = [r * cos(θ), r * sin(θ)]
                @test to_cartesian([r, θ], KSPolarCoordinates{Float64}(r, θ)) == expected
            end

            @testset "Spherical" begin
                r, θ, φ = 1.0, π / 4, π / 2
                expected = [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)]
                @test to_cartesian([r, θ, φ], KSSphericalCoordinates{Float64}(r, θ, φ)) == expected
            end

            @testset "Cylindrical" begin
                r, θ, z = 1.0, π / 4, 2.0
                expected = [r * cos(θ), r * sin(θ), z]
                @test to_cartesian([r, θ, z], KSCylindricalCoordinates{Float64}(r, θ, z)) == expected
            end
        end

        @testset "Edge Cases" begin
            @testset "Polar" begin
                r, θ = 0.0, π / 4
                expected = [0.0, 0.0]
                @test to_cartesian([r, θ], KSPolarCoordinates{Float64}(r, θ)) == expected

                r, θ = 1.0, 0.0
                expected = [1.0, 0.0]
                @test to_cartesian([r, θ], KSPolarCoordinates{Float64}(r, θ)) == expected
            end

            @testset "Spherical" begin
                r, θ, φ = 0.0, π / 4, π / 2
                expected = [0.0, 0.0, 0.0]
                @test to_cartesian([r, θ, φ], KSSphericalCoordinates{Float64}(r, θ, φ)) == expected

                r, θ, φ = 1.0, 0.0, π / 2
                expected = [0.0, 0.0, 1.0]
                @test to_cartesian([r, θ, φ], KSSphericalCoordinates{Float64}(r, θ, φ)) == expected
            end

            @testset "Cylindrical" begin
                r, θ, z = 0.0, π / 4, 2.0
                expected = [0.0, 0.0, 2.0]
                @test to_cartesian([r, θ, z], KSCylindricalCoordinates{Float64}(r, θ, z)) == expected

                r, θ, z = 1.0, 0.0, 2.0
                expected = [1.0, 0.0, 2.0]
                @test to_cartesian([r, θ, z], KSCylindricalCoordinates{Float64}(r, θ, z)) == expected
            end
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError to_cartesian([1.0, 2.0], KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0)))
            @test_throws ArgumentError to_cartesian([1.0], KSPolarCoordinates{Float64}(1.0, π / 4))
            @test_throws ArgumentError to_cartesian([1.0, π / 4], KSSphericalCoordinates{Float64}(1.0, π / 4, π / 2))
        end
    end

    # Tests for from_cartesian
    @testset "from_cartesian tests" begin
        @testset "Valid Cases" begin
            @testset "Cartesian" begin
                @test from_cartesian([1.0, 2.0, 3.0], KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0))) == [1.0, 2.0, 3.0]
            end

            @testset "Polar" begin
                x, y = sqrt(2) / 2, sqrt(2) / 2
                expected = [1.0, π / 4]
                @test from_cartesian([x, y], KSPolarCoordinates{Float64}(1.0, π / 4)) == expected
            end

            @testset "Spherical" begin
                x, y, z = 0.5, 0.5, sqrt(2) / 2
                expected = [1.0, π / 4, π / 4]
                @test from_cartesian([x, y, z], KSSphericalCoordinates{Float64}(1.0, π / 4, π / 4)) == expected
            end

            @testset "Cylindrical" begin
                x, y, z = sqrt(2) / 2, sqrt(2) / 2, 2.0
                expected = [1.0, π / 4, 2.0]
                @test from_cartesian([x, y, z], KSCylindricalCoordinates{Float64}(1.0, π / 4, 2.0)) == expected
            end
        end

        @testset "Edge Cases" begin
            @testset "Polar" begin
                x, y = 0.0, 0.0
                expected = [0.0, 0.0]
                @test from_cartesian([x, y], KSPolarCoordinates{Float64}(0.0, 0.0)) == expected

                x, y = 1.0, 0.0
                expected = [1.0, 0.0]
                @test from_cartesian([x, y], KSPolarCoordinates{Float64}(1.0, 0.0)) == expected
            end

            @testset "Spherical" begin
                x, y, z = 0.0, 0.0, 0.0
                expected = [0.0, 0.0, 0.0]
                @test from_cartesian([x, y, z], KSSphericalCoordinates{Float64}(0.0, 0.0, 0.0)) == expected

                x, y, z = 0.0, 0.0, 1.0
                expected = [1.0, 0.0, 0.0]
                @test from_cartesian([x, y, z], KSSphericalCoordinates{Float64}(1.0, 0.0, 0.0)) == expected
            end

            @testset "Cylindrical" begin
                x, y, z = 0.0, 0.0, 2.0
                expected = [0.0, 0.0, 2.0]
                @test from_cartesian([x, y, z], KSCylindricalCoordinates{Float64}(0.0, 0.0, 2.0)) == expected

                x, y, z = 1.0, 0.0, 2.0
                expected = [1.0, 0.0, 2.0]
                @test from_cartesian([x, y, z], KSCylindricalCoordinates{Float64}(1.0, 0.0, 2.0)) == expected
            end
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError from_cartesian([1.0, 2.0], KSCartesianCoordinates{3,Float64}((1.0, 2.0, 3.0)))
            @test_throws ArgumentError from_cartesian([1.0], KSPolarCoordinates{Float64}(1.0, π / 4))
            @test_throws ArgumentError from_cartesian([1.0, π / 4], KSSphericalCoordinates{Float64}(1.0, π / 4, π / 2))
        end
    end
end
