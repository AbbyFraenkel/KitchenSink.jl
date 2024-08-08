

@testset "Collocation Points" begin
    # Tests for create_nodes
    @testset "create_nodes tests" begin
        @testset "Valid Cases" begin
            @testset "Default Interval [0, 1]" begin
                p = 5
                known_nodes, known_weights = gausslegendre(p - 2)
                known_nodes = vcat(-1.0, known_nodes, 1.0)
                known_weights = vcat(0.0, known_weights, 0.0)
                transformed_nodes = (1.0 .* (known_nodes .+ 1.0) / 2)
                transformed_weights = (1.0 .* known_weights / 2)
                nodes, weights = create_nodes(p, 0.0, 1.0)

                @test length(nodes) === length(transformed_nodes)
                @test length(weights) === length(transformed_weights)
                @test all(isapprox.(nodes, transformed_nodes, atol=1e-8))
                @test all(isapprox.(weights, transformed_weights, atol=1e-8))
            end

            @testset "Custom Interval [-1, 1]" begin
                p = 5
                known_nodes, known_weights = gausslegendre(p - 2)
                known_nodes = vcat(-1.0, known_nodes, 1.0)
                known_weights = vcat(0.0, known_weights, 0.0)
                nodes, weights = create_nodes(p, -1.0, 1.0)

                @test length(nodes) == length(known_nodes)
                @test length(weights) == length(known_weights)
                @test all(isapprox.(nodes, known_nodes, atol=1e-8))
                @test all(isapprox.(weights, known_weights, atol=1e-8))
            end
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError create_nodes(2, 1, 2)
        end

        @testset "Edge Cases" begin
            @testset "Polynomial Degree p = 3" begin
                p = 3
                known_nodes, known_weights = gausslegendre(p - 2)
                known_nodes = vcat(-1.0, known_nodes, 1.0)
                known_weights = vcat(0.0, known_weights, 0.0)
                transformed_nodes = (1.0 .* (known_nodes .+ 1.0) / 2)
                transformed_weights = (1.0 .* known_weights / 2)
                nodes, weights = create_nodes(p, 0.0, 1.0)

                @test length(nodes) == length(transformed_nodes)
                @test length(weights) == length(transformed_weights)
                @test all(isapprox.(nodes, transformed_nodes, atol=1e-8))
                @test all(isapprox.(weights, transformed_weights, atol=1e-8))
            end

            @testset "Polynomial Degree p = 4" begin
                p = 4
                known_nodes, known_weights = gausslegendre(p - 2)
                known_nodes = vcat(-1.0, known_nodes, 1.0)
                known_weights = vcat(0.0, known_weights, 0.0)
                transformed_nodes = (1.0 .* (known_nodes .+ 1.0) / 2)
                transformed_weights = (1.0 .* known_weights / 2)
                nodes, weights = create_nodes(p, 0.0, 1.0)

                @test length(nodes) == length(transformed_nodes)
                @test length(weights) == length(transformed_weights)
                @test all(isapprox.(nodes, transformed_nodes, atol=1e-8))
                @test all(isapprox.(weights, transformed_weights, atol=1e-8))
            end
        end
    end

    # Tests for gauss_legendre_with_boundary_nd
    @testset "Collocation Points" begin
        # Tests for create_nodes
        @testset "create_nodes tests" begin
            @testset "Valid Cases" begin
                @testset "Default Interval [0, 1]" begin
                    p = 5
                    known_nodes, known_weights = gausslegendre(p - 2)
                    known_nodes = vcat(-1.0, known_nodes, 1.0)
                    known_weights = vcat(0.0, known_weights, 0.0)
                    transformed_nodes = (1.0 .* (known_nodes .+ 1.0) / 2)
                    transformed_weights = (1.0 .* known_weights / 2)
                    nodes, weights = create_nodes(p, 0.0, 1.0)

                    @test length(nodes) === length(transformed_nodes)
                    @test length(weights) === length(transformed_weights)
                    @test all(isapprox.(nodes, transformed_nodes, atol=1e-8))
                    @test all(isapprox.(weights, transformed_weights, atol=1e-8))
                end

                @testset "Custom Interval [-1, 1]" begin
                    p = 5
                    known_nodes, known_weights = gausslegendre(p - 2)
                    known_nodes = vcat(-1.0, known_nodes, 1.0)
                    known_weights = vcat(0.0, known_weights, 0.0)
                    nodes, weights = create_nodes(p, -1.0, 1.0)

                    @test length(nodes) == length(known_nodes)
                    @test length(weights) == length(known_weights)
                    @test all(isapprox.(nodes, known_nodes, atol=1e-8))
                    @test all(isapprox.(weights, known_weights, atol=1e-8))
                end
            end

            @testset "Invalid Cases" begin
                @test_throws ArgumentError create_nodes(2, 1, 2)
            end

            @testset "Edge Cases" begin
                @testset "Polynomial Degree p = 3" begin
                    p = 3
                    known_nodes, known_weights = gausslegendre(p - 2)
                    known_nodes = vcat(-1.0, known_nodes, 1.0)
                    known_weights = vcat(0.0, known_weights, 0.0)
                    transformed_nodes = (1.0 .* (known_nodes .+ 1.0) / 2)
                    transformed_weights = (1.0 .* known_weights / 2)
                    nodes, weights = create_nodes(p, 0.0, 1.0)

                    @test length(nodes) == length(transformed_nodes)
                    @test length(weights) == length(transformed_weights)
                    @test all(isapprox.(nodes, transformed_nodes, atol=1e-8))
                    @test all(isapprox.(weights, transformed_weights, atol=1e-8))
                end

                @testset "Polynomial Degree p = 4" begin
                    p = 4
                    known_nodes, known_weights = gausslegendre(p - 2)
                    known_nodes = vcat(-1.0, known_nodes, 1.0)
                    known_weights = vcat(0.0, known_weights, 0.0)
                    transformed_nodes = (1.0 .* (known_nodes .+ 1.0) / 2)
                    transformed_weights = (1.0 .* known_weights / 2)
                    nodes, weights = create_nodes(p, 0.0, 1.0)

                    @test length(nodes) == length(transformed_nodes)
                    @test length(weights) == length(transformed_weights)
                    @test all(isapprox.(nodes, transformed_nodes, atol=1e-8))
                    @test all(isapprox.(weights, transformed_weights, atol=1e-8))
                end
            end
        end

        # Tests for gauss_legendre_with_boundary_nd
        @testset "gauss_legendre_with_boundary_nd tests" begin
            @testset "Valid Cases" begin
                @testset "Default Interval [0, 1]" begin
                    degree, dim = 5, 3
                    points, weights = gauss_legendre_with_boundary_nd(degree, dim, zeros(dim), ones(dim))

                    @test length(points) == dim
                    @test length(weights) == dim
                    @test all(length.(points) .== degree)
                    @test all(length.(weights) .== degree)

                    # Check against known values for each dimension
                    for i in 1:dim
                        known_nodes, known_weights = create_nodes(degree)
                        @test all(isapprox.(points[i], known_nodes, atol=1e-8))
                        @test all(isapprox.(weights[i], known_weights, atol=1e-8))
                    end
                end

                @testset "Custom Intervals" begin
                    degree, dim = 5, 3
                    a = [-1.0, 0.0, 2.0]
                    b = [1.0, 2.0, 3.0]
                    points, weights = gauss_legendre_with_boundary_nd(degree, dim, a, b)

                    @test length(points) == dim
                    @test length(weights) == dim
                    @test all(length.(points) .== degree)
                    @test all(length.(weights) .== degree)

                    # Check against known values for each dimension
                    for i in 1:dim
                        known_nodes, known_weights = create_nodes(degree, a[i], b[i])
                        @test all(isapprox.(points[i], known_nodes, atol=1e-8))
                        @test all(isapprox.(weights[i], known_weights, atol=1e-8))
                    end
                end
            end

            @testset "Invalid Cases" begin
                @test_throws ArgumentError gauss_legendre_with_boundary_nd(5, 3, [0.0, 0.0], [1.0, 1.0, 1.0])
            end

            @testset "Edge Cases" begin
                @testset "Single Dimension" begin
                    degree, dim = 5, 1
                    points, weights = gauss_legendre_with_boundary_nd(degree, dim, zeros(dim), ones(dim))

                    @test length(points) == dim
                    @test length(weights) == dim
                    @test all(length.(points) .== degree)
                    @test all(length.(weights) .== degree)

                    known_nodes, known_weights = create_nodes(degree)
                    @test all(isapprox.(points[1], known_nodes, atol=1e-8))
                    @test all(isapprox.(weights[1], known_weights, atol=1e-8))
                end

                @testset "Multiple Dimensions with Varying Intervals" begin
                    degree, dim = 4, 2
                    a = [0.0, -1.0]
                    b = [1.0, 1.0]
                    points, weights = gauss_legendre_with_boundary_nd(degree, dim, a, b)

                    @test length(points) == dim
                    @test length(weights) == dim
                    @test all(length.(points) .== degree)
                    @test all(length.(weights) .== degree)

                    for i in 1:dim
                        known_nodes, known_weights = create_nodes(degree, a[i], b[i])
                        @test all(isapprox.(points[i], known_nodes, atol=1e-8))
                        @test all(isapprox.(weights[i], known_weights, atol=1e-8))
                    end
                end
            end
        end
    end
end
