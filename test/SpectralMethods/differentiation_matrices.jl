@testset "Differentiation Matrices" begin
    # Tests for barycentric_weights

    # Tests for derivative_matrix
    @testset "derivative_matrix tests" begin
        @testset "Valid Cases" begin
            f(x) = sin.(x)
            df(x) = cos.(x)
            nodes, _ = create_nodes(15, 0.0, 2.0 * pi)
            D = derivative_matrix(nodes)
            @test all(isapprox.(df(nodes), D * f(nodes), atol=1e-6))
            nodes, _ = create_nodes(50, 0.0, 2.0 * pi)
            D = derivative_matrix(nodes)
            @test all(isapprox.(df(nodes), D * f(nodes), atol=1e-14))
            nodes, _ = create_nodes(250, 0.0, 2.0 * pi)
            D = derivative_matrix(nodes)
            @test all(isapprox.(df(nodes), D * f(nodes), atol=1e-13))
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError derivative_matrix([1.0, 2.0])
        end

        @testset "Edge Cases" begin
            nodes, _ = create_nodes(3)
            D = derivative_matrix(nodes)
            @test size(D) == (3, 3)

            # Compute the reference derivative matrix
            D_ref = derivative_matrix(nodes)
            @test isapprox(D, D_ref, atol=1e-8)
        end

        @testset "Complex Cases" begin
            nodes, _ = create_nodes(6, -2.0, 2.0)
            D = derivative_matrix(nodes)
            @test size(D) == (6, 6)

            # Compute the reference derivative matrix
            D_ref = derivative_matrix(nodes)
            @test isapprox(D, D_ref, atol=1e-8)
        end
    end

    # Tests for kth_derivative_matrix
    @testset "kth_derivative_matrix tests" begin
        @testset "Valid Cases" begin

            f(x) = sin.(x)
            df(x) = cos.(x)
            d2f(x) = -sin.(x)
            d3f(x) = -cos.(x)
            nodes, _ = create_nodes(25, 0.0, 2.0 * pi)
           Dks = kth_derivative_matrix(nodes, 3)
            @test length(Dks) == 3

            @test all(isapprox.(Dks[1] * f(nodes), df(nodes), atol=1e-6))
            @test all(isapprox.(Dks[2] * f(nodes), d2f(nodes), atol=1e-6))
            @test all(isapprox.(Dks[3] * f(nodes), d3f(nodes), atol=1e-6))

        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError kth_derivative_matrix([1.0, 2.0], 3)
            @test_throws ArgumentError kth_derivative_matrix([1.0, 2.0, 3.0], -1)
        end
    end

    @testset "derivative_matrix_nd tests" begin
        @testset "Valid Cases" begin
            nodes1, _ = create_nodes(5)
            nodes2, _ = create_nodes(4, -1.0, 1.0)
            nodes_nd = [nodes1, nodes2]
            D_nd = derivative_matrix_nd(nodes_nd, 2, 1)
            @test length(D_nd) == 2
            @test size(D_nd[1]) == (5, 5)
            @test size(D_nd[2]) == (4, 4)

            # Compute the reference derivative matrices
            D_ref1 = enforce_ck_continuity(derivative_matrix(nodes1), 1)
            D_ref2 = enforce_ck_continuity(derivative_matrix(nodes2), 1)
            @test isapprox(D_nd[1], D_ref1, atol=1e-8)
            @test isapprox(D_nd[2], D_ref2, atol=1e-8)
        end

        @testset "Invalid Cases" begin
            @test_throws MethodError derivative_matrix_nd([1.0, 2.0], 2, 1)
            @test_throws MethodError derivative_matrix_nd([], 2, 1)
            @test_throws ArgumentError derivative_matrix_nd([create_nodes(5)[1]], 0, 1)
        end

        @testset "Edge Cases" begin
            nodes1, _ = create_nodes(3)
            nodes2, _ = create_nodes(3, -1.0, 1.0)
            nodes_nd = [nodes1, nodes2]
            D_nd = derivative_matrix_nd(nodes_nd, 2, 1)
            @test length(D_nd) == 2
            @test size(D_nd[1]) == (3, 3)
            @test size(D_nd[2]) == (3, 3)

            # Compute the reference derivative matrices
            D_ref1 = enforce_ck_continuity(derivative_matrix(nodes1), 1)
            D_ref2 = enforce_ck_continuity(derivative_matrix(nodes2), 1)
            @test isapprox(D_nd[1], D_ref1, atol=1e-8)
            @test isapprox(D_nd[2], D_ref2, atol=1e-8)
        end

        @testset "Complex Cases" begin
            nodes1, _ = create_nodes(6, -2.0, 2.0)
            nodes2, _ = create_nodes(5, -1.0, 1.0)
            nodes_nd = [nodes1, nodes2]
            D_nd = derivative_matrix_nd(nodes_nd, 2, 2)
            @test length(D_nd) == 2
            @test size(D_nd[1]) == (6, 6)
            @test size(D_nd[2]) == (5, 5)

            # Compute the reference derivative matrices
            D_ref1 = enforce_ck_continuity(derivative_matrix(nodes1), 2)
            D_ref2 = enforce_ck_continuity(derivative_matrix(nodes2), 2)
            @test isapprox(D_nd[1], D_ref1, atol=1e-8)
            @test isapprox(D_nd[2], D_ref2, atol=1e-8)
        end
    end
end
