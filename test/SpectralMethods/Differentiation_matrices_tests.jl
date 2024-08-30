@testset "Differentiation Matrices" begin
    f(x) = sin.(x)
    df(x) = cos.(x)
    d2f(x) = -sin.(x)
    d3f(x) = -cos.(x)

    @testset "derivative_matrix tests" begin
        @testset "Valid Cases" begin
            for n in [15, 25, 50]
                nodes, _ = create_legendre_nodes_and_weights(n, (0.0, 2pi))
                D = derivative_matrix!(nodes)
                x_vals = [node[1] for node in nodes]
                f_vals = f(x_vals)
                df_vals = df(x_vals)
                @test size(D) == (n, n)
                @test all(isapprox.(df_vals, D * f_vals, atol = 1e-6))
            end
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError derivative_matrix!([[1.0], [2.0]])
            @test_throws ArgumentError derivative_matrix!(AbstractVector{Float64}[])
        end

        @testset "Edge Cases" begin
            nodes = [[x] for x in [0.0, 0.5, 1.0]]
            D = derivative_matrix!(nodes)
            @test size(D) == (3, 3)
            @test isapprox(D[2, :], [-1.0, 0.0, 1.0], atol = 1e-8)
        end
    end

    @testset "kth_derivative_matrix tests" begin
        @testset "Valid Cases" begin
            for n in [15, 25, 50]
                nodes, _ = create_legendre_nodes_and_weights(n, (0.0, 2pi))
                Dks = kth_derivative_matrix!(nodes, 3)
                x_vals = [node[1] for node in nodes]
                f_vals = f(x_vals)
                D1 = Dks[1]
                D2 = Dks[2]
                D3 = Dks[3]
                Df1 = D1 * f_vals
                Df2 = D2 * f_vals
                Df3 = D3 * f_vals
                df_vals = df(x_vals)
                d2f_vals = d2f(x_vals)
                d3f_vals = d3f(x_vals)
                @test all(isapprox.(df_vals, Df1, atol = 1e-6))
                @test all(isapprox.(d2f_vals, Df2, atol = 1e-3))
                @test all(isapprox.(d3f_vals, Df3, atol = 1e-3))
            end
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError kth_derivative_matrix!([[1.0], [2.0]], 1)
            @test_throws ArgumentError kth_derivative_matrix!([[1.0], [2.0], [3.0]], 0)
            @test_throws ArgumentError kth_derivative_matrix!([[1.0], [2.0], [3.0]], -1)
        end
    end

    @testset "derivative_matrix_nd tests" begin
        @testset "Valid Cases" begin
            p = 5
            # 1D case
            points_1d, _ = create_legendre_nodes_and_weights(p, (-1.0, 1.0))
            D_1d = derivative_matrix_nd(points_1d, 1, 1)
            @test length(D_1d) == 1
            @test size(D_1d[1]) == (p, p)

            # 2D case
            points_2d, _ = gauss_legendre_with_boundary_nd(p, 2, [-1.0, -1.0], [1.0, 1.0])
            D_2d = derivative_matrix_nd(points_2d, 2, 1)
            @test length(D_2d) == 2
            @test all(size(D) == (p^2, p^2) for D in D_2d)

            # 3D case
            points_3d, _ = gauss_legendre_with_boundary_nd(p, 3, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
            D_3d = derivative_matrix_nd(points_3d, 3, 1)
            @test length(D_3d) == 3
            @test all(size(D) == (p^3, p^3) for D in D_3d)
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError derivative_matrix_nd([[0.0], [1.0]], 2, 1)
        end

        @testset "Edge Cases" begin
            points_min = [[x] for x in [-1.0, 0.0, 1.0]]
            D_min = derivative_matrix_nd(points_min, 1, 1)
            @test length(D_min) == 1
            @test size(D_min[1]) == (3, 3)

            points_high_degree = [[x] for x in range(-1, 1, length = 10)]
            D_high = derivative_matrix_nd(points_high_degree, 1, 5)
            @test length(D_high) == 1
            @test size(D_high[1]) == (10, 10)

            points_close = [[x] for x in [-1e-6, 0.0, 1e-6]]
            D_close = derivative_matrix_nd(points_close, 1, 1)
            @test isapprox(D_close[1][2, :], SparseVector([-5e5, 0.0, 5e5]), rtol = 1e-3)
        end
    end
end
