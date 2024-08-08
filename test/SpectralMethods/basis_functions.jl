
function Lagrange_polynomials_full_def(nodes::AbstractVector{T})::Matrix{Float64} where {T}
    n = length(nodes)
    if n < 3
        throw(ArgumentError("The length of nodes must be at least 3."))
    end

    L = zeros(Float64, n, n)
    for i in 1:n
        p = ones(Float64, n)
        for j in 1:n
            if i != j
                p .*= (nodes .- nodes[j]) ./ (nodes[i] - nodes[j])
            end
        end
        L[:, i] = p
    end
    return L
end

@testset "Basis Functions" begin
    # Tests for barycentric_weights
    @testset "barycentric_weights tests" begin
        @testset "Valid Cases" begin
            nodes, _ = create_nodes(3)
            weights = barycentric_weights(nodes)
            @test length(weights) == 3
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError barycentric_weights([1.0])
            @test_throws ArgumentError barycentric_weights([1.0, 2.0])
        end
    end

    # Tests for Lagrange_polynomials
    @testset "Lagrange_polynomials tests" begin
        @testset "Valid Cases" begin
            n = 5
            nodes, _ = create_nodes(n)
            L_full_def = Lagrange_polynomials_full_def(nodes)
            L_full_def = sparse(L_full_def)
            @test sparse(I, n, n) == L_full_def
            n = 50
            nodes, _ = create_nodes(n)
            L_full_def = Lagrange_polynomials_full_def(nodes)
            L_full_def = sparse(L_full_def)
            @test sparse(I, n, n) == L_full_def
            n = 500
            nodes, _ = create_nodes(n)
            L_full_def = Lagrange_polynomials_full_def(nodes)
            L_full_def = sparse(L_full_def)
            @test sparse(I, n, n) == L_full_def
        end

        @testset "Invalid Cases" begin
            @test_throws ArgumentError Lagrange_polynomials([1.0])
            @test_throws ArgumentError Lagrange_polynomials([1.0, 2.0])
        end
    end

    # Tests for barycentric_interpolation
    @testset "barycentric_interpolation tests" begin
        @testset "Valid Cases" begin
            nodes = [-1.0, 0.0, 1.0]
            values = [1.0, 2.0, 3.0]
            x = 0.5
            interpolated_value = barycentric_interpolation(nodes, values, x)
            @test interpolated_value ≈ 2.5
        end

        @testset "Edge Cases" begin
            nodes = [-1.0, 0.0, 1.0]
            values = [1.0, 2.0, 3.0]
            @test barycentric_interpolation(nodes, values, 0.0) == 2.0
        end
    end

    # Tests for interpolate_nd
    @testset "interpolate_nd tests" begin
        @testset "Valid Cases" begin
            nodes = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
            valuesx = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
            point = [0.5, -0.5]
            interpolated_value = interpolate_nd(nodes, valuesx, point)
            @test interpolated_value ≈ 6
        end

        @testset "Edge Cases" begin
            nodes = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
            valuesx = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3)
            @test interpolate_nd(nodes, valuesx, [-1.0, -1.0]) == 1.0
            @test interpolate_nd(nodes, valuesx, [1.0, 1.0]) == 9.0
        end
    end
end
