@testset "Direct Solvers" begin
    @testset "solve_direct" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        x_expected = A \ b

        @testset "LU decomposition" begin
            x = solve_direct(A, b, :lu)
            @test x ≈ x_expected
        end

        @testset "QR decomposition" begin
            x = solve_direct(A, b, :qr)
            @test x ≈ x_expected
        end

        @testset "Cholesky decomposition" begin
            A_spd = A'A
            b_spd = A'b
            x = solve_direct(A_spd, b_spd, :cholesky)
            @test x ≈ A_spd \ b_spd
        end

        @testset "Integer input" begin
            A_int = [4 1; 1 3]
            b_int = [1, 2]
            x = solve_direct(A_int, b_int, :lu)

            @test x ≈ x_expected
        end

        @testset "Large system" begin
            n = 1000
            A_large = sprand(n, n, 0.01) + 10I
            b_large = rand(n)
            x = solve_direct(A_large, b_large, :lu)
            @test A_large * x ≈ b_large
        end

        @testset "Error handling" begin
            @test_throws ArgumentError solve_direct(A, b, :invalid_method)
            @test_throws ArgumentError solve_direct([1.0 2.0 3.0; 4.0 5.0 6.0], [1.0, 2.0], :lu)
            @test_throws ArgumentError solve_direct(A, [1.0, 2.0, 3.0], :lu)
        end
    end

    @testset "solve_direct_multiple_rhs" begin
        A = [4.0 1.0; 1.0 3.0]
        B = [1.0 2.0; 2.0 3.0]
        X_expected = A \ B

        @testset "LU decomposition" begin
            X = solve_direct_multiple_rhs(A, B, :lu)
            @test X ≈ X_expected
        end

        @testset "QR decomposition" begin
            X = solve_direct_multiple_rhs(A, B, :qr)
            @test X ≈ X_expected
        end

        @testset "Cholesky decomposition" begin
            A_spd = A'A
            B_spd = A'B
            X = solve_direct_multiple_rhs(A_spd, B_spd, :cholesky)
            @test X ≈ A_spd \ B_spd
        end

        @testset "Large system" begin
            n = 1000
            m = 5
            A_large = sprand(n, n, 0.01) + 10I
            B_large = rand(n, m)
            X = solve_direct_multiple_rhs(A_large, B_large, :lu)
            @test A_large * X ≈ B_large
        end

        @testset "Error handling" begin
            @test_throws ArgumentError solve_direct_multiple_rhs(A, B, :invalid_method)
            @test_throws ArgumentError solve_direct_multiple_rhs([1.0 2.0 3.0; 4.0 5.0 6.0], B, :lu)
            @test_throws ArgumentError solve_direct_multiple_rhs(A, [1.0 2.0; 2.0 3.0; 3.0 4.0], :lu)
        end
    end
end
