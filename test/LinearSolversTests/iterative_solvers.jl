@testset "Iterative Solvers" begin
    @testset "solve_iterative" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        x_expected = A \ b

        @testset "Conjugate Gradient (CG) Method" begin
            solver = KSIterativeSolver(:cg, 100, 1e-6, nothing)
            x = solve_iterative(A, b, solver)
            @test x ≈ x_expected
        end

        @testset "Generalized Minimal Residual (GMRES) Method" begin
            solver = KSIterativeSolver(:gmres, 100, 1e-6, nothing)
            x = solve_iterative(A, b, solver)
            @test x ≈ x_expected
        end
``
        # @testset "BiConjugate Gradient Stabilized (BiCGSTAB) Method" begin
        #     solver = KSIterativeSolver(:bicgstab, 100, 1e-6, nothing)
        #     x = solve_iterative(A, b, solver)
        #     @test x ≈ x_expected
        # end

        # @testset "Preconditioned system" begin
        #     n = 100
        #     A_large = sprand(n, n, 0.01) + 10I
        #     b_large = rand(n)
        #     preconditioner = ilu_preconditioner(A_large)
        #     solver = KSIterativeSolver(:gmres, 1000, 1e-8, preconditioner)
        #     x = solve_iterative(A_large, b_large, solver)
        #     @test A_large * x ≈ b_large
        # end

        @testset "Error handling" begin
            @test_throws ArgumentError solve_iterative([1.0 2.0; 3.0 4.0; 5.0 6.0], [1.0, 2.0], KSIterativeSolver(:cg, 100, 1e-6, nothing))
            @test_throws ArgumentError solve_iterative(A, [1.0, 2.0, 3.0], KSIterativeSolver(:cg, 100, 1e-6, nothing))
            @test_throws ArgumentError solve_iterative(A, b, KSIterativeSolver(:invalid_method, 100, 1e-6, nothing))
        end
    end

    @testset "solve_iterative_multiple_rhs" begin
        A = [4.0 1.0; 1.0 3.0]
        B = [1.0 2.0; 2.0 3.0]
        X_expected = A \ B

        @testset "Conjugate Gradient (CG) Method" begin
            solver = KSIterativeSolver(:cg, 100, 1e-6, nothing)
            X = solve_iterative_multiple_rhs(A, B, solver)
            @test X ≈ X_expected
        end

        @testset "Generalized Minimal Residual (GMRES) Method" begin
            solver = KSIterativeSolver(:gmres, 100, 1e-6, nothing)
            X = solve_iterative_multiple_rhs(A, B, solver)
            @test X ≈ X_expected
        end

        # @testset "BiConjugate Gradient Stabilized (BiCGSTAB) Method" begin
        #     solver = KSIterativeSolver(:bicgstab, 100, 1e-6, nothing)
        #     X = solve_iterative_multiple_rhs(A, B, solver)
        #     @test X ≈ X_expected
        # end

        # @testset "Large sparse system" begin
        #     n = 1000
        #     m = 5
        #     A_large = sprand(n, n, 0.01) + 10I
        #     B_large = rand(n, m)
        #     solver = KSIterativeSolver(:cg, 1000, 1e-8, nothing)
        #     X = solve_iterative_multiple_rhs(A_large, B_large, solver)
        #     @test A_large * X ≈ B_large
        # end

        @testset "Error handling" begin
            @test_throws ArgumentError solve_iterative_multiple_rhs([1.0 2.0; 3.0 4.0; 5.0 6.0], B, KSIterativeSolver(:cg, 100, 1e-6, nothing))
            @test_throws ArgumentError solve_iterative_multiple_rhs(A, [1.0 2.0; 2.0 3.0; 3.0 4.0], KSIterativeSolver(:cg, 100, 1e-6, nothing))
        end
    end

    @testset "iterative_refinement" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        x_expected = A \ b

        @testset "Basic functionality" begin
            x0 = [0.0, 0.0]
            x = iterative_refinement(A, x0, b)
            @test x ≈ x_expected
        end

        @testset "Close initial guess" begin
            x0 = x_expected + 1e-6 * rand(2)
            x = iterative_refinement(A, x0, b)
            @test x ≈ x_expected
        end

        @testset "Maximum iterations" begin
            x0 = [0.0, 0.0]
            x = iterative_refinement(A, x0, b; max_iter=1, tol=1e-12)
            @test norm(b - A * x) < 1e-12
        end

        @testset "Large system" begin
            n = 1000
            A_large = sprand(n, n, 0.01) + 10I
            b_large = rand(n)
            x0 = zeros(n)
            x = iterative_refinement(A_large, x0, b_large; max_iter=10, tol=1e-8)
            @test norm(b_large - A_large * x) < 1e-8
        end

        @testset "Error handling" begin
            @test_throws ArgumentError iterative_refinement([1.0 2.0; 3.0 4.0; 5.0 6.0], [0.0, 0.0], [1.0, 2.0])
            @test_throws ArgumentError iterative_refinement(A, [0.0, 0.0, 0.0], b)
            @test_throws ArgumentError iterative_refinement(A, [0.0, 0.0], [1.0, 2.0, 3.0])
        end
    end
end
