using Test, LinearAlgebra, SparseArrays, Logging, BenchmarkTools, IterativeSolvers
using IncompleteLU, AlgebraicMultigrid
using KitchenSink.Preconditioners

@testset "Preconditioners" begin
	@testset "Jacobi Preconditioner" begin
		@testset "Valid inputs" begin
			# Test with a small dense matrix
			A_small = [4.0 1.0; 1.0 3.0]
			P_small = jacobi_preconditioner(A_small)
			@test P_small ≈ Diagonal([1 / 4, 1 / 3])

			# Test with a larger sparse matrix
			n = 100
			A_large = spdiagm(-1 => ones(n - 1), 0 => 4 * ones(n), 1 => ones(n - 1))
			P_large = jacobi_preconditioner(A_large)
			@test P_large ≈ Diagonal(fill(1 / 4, n))

			# Test with a matrix of integers
			A_int = [4 1; 1 3]
			P_int = jacobi_preconditioner(A_int)
			@test eltype(P_int) == Float64
			@test P_int ≈ Diagonal([1 / 4, 1 / 3])
		end

		@testset "Error conditions" begin
			# Test with non-symmetric matrix
			A_nonsym = [1.0 2.0; 3.0 4.0]
			@test_throws ArgumentError jacobi_preconditioner(A_nonsym)

			# Test with non-square matrix
			A_nonsquare = [1.0 2.0 3.0; 4.0 5.0 6.0]
			@test_throws ArgumentError jacobi_preconditioner(A_nonsquare)
		end
	end

	@testset "ILU Preconditioner" begin
		@testset "Valid inputs" begin
			# Test with a small sparse matrix
			A_small = sparse([4.0 1.0; 1.0 3.0])
			P_small = ilu_preconditioner(A_small)
			@test typeof(P_small) <: IncompleteLU.ILUFactorization

			# Test with a larger sparse matrix
			n = 100
			A_large = spdiagm(-1 => ones(n - 1), 0 => 4 * ones(n), 1 => ones(n - 1))
			P_large = ilu_preconditioner(A_large)
			@test typeof(P_large) <: IncompleteLU.ILUFactorization

			# Verify that the preconditioner can be applied
			b = ones(n)
			@test length(P_large \ b) == n
		end

		@testset "Error conditions" begin
			# Test with non-square matrix
			A_nonsquare = sparse([1.0 2.0 3.0; 4.0 5.0 6.0])
			@test_throws ArgumentError ilu_preconditioner(A_nonsquare)

			# Test with dense matrix (should throw MethodError)
			A_dense = [4.0 1.0; 1.0 3.0]
			@test_throws MethodError ilu_preconditioner(A_dense)
		end
	end

	    @testset "AMG Preconditioner" begin
        @testset "Valid inputs" begin
            # Test with a small sparse symmetric matrix
            A_small = sparse(Symmetric([4.0 1.0; 1.0 3.0]))
            P_small = amg_preconditioner(Matrix(A_small))
            @test typeof(P_small) <: AlgebraicMultigrid.MultilevelSolver

            # Test with a larger sparse symmetric matrix
            n = 100
            A_large = spdiagm(-1 => ones(n - 1), 0 => 4 * ones(n), 1 => ones(n - 1))
            A_large_sym = Symmetric(A_large)
            P_large = amg_preconditioner(Matrix(A_large_sym))
            @test typeof(P_large) <: AlgebraicMultigrid.MultilevelSolver

            # Verify that the preconditioner can be applied
            b = ones(n)
            @test length(P_large \ b) == n
        end

        @testset "Error conditions" begin
            # Test with non-symmetric matrix
            A_nonsym = sparse([1.0 2.0; 3.0 4.0])
            @test_throws ArgumentError amg_preconditioner(A_nonsym)

            # Test with non-square matrix
            A_nonsquare = sparse([1.0 2.0 3.0; 4.0 5.0 6.0])
            @test_throws ArgumentError amg_preconditioner(A_nonsquare)

            # Test with dense matrix (should throw MethodError)
            A_dense = Symmetric([4.0 1.0; 1.0 3.0])
            @test_throws MethodError amg_preconditioner(sparse(A_dense))
        end
    end

    @testset "Preconditioner Performance" begin
        n = 1000
        A = spdiagm(-1 => ones(n - 1), 0 => 4 * ones(n), 1 => ones(n - 1))
        A_sym = Symmetric(A)
        b = ones(n)
        A_sparse = sparse(A_sym)
        # Create preconditioners
        P_jacobi = jacobi_preconditioner(A)
        P_ilu = ilu_preconditioner(A)
        P_amg = amg_preconditioner(Matrix(A_sym))

        # Solve the system with each preconditioner
        x_jacobi = P_jacobi \ b
        x_ilu = P_ilu \ b
        x_amg = P_amg \ b

        # Check that solutions are reasonably close
        @test norm(A * x_jacobi - b) / norm(b) < 1e-6
        @test norm(A * x_ilu - b) / norm(b) < 1e-6
        @test norm(A * x_amg - b) / norm(b) < 1e-6
    end
end
