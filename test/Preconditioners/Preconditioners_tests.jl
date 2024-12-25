using Test, LinearAlgebra, SparseArrays, Statistics
using KitchenSink.Preconditioners
using KitchenSink.NumericUtilities
include("../test_utils.jl")

@testset "Preconditioners" begin
	@testset "Matrix Properties" begin
		@testset "SPD Matrices" begin
			n = 50
			A = create_spd_matrix(n; condition_number = 10.0)
			props = analyze_matrix_properties(A)
			@test props[:symmetric] == true
			@test props[:posdef] == true
			@test !props[:ill_conditioned]

			# Test ill-conditioned SPD
			A_ill = create_spd_matrix(n; condition_number = 1e12)
			props_ill = analyze_matrix_properties(A_ill)
			@test props_ill[:ill_conditioned] == true
		end

		@testset "Diagonal Dominance" begin
			n = 50
			A = create_spd_matrix(n; condition_number = 10.0)
			props = analyze_matrix_properties(A)
			@test props[:symmetric] == true
			@test props[:posdef] == true
			@test !props[:ill_conditioned]

			# Test ill-conditioned SPD
			A_ill = create_spd_matrix(n; condition_number = 1e12)
			props_ill = analyze_matrix_properties(A_ill)
			@test props_ill[:ill_conditioned] == true
		end

		@testset "Diagonal Dominance" begin
			n = 50
			A_strong = make_diag_dominant_matrix(n, 5.0)
			props_strong = analyze_matrix_properties(A_strong)
			@test props_strong[:diag_dominant] == true  # Fix property name

			A_weak = make_non_diag_dominant_matrix(n)
			props_weak = analyze_matrix_properties(A_weak)
			@test !props_weak[:diag_dominant]  # Fix property name
		end

		@testset "Sparsity" begin
			n = 100
			A_sparse = create_sparse_matrix(n; sparsity = 0.01)
			props_sparse = analyze_matrix_properties(A_sparse)
			@test props_sparse[:sparse] == true
			@test props_sparse[:sparsity] < 0.05

			A_dense = Matrix(create_sparse_matrix(n; sparsity = 0.8))
			props_dense = analyze_matrix_properties(A_dense)
			@test !props_dense[:sparse]
		end
	end

	@testset "Direct Methods" begin
		@testset "LU Preconditioner" begin
			for n in [10, 50]
				@testset "Size $n" begin
					A = create_test_matrix(n)
					P = lu_preconditioner(A)
					x = randn(n)
					b = A * x
					y = P(b)

					@test length(y) == n
					@test all(isfinite, y)
					@test norm(A * y - b) / norm(b) < 1e-6

					# Error cases
					@test_throws ArgumentError lu_preconditioner(zeros(n, n))  # Singular
					@test_throws ArgumentError lu_preconditioner(randn(n, n + 1))  # Non-square
				end
			end
		end
		@testset "Cholesky Preconditioner" begin
			for n in [10, 50]
				@testset "Size $n" begin
					# Create a symmetric positive definite matrix using helper function
					A = Symmetric(create_spd_matrix(n; condition_number = 10.0))  # Ensure matrix is explicitly symmetric

					# Basic functionality test
					P = cholesky_preconditioner(A)
					x = randn(n)
					b = A * x
					y = P(b)

					@test length(y) == n
					@test all(isfinite, y)
					@test norm(A * y - b) / norm(b) < 1e-6
					@test dot(x, y) > 0  # Preserves positive definiteness

					# Error cases
					@test_throws ArgumentError cholesky_preconditioner(-I(n))  # Negative definite
					@test_throws ArgumentError cholesky_preconditioner(triu(randn(n, n)))  # Non-symmetric
					@test_throws ArgumentError cholesky_preconditioner(
						create_symmetric_indefinite_matrix(n)
					)  # Symmetric but indefinite
				end
			end
		end

		@testset "QR Preconditioner" begin
			for n in [10, 50]
				@testset "Size $n" begin
					# Test dense matrix
					A = randn(n, n)
					P = qr_preconditioner(A)
					x = randn(n)
					b = A * x
					y = P(b)
					@test norm(A * y - b) / norm(b) < 1e-6

					# Test sparse matrix without expecting pivoting
					A_sparse = sparse(A)
					P_sparse = qr_preconditioner(A_sparse)
					y_sparse = P_sparse(b)
					@test norm(A_sparse * y_sparse - b) / norm(b) < 1e-6
				end
			end
		end
	end

	@testset "Iterative Methods" begin
		@testset "Jacobi Preconditioner" begin
			for n in [10, 50, 100]
				@testset "Size $n" begin
					# Diagonally dominant case
					A_dom = make_diag_dominant_matrix(n, 2.0)
					P_dom = jacobi_preconditioner(A_dom)
					x = randn(n)
					b = A_dom * x
					y = P_dom(b)

					@test length(y) == n
					@test all(isfinite, y)
					@test norm(A_dom * y - b) / norm(b) < 1e-3

					# Non-diagonally dominant case
					A_nondom = make_non_diag_dominant_matrix(n)
					P_nondom = jacobi_preconditioner(A_nondom)
					y_nondom = P_nondom(b)
					@test all(isfinite, y_nondom)  # Should still be stable
				end
			end
		end

		@testset "Symmetric Gauss-Seidel" begin
			for size in [10, 50]
				A = rand(size, size)
				A = A + A'
				b = rand(size)
				P = symmetric_gauss_seidel_preconditioner(A)
				converged, _ = test_preconditioner_convergence(A, b, P)
				@test converged
			end
		end
	end

	@testset "Advanced Methods" begin
		@testset "AMG Preconditioner" begin
			for n in [50, 100]
				@testset "Size $n" begin
					A = sparse(create_spd_matrix(n; condition_number = 10.0))
					P = amg_preconditioner(A)
					x = randn(n)
					b = A * x
					y = P(b)

					# Test dimensions and convergence
					@test length(y) == n
					@test all(isfinite, y)
					@test norm(A * y - b) / norm(b) < 0.1
				end
			end

			@testset "SPAI Tests" begin
				for n in [20, 50]
					# Test with explicitly sparse matrix
					A = sprand(n, n, 0.1) + sparse(I, n, n) * 2.0
					P = spai_preconditioner(A)

					# Test dimensions
					x = randn(n)
					y = P(x)
					@test length(y) == n
					@test all(isfinite, y)

					# Test approximate inverse property - use matrix multiplication directly
					@test norm(A * y - x) / norm(x) < 1.0  # Should reduce the residual
				end
			end
		end

		@testset "SPAI Preconditioner" begin
			for n in [20, 50]
				@testset "Size $n" begin
					# Test with explicitly sparse matrix with better conditioning
					A = sprand(n, n, 0.1)
					A = A + sparse(2.0I, n, n)  # Better diagonal dominance
					P = spai_preconditioner(A)

					# Basic functionality test
					x = randn(n)
					y = P(x)
					@test length(y) == n
					@test all(isfinite, y)

					# Test approximate inverse property directly with vectors
					@test norm(A * y - x) / norm(x) < 1.0  # Should reduce the residual
				end
			end
		end

		@testset "Stability Tests" begin
			n = 30
			# Test with moderately ill-conditioned matrix
			A_ill = make_ill_conditioned_sparse(n)
			A_ill = A_ill + sparse(sqrt(eps()) * I, n, n)  # Add regularization
			P = spai_preconditioner(A_ill)
			x = randn(n)
			b = A_ill * x
			y = P(b)

			# Should handle ill-conditioning gracefully
			@test all(isfinite, y)
			@test norm(y) < 1e8 * norm(b)

			# Test with nearly singular matrix
			A_sing = sparse(create_nearly_singular_matrix(n))
			A_sing = A_sing + sparse(sqrt(eps()) * I, n, n)  # Add regularization
			P_sing = spai_preconditioner(A_sing)
			b_sing = A_sing * x
			y_sing = P_sing(b_sing)
			@test all(isfinite, y_sing)
		end
	end

	@testset "Selection Strategy" begin
		@testset "SPD Cases" begin
			n = 50
			# Test well-conditioned SPD
			A_well = create_spd_matrix(n; condition_number = 10.0)
			method_well = choose_preconditioner(A_well)
			@test method_well in [:qr, :cholesky, :block_jacobi, :lu]

			# Test ill-conditioned SPD
			A_ill = create_spd_matrix(n; condition_number = 1e12)
			method_ill = choose_preconditioner(A_ill)
			@test method_ill in [:qr, :modified_sgs, :gmres]
		end

		@testset "Sparse Cases" begin
			n = 100  # Large enough to trigger sparse handling
			# Very sparse symmetric case
			A_sparse_symm = create_sparse_matrix(n, 0.01)
			A_sparse_symm = (A_sparse_symm + A_sparse_symm')/2
			method_sparse = choose_preconditioner(A_sparse_symm)
			@test method_sparse in [:qr, :amg_symm, :parallel_jacobi, :jacobi, :spai]

			# Sparse non-symmetric case
			A_sparse = create_sparse_matrix(n, 0.05)
			method_sparse_nonsym = choose_preconditioner(A_sparse)
			@test method_sparse_nonsym in [:qr, :spai, :parallel_jacobi, :jacobi, :gmres]
		end

		@testset "Special Cases" begin
			n = 50
			cases = [
				("Nearly Singular",
					create_spd_matrix(n; condition_number = 1e15),  # Extremely ill-conditioned
					[:qr]),
				("Diagonally Dominant",
					make_diag_dominant_matrix(n, 5.0),
					[:block_jacobi, :qr, :lu]),
				("Small Dense Symmetric",
					Symmetric(randn(20, 20)),
					[:qr, :symmetric_gauss_seidel, :modified_sgs, :lu]),
				("Large Sparse Symmetric",
					sparse(Symmetric(create_sparse_matrix(200; sparsity = 0.01))),
					[:qr, :amg_symm, :parallel_jacobi, :spai, :symmetric_gauss_seidel, :lu]),
			]

			for (name, A, expected) in cases
				@testset "$name" begin
					method = choose_preconditioner(A)
					@test method in expected

					# Test that selected preconditioner can be created
					P = select_preconditioner(A)
					x = randn(size(A, 1))
					y = P(x)
					@test length(y) == length(x)
					@test all(isfinite, y)
				end
			end
		end

		@testset "Fallback Behavior" begin
			n = 50
			# Test fallback for deep recursion
			A = randn(n, n)
			method = choose_preconditioner(A; max_depth = 0)
			@test method == :jacobi

			# Test fallback for ill-conditioned matrix
			A_ill = create_spd_matrix(n; condition_number = 1e15)
			method_ill = choose_preconditioner(A_ill)
			@test method_ill in [:qr, :gmres]

			# Test thread-aware selection
			if Threads.nthreads() > 1
				A_large = randn(200, 200)
				method_parallel = choose_preconditioner(A_large)
				@test method_parallel in [:parallel_jacobi, :block_jacobi, :lu, :qr, :diagonal]
			end
		end
	end

	@testset "Fallback Behavior" begin
		n = 50
		@testset "Forced Fallbacks" begin
			A = create_spd_matrix(n; condition_number = 10.0)

			# Force Cholesky to fail
			A_mod = A + 1e-14 * I
			method = choose_preconditioner(A_mod)
			@test method != :cholesky

			# Ensure fallback works
			P = select_preconditioner(A_mod)
			x = randn(n)
			b = A_mod * x
			y = P(b)
			@test all(isfinite, y)
			@test norm(A_mod * y - b) / norm(b) < 1e-2
		end

		@testset "Method Override" begin
			A = create_spd_matrix(n; condition_number = 10.0)

			# Try different method than default
			P_alt = select_preconditioner(A; method = :qr)
			x = randn(n)
			b = A * x
			y = P_alt(b)
			@test all(isfinite, y)
			@test norm(A * y - b) / norm(b) < 1e-2
		end
	end

	@testset "Cache Management" begin
		n = 30  # Small size for cache tests
		@testset "Basic Caching" begin
			A = create_spd_matrix(n; condition_number = 10.0)

			# Initial creation
			P1 = cached_get_preconditioner(A)
			# Second request should hit cache
			P2 = cached_get_preconditioner(A)

			# Test functional equivalence instead of identity
			x = randn(n)
			@test norm(P1(x) - P2(x)) ≤ sqrt(eps()) * norm(x)

			# Modified matrix should get new preconditioner
			A_mod = A + 0.1 * I
			P3 = cached_get_preconditioner(A_mod)
			@test norm(P1(x) - P3(x)) > sqrt(eps()) * norm(x)
		end

		@testset "Cache Invalidation" begin
			A = create_spd_matrix(n; condition_number = 10.0)
			P1 = cached_get_preconditioner(A)
			x = randn(n)

			# Significant change should invalidate cache
			A_new = A + I
			P2 = cached_get_preconditioner(A_new)
			@test norm(P1(x) - P2(x)) > sqrt(eps()) * norm(x)

			# Small perturbation should reuse cache
			A_small = A + 1e-14 * I
			P3 = cached_get_preconditioner(A_small)
			@test norm(P1(x) - P3(x)) ≤ sqrt(eps()) * norm(x)
		end

		@testset "Thread Safety" begin
			n = 50
			A = create_spd_matrix(n)
			cache_dict = IdDict()

			@testset "Concurrent Cache Access" begin
				# Initialize results vector before the threaded loop
				results = Vector{Any}(undef, 10)

				# Test concurrent access
				Threads.@threads for i in 1:10
					results[i] = cached_get_preconditioner(A; cache_dict = cache_dict)
				end

				# All preconditioners should be functionally equivalent
				x = randn(n)
				y_ref = results[1](x)
				for i in 2:10
					@test norm(results[i](x) - y_ref) ≤ sqrt(eps()) * norm(x)
				end
			end

			@testset "Concurrent Updates" begin
				P = cached_get_preconditioner(A)
				updates = Vector{Any}(undef, 5)
				results = Vector{Vector{Float64}}(undef, 5)

				# Create test vector before the threaded loop
				x = randn(n)

				# Test concurrent updates with different perturbations
				Threads.@threads for i in 1:5
					new_A = A + (0.1 * i) * I
					updates[i] = update_preconditioner!(
						P, A, new_A; cache_dict = cache_dict
					)
					results[i] = updates[i](x)
				end

				# Results should be different for different perturbations
				for i in 2:5
					@test norm(results[i] - results[1]) > sqrt(eps()) * norm(x)
				end
			end
		end
	end

	@testset "Thread Safety" begin
		n = 50
		A = create_spd_matrix(n)
		cache_dict = IdDict()

		@testset "Concurrent Cache Access" begin
			results = Vector{Any}(undef, 10)

			# Test concurrent access
			Threads.@threads for i in 1:10
				results[i] = cached_get_preconditioner(A; cache_dict = cache_dict)
			end

			# All preconditioners should be identical
			for i in 2:10
				@test results[i] === results[1]
			end
		end

		@testset "Concurrent Updates" begin
			P = cached_get_preconditioner(A)
			updates = Vector{Any}(undef, 5)

			# Test concurrent updates
			Threads.@threads for i in 1:5
				new_A = A + (0.1 * i) * I
				updates[i] = update_preconditioner!(P, A, new_A; cache_dict = cache_dict)
			end

			# All updates should complete without errors
			@test all(x -> x isa Function, updates)
		end
	end

	@testset "Stack Safety" begin
		n = 20

		@testset "Deep Recursion" begin
			A = rand(100, 100)
			method = choose_preconditioner(A; max_depth = 0)
			@test method == :jacobi  # Should fall back to Jacobi
		end

		@testset "Recursive Updates" begin
			A = rand(100, 100)
			method = choose_preconditioner(A; max_depth = 0)
			@test method == :jacobi
		end

		@testset "Strategy Selection" begin
			# Create challenging matrix
			A = create_spd_matrix(n; condition_number = 1e12)

			# Should handle deep recursion gracefully
			method = choose_preconditioner(A; max_depth = 5)
			@test method isa Symbol

			P = select_preconditioner(A)
			@test P isa Function
		end
	end

	@testset "Convergence Tests" begin
		for size in [10, 50]
			A = rand(size, size)
			A = A + A'
			b = rand(size)
			P = symmetric_gauss_seidel_preconditioner(A)
			converged, _ = test_preconditioner_convergence(A, b, P)
			@test converged
		end
	end

	@testset "Convergence Tests" begin
		for size in [10, 50]
			A = rand(size, size)
			A = A + A'
			b = rand(size)
			P = symmetric_gauss_seidel_preconditioner(A)
			converged, _ = test_preconditioner_convergence(A, b, P)
			@test converged
		end
	end
end
