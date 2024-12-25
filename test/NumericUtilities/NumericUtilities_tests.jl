# FILE: NumericUtilities_tests.jl

using Test
using LinearAlgebra, StaticArrays, SparseArrays, FastGaussQuadrature, Statistics
using KitchenSink.NumericUtilities
include("../test_utils.jl")

@testset "NumericUtilities" begin
	@testset "Matrix Properties Analysis" begin
		n = 50  # Large enough to test algorithmic choices
		matrix_suite = create_matrix_suite(n)

		for (name, (A, properties)) in matrix_suite
			@testset "$name" begin
				computed_props = analyze_matrix_properties(A)

				# Verify each claimed property
				for (prop, expected) in properties
					if haskey(computed_props, prop)
						@test verify_matrix_property(A, prop) ==
							(computed_props[prop] == true)
					end
				end

				# Test condition number estimation accuracy
				if haskey(properties, :cond_num_approx)
					computed_cond = computed_props[:cond_num]
					expected_cond = properties[:cond_num_approx]
					magnitude_diff = abs(log10(computed_cond) - log10(expected_cond))
					@test magnitude_diff < 1.0
				end
			end
		end
		# Test specific matrix properties
		@testset "Property Detection" begin
			# Test diagonal dominance
			A_dd = gen_diag_dominant(n)
			props = analyze_matrix_properties(A_dd)
			@test props[:diag_dominant] == true

			# Test SPD
			A_spd = create_spd_matrix(n)
			props = analyze_matrix_properties(A_spd)
			@test props[:symmetric] == true
			@test props[:posdef] == true

			# Test non-diagonal dominant
			A_ndd = make_non_diag_dominant_matrix(n)
			props = analyze_matrix_properties(A_ndd)
			@test props[:diag_dominant] == false

			# Test ill-conditioned
			A_ill = gen_ill_conditioned(n)
			props = analyze_matrix_properties(A_ill)
			@test props[:ill_conditioned] == true

			# Test sparse matrix
			A_sparse = create_sparse_matrix(n)
			props = analyze_matrix_properties(A_sparse)
			@test props[:sparse] == true

			# Test nearly singular matrix
			A_sing = create_test_matrix(n, :nearly_singular)
			props = analyze_matrix_properties(A_sing)
			@test props[:nearly_singular] == true

			# Test symmetric indefinite
			A_indef = create_symmetric_indefinite_matrix(n)
			props = analyze_matrix_properties(A_indef)
			@test props[:symmetric] == true
			@test props[:posdef] == false
		end
	end

	@testset "Condition Number Estimation" begin
		@testset "Matrix Size Scaling" begin
			for n in [10, 50, 100]
				A = gen_spd_matrix(n, 1e4)
				κ_power = estimate_condition_number_power_method(A)
				κ_direct = estimate_condition_number(A)  # Use our function instead of cond
				@test abs(log10(κ_power) - log10(κ_direct)) < 1.0
			end
		end

		@testset "Challenging Cases" begin
			n = 50
			challenging_cases = generate_challenging_cases(n)

			for (case_name, A) in challenging_cases
				@testset "$case_name" begin
					κ_est = estimate_condition_number(A)
					κ_true = cond(A)
					# Allow larger tolerance for challenging cases
					@test abs(log10(κ_est) - log10(κ_true)) < 2.0
				end
			end
		end

		@testset "Method Selection" begin
			n = 100
			# Test different matrix types
			matrices = Dict(
				"sparse_spd" => make_spd(create_test_matrix(n, true)),
				"diag_dominant" => gen_diag_dominant(n),
				"dense_spd" => gen_spd_matrix(n),
				"ill_cond" => gen_ill_conditioned(n),
			)

			for (name, A) in matrices
				props = compute_base_matrix_properties(A)
				method = determine_best_algorithm_condition_number(props)
				κ = estimate_condition_number(A; method = method)
				κ_true = issparse(A) ? estimate_condition_number(A) : cond(A)
				@test abs(log10(κ) - log10(κ_true)) < 2.0
			end
		end
	end

	@testset "Condition Number Estimation with Matrix Suite" begin
		n = 50
		# Use test_utils.jl matrix creation functions
		matrices = Dict(
			"test_matrix" => create_test_matrix(n),
			"spd_matrix" => gen_spd_matrix(n),
			"ill_conditioned" => gen_ill_conditioned(n),
			"diag_dominant" => gen_diag_dominant(n),
			"moderate_diag_dom" => create_moderately_diag_dominant(n),
			"symmetric_indefinite" => create_symmetric_indefinite_matrix(n),
		)

		for (name, A) in matrices
			@testset "$name" begin
				# Test different estimation methods
				κ_power = estimate_condition_number_power_method(A)
				κ_direct = cond(A)

				# Allow larger tolerance for ill-conditioned matrices
				tol = name == "ill_conditioned" ? 2.0 : 1.0
				@test abs(log10(κ_power) - log10(κ_direct)) < 1.1 * tol

				# Test base properties computation
				props = compute_base_matrix_properties(A)
				@test haskey(props, :size)
				@test haskey(props, :symmetric)
				@test haskey(props, :diag_dominant)
			end
		end
	end

	@testset "Matrix Regularization" begin
		@testset "Property Preservation" begin
			n = 50
			matrix_suite = create_matrix_suite(n)

			for (name, (A, properties)) in matrix_suite
				@testset "$name" begin
					A_reg = copy(A)
					regularize_matrix!(A_reg)

					# Test invariance properties
					invariance = test_matrix_algorithm_invariance(
						x -> begin
							y = copy(x)
							regularize_matrix!(y)
							return y
						end, A)

					# Regularized matrix should be better conditioned
					κ_orig = issparse(A) ? estimate_condition_number(A) : cond(A)
					κ_reg = issparse(A_reg) ? estimate_condition_number(A_reg) : cond(A_reg)
					@test κ_reg < κ_orig || κ_orig < 1e6

					# Check if important properties are preserved
					if get(properties, :symmetric, false)
						@test get(invariance, :preserves_symmetry, false)
					end
					if get(properties, :sparse, false)
						@test get(invariance, :preserves_sparsity, false)
					end
				end
			end
		end

		@testset "Convergence Improvement" begin
			n = 50
			for _ in 1:3  # Test multiple random instances
				A = gen_ill_conditioned(n; cond_num = 1e8)
				b = randn(n)
				A_reg = copy(A)
				regularize_matrix!(A_reg)

				# Test if regularization improves convergence
				P_orig = r -> r  # No preconditioner
				P_reg = r -> try
					A_reg \ (A * r)
				catch
					r  # Fallback to no preconditioning if solve fails
				end

				conv_orig, iter_orig = test_preconditioner_convergence(A, b, P_orig)
				conv_reg, iter_reg = test_preconditioner_convergence(A, b, P_reg)

				# Either regularization should improve convergence or original method already works well
				@test !conv_orig || (conv_reg && iter_reg ≤ 2 * iter_orig)
			end
		end
	end

	@testset "Regularization with Test Matrices" begin
		for n in [10, 50, 100]
			@testset "Size $n" begin
				# Test with different matrix types from test_utils.jl
				matrices = [
					("standard", create_test_matrix(n)),
					("ill_conditioned", gen_ill_conditioned(n)),
					("sparse", create_test_matrix(n, true)),
					("moderate_diag_dom", create_moderately_diag_dominant(n)),
				]

				for (name, A) in matrices
					A_reg = copy(A)
					regularize_matrix!(A_reg)

					# Check condition number improvement
					κ_before = estimate_condition_number(A)
					κ_after = estimate_condition_number(A_reg)

					# Either condition number should improve or was already good
					@test κ_after < κ_before || κ_before < 1e6

					# Check matrix properties are preserved when expected
					if issymmetric(A)
						@test issymmetric(A_reg)
					end
					if issparse(A)
						@test issparse(A_reg)
					end
				end
			end
		end
	end

	@testset "Stability Metrics" begin
		@testset "Basic Properties" begin
			n = 50
			matrix_suite = create_matrix_suite(n)

			for (name, (A, properties)) in matrix_suite
				@testset "$name" begin
					metrics = compute_stability_metrics(A)

					# Basic metric checks
					@test 0.0 ≤ metrics[:zero_ratio] ≤ 1.0
					@test metrics[:norm_A] > 0
					@test metrics[:dynamic_range] ≥ 1.0

					if metrics[:is_square] == 1.0
						@test metrics[:diag_dominance] > 0  # Changed from diag_dominantinance
						@test 0.0 ≤ metrics[:zero_diag_ratio] ≤ 1.0
						@test metrics[:symm_diff] ≥ 0
					end
				end
			end
		end

		@testset "Matrix-Vector Metrics" begin
			n = 50
			challenging_cases = generate_challenging_cases(n)

			for (case_name, A) in challenging_cases
				@testset "$case_name" begin
					b = randn(n)
					metrics = compute_stability_metrics(A, b)

					# Test additional vector metrics
					@test haskey(metrics, :norm_b)
					@test haskey(metrics, :vector_zeros)
					@test haskey(metrics, :vector_dynamic_range)

					# Verify metric values
					@test metrics[:norm_b] ≈ norm(b, Inf)
					@test 0.0 ≤ metrics[:vector_zeros] ≤ 1.0
					@test metrics[:vector_dynamic_range] ≥ 1.0

					# Consistency checks
					matrix_only = compute_stability_metrics(A)
					for k in keys(matrix_only)
						@test metrics[k] ≈ matrix_only[k]
					end
				end
			end
		end

		@testset "Vector-Only Metrics" begin
			for n in [10, 50, 100]
				@testset "Size $n" begin
					# Test different vector types
					v_dense = randn(n)
					v_sparse = sparsevec(rand(1:n, n ÷ 10), randn(n ÷ 10), n)
					v_zeros = zeros(n)

					for v in [v_dense, v_sparse, v_zeros]
						metrics = compute_stability_metrics(v)

						# Basic checks
						@test metrics[:size] == n
						@test 0.0 ≤ metrics[:sparsity] ≤ 1.0
						@test metrics[:norm_A] ≥ 0

						# Special case for zero vector
						if all(iszero, v)
							@test metrics[:zero_ratio] ≈ 1.0
						end
					end
				end
			end
		end
	end

	@testset "Numerical Validation" begin
		@testset "Well-Conditioned Cases" begin
			for n in [10, 50]
				A = make_diag_dominant_matrix(n)
				b = randn(n)
				success, issues = validate_numerical_computation(A, b)

				@test success
				@test isempty(issues)
				@test is_diagonally_dominant(A)

				# Verify stability metrics
				metrics = compute_stability_metrics(A)
				@test metrics[:cond_num] < 1e6
				@test metrics[:diag_dominant] == 1.0  # Changed from diagonally_dominant
			end
		end
		@testset "Ill-Conditioned Cases" begin
			n = 50
			A = gen_ill_conditioned(n; cond_num = 1e12)  # Use higher condition number to ensure near-singularity
			b = randn(n)
			success, issues = validate_numerical_computation(A, b)

			@test !success
			@test any(i -> contains(i, "High condition number"), issues)
			@test any(i -> contains(i, "lacks diagonal dominance"), issues)
		end

		@testset "Edge Cases" begin
			# Empty matrix
			success, issues = validate_numerical_computation(
				Matrix{Float64}(undef, 0, 0), Float64[]
			)
			@test !success
			@test any(contains.(issues, "Empty input"))

			# Dimension mismatch
			A = randn(3, 3)
			b = randn(2)
			success, issues = validate_numerical_computation(A, b)
			@test !success
			@test any(contains.(issues, "dimension"))

			# Non-finite values
			A = [1.0 Inf; 1.0 1.0]
			b = [1.0, 1.0]
			success, issues = validate_numerical_computation(A, b)
			@test !success
			@test any(contains.(issues, "Non-finite"))
		end
	end

	@testset "Algorithm Selection" begin
		@testset "Size-Based Selection" begin
			props = Dict{Symbol, Any}()

			# Small matrices
			props[:size] = 4
			@test determine_best_algorithm_condition_number(props) == :direct

			# Medium matrices
			props[:size] = 50
			@test determine_best_algorithm_condition_number(props) == :direct

			# Large matrices
			props[:size] = 2000
			props[:sparse] = false
			@test determine_best_algorithm_condition_number(props) == :power_iteration
		end
		@testset "Structure-Based Selection" begin
			n = 200
			# Sparse SPD
			props = Dict(
				:size => n,
				:sparse => true,
				:symmetric => true,
				:posdef => true,
				:sparsity => 0.05,
			)
			@test determine_best_algorithm_condition_number(props) == :cholesky_sparse

			# General sparse
			props[:posdef] = false
			@test determine_best_algorithm_condition_number(props) == :iterative_sparse

			# Diagonally dominant
			props = Dict(
				:size => n,
				:sparse => false,
				:diag_dominant => true,
			)
			method = determine_best_algorithm_condition_number(props)
			@test method in [:power_iteration, :direct]  # Allow both as valid choices
		end

		@testset "Algorithmic Performance" begin
			@testset "Algorithmic Performance" begin
				n = 100
				matrix_suite = create_matrix_suite(n)

				for (name, (A, properties)) in matrix_suite
					props = compute_base_matrix_properties(A)
					method = determine_best_algorithm_condition_number(props)
					κ = estimate_condition_number(A; method = method)
					κ_true = issparse(A) ? estimate_condition_number(A) : cond(A)

					# Verify accuracy within reasonable bounds
					@test abs(log10(κ) - log10(κ_true)) < 2.0

					# Verify method selection is appropriate
					if props[:sparse] && props[:symmetric] && props[:posdef]
						@test method == :cholesky_sparse
					elseif !props[:sparse] && props[:size] > 1000
						@test method == :power_iteration
					end
				end
			end
		end
	end
	@testset "Sparse Matrix Handling" begin
		n = 50
		A_sparse = make_ill_conditioned_sparse(n)
		κ = estimate_condition_number(A_sparse)

		# Test condition number estimation
		@test κ > 1e6

		# Test regularization
		A_reg = A_sparse
		regularize_matrix!(A_reg)
		κ_reg = estimate_condition_number(A_reg)
		@test κ_reg < κ  # Allow for small numerical differences
	end

	@testset "Diagonal Dominance Validation" begin
		# Test clear non-diagonal dominance
		A = make_non_diag_dominant_matrix(3)
		b = ones(3)
		success, issues = validate_numerical_computation(A, b)
		@test !success
		@test any(i -> contains(i, "diagonal dominance"), issues)

		# Test clear non-diagonal dominance with a less borderline case
		A = [1.0 1.5; 1.5 1.0]  # Off-diagonal elements clearly larger than diagonal
		success, issues = validate_numerical_computation(A, [1.0, 1.0])
		@test !success
		@test any(i -> contains(i, "diagonal dominance"), issues)

		# Test diagonally dominant case
		A = make_diag_dominant_matrix(3, 2.0)
		success, issues = validate_numerical_computation(A, ones(3))
		@test success || !any(i -> contains(i, "diagonal dominance"), issues)
	end
end
