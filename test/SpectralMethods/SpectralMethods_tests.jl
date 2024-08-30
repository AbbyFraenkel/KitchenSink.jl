using Test
using LinearAlgebra, SparseArrays
import FastGaussQuadrature: gausslegendre
# using ..KSTypes, ..SpectralMethods, ..CoordinateSystems
using KitchenSink.SpectralMethods, KitchenSink.KSTypes, KitchenSink.CoordinateSystems

# Function to generate expected values based on the test case
function expected_values(test_case::Symbol)
	if test_case == :case1
		nodes, weights = gausslegendre(3)
		nodes_inner = nodes
		weights_inner = weights
		nodes = vcat(-1.0, nodes, 1.0)
		weights = vcat(0.0, weights, 0.0)
		return nodes, weights, nodes_inner, weights_inner
	elseif test_case == :case2
		nodes, weights = gausslegendre(2)
		nodes_inner = nodes
		weights_inner = weights
		nodes = vcat(-1.0, nodes, 1.0)
		weights = vcat(0.0, weights, 0.0)
		return nodes, weights, nodes_inner, weights_inner
	elseif test_case == :case3
		nodes = [-1.0, 0.0, 1.0]
		weights = [0.0, 2.0, 0.0]
		return nodes, weights
	elseif test_case == :case4
		nodes, weights = gausslegendre(1)
		nodes_inner = nodes
		weights_inner = weights
		nodes = vcat(-1.0, nodes, 1.0)
		weights = vcat(0.0, weights, 0.0)
		return nodes, weights, nodes_inner, weights_inner
	else
		error("Unknown test case")
	end
end

@testset "Spectral Methods" begin
	@testset "create_legendre_nodes_and_weights Tests" begin
		@testset "Valid Cases" begin
			nodes_with_boundary, weights_with_boundary, nodes_interior, weights_interior = create_legendre_nodes_and_weights(5)
			expected_nodes, expected_weights = expected_values(:case1)
			@test all(isapprox(nodes_with_boundary[i], expected_nodes[i], atol = 1e-8) for i in eachindex(nodes_with_boundary))
			@test all(isapprox(weights_with_boundary[i], expected_weights[i], atol = 1e-8) for i in eachindex(weights_with_boundary))

			nodes_with_boundary, weights_with_boundary, nodes_interior, weights_interior = create_legendre_nodes_and_weights(4)
			expected_nodes, expected_weights = expected_values(:case2)
			@test all(isapprox(nodes_with_boundary[i], expected_nodes[i], atol = 1e-8) for i in eachindex(nodes_with_boundary))
			@test all(isapprox(weights_with_boundary[i], expected_weights[i], atol = 1e-8) for i in eachindex(weights_with_boundary))
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError create_legendre_nodes_and_weights(2)
		end

		@testset "Edge Cases" begin
			nodes_with_boundary, weights_with_boundary, nodes_interior, weights_interior = create_legendre_nodes_and_weights(3)
			expected_nodes, expected_weights = expected_values(:case3)
			@test all(isapprox(nodes_with_boundary[i], expected_nodes[i], atol = 1e-8) for i in eachindex(nodes_with_boundary))
			@test all(isapprox(weights_with_boundary[i], expected_weights[i], atol = 1e-8) for i in eachindex(weights_with_boundary))
		end
	end

	@testset "gauss_legendre_with_boundary_nd" begin
		@testset "Valid cases" begin
			@testset "1D case" begin
				p, dim = 5, 1
				nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)
				@test length(nodes) == p
				@test length(weights) == p
				@test all(n -> isa(n, Vector{Float64}), nodes)
				@test nodes[1][1] ≈ -1.0
				@test nodes[end][1]≈1.0 atol=1e-10
				@test sum(weights)≈2.0 atol=1e-10
			end

			@testset "2D case" begin
				p, dim = 4, 2
				nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)
				@test length(nodes) == p^2
				@test length(weights) == p^2
				@test all(n -> isa(n, Vector{Float64}), nodes)
				@test nodes[1]≈[-1.0, -1.0] atol=1e-10
				@test nodes[end] ≈ [1.0, 1.0]
				@test sum(weights)≈4.0 atol=1e-10
			end

			@testset "3D case" begin
				p, dim = 3, 3
				nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)
				@test length(nodes) == p^3
				@test length(weights) == p^3
				@test all(n -> isa(n, Vector{Float64}), nodes)
				@test nodes[1] ≈ [-1.0, -1.0, -1.0]
				@test nodes[end] ≈ [1.0, 1.0, 1.0]
				@test sum(weights)≈8.0 atol=1e-10
			end
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError SpectralMethods.gauss_legendre_with_boundary_nd(2, 2)
		end

		@testset "Edge cases" begin
			@testset "Minimum valid polynomial degree" begin
				p, dim = 3, 2
				nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)
				@test length(nodes) == p^2
				@test length(weights) == p^2
			end

			@testset "Large polynomial degree" begin
				p, dim = 20, 1
				nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)
				@test length(nodes) == p
				@test length(weights) == p
			end
		end

		@testset "Invalid cases" begin
			@testset "Invalid polynomial degree" begin
				@test_throws ArgumentError SpectralMethods.gauss_legendre_with_boundary_nd(2, 2)
			end
		end

		@testset "Numerical integration accuracy" begin
			p, dim = 5, 1
			nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)

			# Integrate f(x) = x^2 over [-1, 1], exact integral is 2/3
			integral = sum(w * n[1]^2 for (w, n) in zip(weights, nodes))
			@test integral≈2.0 / 3.0 atol=1e-10

			# Integrate f(x) = x^4 over [-1, 1], exact integral is 2/5
			integral = sum(w * n[1]^4 for (w, n) in zip(weights, nodes))
			@test integral≈2.0 / 5.0 atol=1e-10
		end

		@testset "Symmetry check" begin
			p, dim = 4, 1
			nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)

			mid = div(length(nodes), 2)
			for i in 1:mid
				@test nodes[i][1]≈-nodes[end - i + 1][1] atol=1e-10
				@test weights[i]≈weights[end - i + 1] atol=1e-10
			end
		end

		@testset "2D numerical integration accuracy" begin
			p, dim = 4, 2
			nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)

			# Integrate f(x, y) = x^2 + y^2 over [-1, 1] x [-1, 1], exact integral is 8/3
			integral = sum(w * (n[1]^2 + n[2]^2) for (w, n) in zip(weights, nodes))
			@test integral≈8.0 / 3.0 atol=1e-10
		end

		@testset "3D numerical integration accuracy" begin
			p, dim = 3, 3
			nodes, weights, nodes_interior, weights_interior = SpectralMethods.gauss_legendre_with_boundary_nd(p, dim)

			# Integrate f(x, y, z) = x + y + z over [-1, 1] x [-1, 1] x [-1, 1], exact integral is 0
			integral = sum(w * (n[1] + n[2] + n[3]) for (w, n) in zip(weights, nodes))
			@test integral≈0.0 atol=1e-10
		end
	end

	@testset "barycentric_weights Tests" begin
		@testset "Valid Cases" begin
			nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(3)
			weights = barycentric_weights(nodes_with_boundary)
			expected_weights = [0.5, -1.0, 0.5]
			@test all(isapprox(weights[i], expected_weights[i], atol = 1e-8) for i in eachindex(weights))
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError barycentric_weights([0.0, 0.5])
		end

		@testset "Edge Cases" begin
			nodes = [0.0, 0.1, 0.2]
			weights = barycentric_weights(nodes)
			expected_weights = [50.0, -100.0, 50.0]
			@test all(isapprox(weights[i], expected_weights[i], atol = 1e-8) for i in eachindex(weights))
		end
	end

	@testset "barycentric_interpolation Tests" begin
		@testset "Valid Cases" begin
			nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(3)
			values = [0.0, 1.0, 0.0]
			x = 0.0
			interpolated_value = barycentric_interpolation(nodes_with_boundary, values, x)
			@test isapprox(interpolated_value, 1.0, atol = 1e-8)
		end

		@testset "Invalid Cases" begin
			nodes = [0.0, 0.5]
			values = [0.0, 1.0]
			x = 0.25
			@test_throws ArgumentError barycentric_interpolation(nodes, values, x)
		end

		@testset "Edge Cases" begin
			nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(3)
			values = [0.0, 0.5, 1.0]
			x = -1.0
			interpolated_value = barycentric_interpolation(nodes_with_boundary, values, x)
			@test isapprox(interpolated_value, 0.0, atol = 1e-8)
		end
	end

	@testset "interpolate_nd Tests" begin
		@testset "Valid Cases" begin
			nodes_x, _, _, _ = create_legendre_nodes_and_weights(5)
			nodes_y, _, _, _ = create_legendre_nodes_and_weights(5)
			nodes = [nodes_x, nodes_y]
			values = [0.0 1.0 2.0 3.0 4.0;
					  1.0 2.0 3.0 4.0 5.0;
					  2.0 3.0 4.0 5.0 6.0;
					  3.0 4.0 5.0 6.0 7.0;
					  4.0 5.0 6.0 7.0 8.0]
			point = [0.0, 0.0]
			interpolated_value = interpolate_nd(nodes, values, point)
			@test isapprox(interpolated_value, 4.0, atol = 1e-8)
		end

		@testset "Invalid Cases" begin
			nodes_x, _, _, _ = create_legendre_nodes_and_weights(5)
			nodes_y, _, _, _ = create_legendre_nodes_and_weights(5)
			nodes = [nodes_x, nodes_y]
			values = [0.0 1.0 2.0 3.0 4.0; 1.0 2.0 3.0 4.0 5.0; 2.0 3.0 4.0 5.0 6.0; 3.0 4.0 5.0 6.0 7.0; 4.0 5.0 6.0 7.0 8.0]
			point = [0.0]
			@test_throws ArgumentError interpolate_nd(nodes, values, point)
		end

		@testset "Edge Cases" begin
			nodes_x, _, _, _ = create_legendre_nodes_and_weights(5)
			nodes_y, _, _, _ = create_legendre_nodes_and_weights(5)
			nodes = [nodes_x, nodes_y]
			values = [0.0 1.0 2.0 3.0 4.0; 1.0 2.0 3.0 4.0 5.0; 2.0 3.0 4.0 5.0 6.0; 3.0 4.0 5.0 6.0 7.0; 4.0 5.0 6.0 7.0 8.0]
			point = [-1.0, -1.0]
			interpolated_value = interpolate_nd(nodes, values, point)
			@test isapprox(interpolated_value, 0.0, atol = 1e-8)
		end
	end

	@testset "Lagrange_polynomials Tests" begin
		@testset "Valid Cases" begin
			nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(3)
			polynomials = Lagrange_polynomials(nodes_with_boundary)
			@test size(polynomials) == (3, 3)
			@test isapprox(polynomials[1, 1], 1.0, atol = 1e-8)
		end

		@testset "Invalid Cases" begin
			@test_throws MethodError Lagrange_polynomials([Vector(0.0), Vector(0.5)])
		end

		@testset "Edge Cases" begin
			nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(3)
			polynomials = Lagrange_polynomials(nodes_with_boundary)
			@test size(polynomials) == (3, 3)
			@test isapprox(polynomials[1, 1], 1.0, atol = 1e-8)
		end
	end

	@testset "create_basis_functions Tests" begin
		@testset "Valid Cases" begin
			for (degree, dim) in [(3, 2), (4, 2), (5, 3)]
				basis_functions = create_basis_functions(degree, dim)
				expected_num_functions = binomial(degree + dim, dim)
				@test length(basis_functions) == expected_num_functions
				@test all(f -> f isa KSBasisFunction, basis_functions)

				for f in basis_functions
					coords = ones(Float64, dim)
					result = f.function_handle(coords)
					@test isa(result, Real)
				end
			end
		end

		@testset "Edge Cases" begin
			degree, dim = 3, 1
			basis_functions = create_basis_functions(degree, dim)
			@test length(basis_functions) == 4

			@test isapprox(basis_functions[1].function_handle([0.0]), 1.0, atol = 1e-6)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError create_basis_functions(0, 1)
			@test_throws ArgumentError create_basis_functions(1, 0)
		end
	end

	@testset "Differentiation Matrices" begin
		f(x) = sin.(x)
		df(x) = cos.(x)
		d2f(x) = -sin.(x)
		d3f(x) = -cos.(x)

		@testset "derivative_matrix! tests" begin
			@testset "Valid Cases" begin
				for n in [15, 25, 50]
					nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(n)
					D = derivative_matrix!(nodes_with_boundary)
					x_vals = nodes_with_boundary
					f_vals = f(x_vals)
					df_vals = df(x_vals)
					@test size(D) == (n, n)
					@test all(isapprox.(df_vals, D * f_vals, atol = 1e-6))
				end
			end

			@testset "Invalid Cases" begin
				@test_throws ArgumentError derivative_matrix!([1.0, 2.0])
				@test_throws ArgumentError derivative_matrix!(Vector{Float64}())
			end

			@testset "Edge Cases" begin
				nodes = [-1.0, 0.0, 1.0]
				D = derivative_matrix!(nodes)
				@test size(D) == (3, 3)
				@test isapprox(D[2, :], [-0.5, 0.0, 0.5], atol = 1e-8)
			end
		end

		@testset "kth_derivative_matrix! tests" begin
			@testset "Valid Cases" begin
				for n in [15, 25, 50]
					nodes_with_boundary, _, _, _ = create_legendre_nodes_and_weights(n)
					Dks = kth_derivative_matrix!(nodes_with_boundary, 3)
					x_vals = nodes_with_boundary
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
				@test_throws ArgumentError kth_derivative_matrix!([1.0, 2.0], 1)
				@test_throws ArgumentError kth_derivative_matrix!([1.0, 2.0, 3.0], 0)
				@test_throws ArgumentError kth_derivative_matrix!([1.0, 2.0, 3.0], -1)
			end
		end

		# Test for `derivative_matrix_nd`
		# Test for `derivative_matrix_nd`
		@testset "derivative_matrix_nd tests" begin
			# 1D Case: Simple linear case
			points_1d = [[-1.0, 0.0, 1.0]]
			dim = 1
			order = 1
			D_matrices = derivative_matrix_nd(points_1d, dim, order)
			D = D_matrices[1]

			# Check if D is a 3x3 sparse matrix
			@test size(D) == (3, 3)
			@test issparse(D)

			# Verify derivative matrix values for linear basis functions
			@test abs(D[1, 1] - (-1.5)) < 1e-8
			@test abs(D[1, 2] - (2.0)) < 1e-8
			@test abs(D[1, 3] - (-0.5)) < 1e-8

			@test abs(D[2, 1] - (-0.5)) < 1e-8
			@test abs(D[2, 2] - 0.0) < 1e-8
			@test abs(D[2, 3] - 0.5) < 1e-8

			@test abs(D[3, 1] - (0.5)) < 1e-8
			@test abs(D[3, 2] - (-2.0)) < 1e-8
			@test abs(D[3, 3] - (1.5)) < 1e-8

			# 2D Case: Check if dimensions match and derivative matrices are computed
			points_2d = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
			dim = 2
			D_matrices = derivative_matrix_nd(points_2d, dim, order)

			@test length(D_matrices) == 2
			@test size(D_matrices[1]) == (3, 3)
			@test size(D_matrices[2]) == (3, 3)
			@test issparse(D_matrices[1])
			@test issparse(D_matrices[2])

			# Check higher-order derivative (second derivative)
			D_matrices_second_order = derivative_matrix_nd(points_1d, 1, 2)
			D_second = D_matrices_second_order[1]

			@test size(D_second) == (3, 3)
			@test issparse(D_second)
		end

		# Test for `enforce_ck_continuity!`
		@testset "enforce_ck_continuity! tests" begin
			# Simple 3x3 matrix
			D = spzeros(3, 3)
			D[1, 2] = 1.0
			D[2, 1] = -1.0
			D[2, 3] = 1.0
			D[3, 2] = -1.0

			# Apply C^1 continuity
			D_continuous = enforce_ck_continuity!(D, 1)

			# Check the first and last row
			@test abs(D_continuous[1, 1] - 1.0) < 1e-8
			@test abs(D_continuous[1, 2]) < 1e-8
			@test abs(D_continuous[1, 3]) < 1e-8

			@test abs(D_continuous[2, 1] + 1) < 1e-8
			@test abs(D_continuous[2, 2] - 0.0) < 1e-8
			@test abs(D_continuous[2, 3] - 1.0) < 1e-8

			@test abs(D_continuous[3, 1]) < 1e-8
			@test abs(D_continuous[3, 2]) < 1e-8
			@test abs(D_continuous[3, 3] - 1.0) < 1e-8

			# Larger matrix (5x5) and check edge cases
			D_large = spzeros(5, 5)
			D_large[2, 1] = -2.0
			D_large[2, 3] = 2.0
			D_large[4, 3] = -2.0
			D_large[4, 5] = 2.0

			# Apply C^2 continuity
			D_large_cont = enforce_ck_continuity!(D_large, 2)

			# Check first and last two rows
			for i in 1:2
				@test abs(D_large_cont[i, i] - 1.0) < 1e-8
				@test sum(abs.(D_large_cont[i, :])) == 1.0
			end

			for i in 4:5
				@test abs(D_large_cont[i, i] - 1.0) < 1e-8
				@test sum(abs.(D_large_cont[i, :])) == 1.0
			end

			# Check middle row remains unchanged
			@test abs(D_large_cont[3, 1]) < 1e-8
			@test abs(D_large_cont[3, 2]) < 1e-8
			@test abs(D_large_cont[3, 3] - 0.0) < 1e-8
			@test abs(D_large_cont[3, 4]) < 1e-8
			@test abs(D_large_cont[3, 5]) < 1e-8
		end
	end
	# Test set for interpolate_nd function
	@testset "interpolate_nd Tests" begin
		@testset "Valid Cases" begin
			nodes_x, _, _, _ = create_legendre_nodes_and_weights(3)
			nodes_y, _, _, _ = create_legendre_nodes_and_weights(3)
			nodes = [nodes_x, nodes_y]
			values = [0.0 1.0 2.0; 1.0 2.0 3.0; 2.0 3.0 4.0]
			point = [0.0, 0.0]
			interpolated_value = interpolate_nd(nodes, values, point)
			@test isapprox(interpolated_value, 2.0)
		end

		@testset "Invalid Cases" begin
			nodes_x, _, _, _ = create_legendre_nodes_and_weights(3)
			nodes_y, _, _, _ = create_legendre_nodes_and_weights(3)
			nodes = [nodes_x, nodes_y]
			values = [0.0 1.0 2.0; 1.0 2.0 3.0; 2.0 3.0 4.0]
			point = [0.0]
			@test_throws ArgumentError interpolate_nd(nodes, values, point)
		end

		@testset "Edge Cases" begin
			nodes_x, _, _, _ = create_legendre_nodes_and_weights(3)
			nodes_y, _, _, _ = create_legendre_nodes_and_weights(3)
			nodes = [nodes_x, nodes_y]
			values = [0.0 1.0 2.0; 1.0 2.0 3.0; 2.0 3.0 4.0]
			point = [-1.0, -1.0]
			interpolated_value = interpolate_nd(nodes, values, point)
			@test isapprox(interpolated_value, 0.0)
		end
	end

	# Test set for derivative_matrix! function
	@testset "derivative_matrix! Tests" begin
		@testset "Valid Cases" begin
			nodes = [-1.0, 0.0, 1.0]
			D = derivative_matrix!(nodes)
			@test size(D) == (3, 3)
			@test isapprox(D[2, 1], -0.5)
			@test isapprox(D[2, 3], 0.5)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError derivative_matrix!([1.0, 2.0])
		end

		@testset "Edge Cases" begin
			nodes = [-1.0, 0.0, 1.0]
			D = derivative_matrix!(nodes)
			@test size(D) == (3, 3)
		end
	end

	# Test set for kth_derivative_matrix! function
	@testset "kth_derivative_matrix! Tests" begin
		@testset "Valid Cases" begin
			nodes = [-1.0, 0.0, 1.0]
			Dks = kth_derivative_matrix!(nodes, 2)
			@test length(Dks) == 2
			@test size(Dks[1]) == (3, 3)
			@test size(Dks[2]) == (3, 3)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError kth_derivative_matrix!([1.0, 2.0], 0)
		end

		@testset "Edge Cases" begin
			nodes = [-1.0, 0.0, 1.0]
			Dks = kth_derivative_matrix!(nodes, 1)
			@test length(Dks) == 1
		end
	end

	# Test set for derivative_matrix_nd function
	@testset "derivative_matrix_nd Tests" begin
		@testset "Valid Cases" begin
			points = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
			D_matrices = derivative_matrix_nd(points, 2, 1)
			@test length(D_matrices) == 2
			@test size(D_matrices[1]) == (3, 3)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError derivative_matrix_nd([[-1.0, 0.0]], 1, 0)
		end
	end

	# Test set for enforce_ck_continuity! function
	@testset "enforce_ck_continuity! Tests" begin
		@testset "Valid Cases" begin
			D = spzeros(3, 3)
			D[1, 2] = 1.0
			D_continuous = enforce_ck_continuity!(D, 1)
			@test isapprox(D_continuous[1, 1], 1.0)
			@test isapprox(D_continuous[3, 3], 1.0)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError enforce_ck_continuity!(spzeros(3, 3), -1)
		end
	end

	# Test set for is_near_zero function
	@testset "is_near_zero Tests" begin
		@testset "Valid Cases" begin
			@test is_near_zero(1e-16)
			@test !is_near_zero(1e-5)
		end
	end

	# Test set for create_basis_functions function
	@testset "create_basis_functions Tests" begin
		@testset "Valid Cases" begin
			basis_functions = create_basis_functions(3, 2)
			@test length(basis_functions) == binomial(3 + 2, 2)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError create_basis_functions(0, 1)
		end
	end

	# Test set for scale_derivative_matrix function
	@testset "scale_derivative_matrix Tests" begin
		@testset "Valid Cases" begin
			D = spzeros(3, 3)
			D[1, 2] = 1.0
			scaled_D = scale_derivative_matrix(D, 2.0)
			@test isapprox(scaled_D[1, 2], 2.0)
		end
	end

	# Test set for create_differentiation_matrices function
	@testset "create_differentiation_matrices Tests" begin
		@testset "Valid Cases" begin
			nodes = [-1.0, 0.0, 1.0]
			diff_matrices = create_differentiation_matrices(nodes, 2)
			@test length(diff_matrices) == 2
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError create_differentiation_matrices([1.0, 2.0], 0)
		end
	end

	# Test set for quadrature_matrix_nd function
	@testset "quadrature_matrix Tests" begin
		@testset "Valid Cases" begin
			_, _, _, weights = expected_values(:case1)
			Q = quadrature_matrix!(weights)
			@test size(Q) == (3, 3)
			@test isapprox(Q[1, 1], 5 / 9)
			@test isapprox(Q[2, 2], 8 / 9)
			@test isapprox(Q[3, 3], 5 / 9)
		end

		@testset "Invalid Cases" begin
			@test_throws MethodError quadrature_matrix!("not a number")
		end
	end

	@testset "compute_integral Tests" begin
		@testset "Valid Cases" begin
			_, _, nodes, weights = expected_values(:case1)
			Q = quadrature_matrix!(weights)
			values = nodes .^ 4
			integral = compute_integral(weights, values)
			@test isapprox(integral, 2 / 5)
		end

		@testset "Invalid Cases" begin
			@test_throws MethodError compute_integral([0.0, 1.0], spzeros(3, 3))
		end
	end

	@testset "quadrature_matrix_nd Tests" begin
		@testset "Valid Cases" begin
			points, weights = [gausslegendre(3), gausslegendre(3)]
			Q_nd = quadrature_matrix_nd(weights, 2)
			@test size(Q_nd) == (9, 9)

			points, weights = (gausslegendre(3), gausslegendre(3))
			Q_nd = quadrature_matrix_nd(weights, 2)
			@test size(Q_nd) == (9, 9)
		end

		@testset "Invalid Cases" begin
			@test_throws MethodError quadrature_matrix_nd([[-1.0, 0.0]], [[1.0]])
		end
	end

	@testset "compute_integral_nd Tests" begin
		@testset "Valid Cases" begin
			values = [1.0 for _ in 1:9]
			points = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
			weights = [[0.5, 1.0, 0.5], [0.5, 1.0, 0.5]]
			Q_nd = quadrature_matrix_nd(weights, 2)
			integral = compute_integral_nd(values, Q_nd)
			@test isapprox(integral, 4.0)
		end

		@testset "Invalid Cases" begin
			@test_throws ArgumentError compute_integral_nd([0.0, 1.0], spzeros(9, 9))
		end
	end

	@testset "scale_quadrature_matrix Tests" begin
		@testset "Valid Cases" begin
			Q = spzeros(3, 3)
			Q[1, 1] = 0.5
			scaled_Q = scale_quadrature_matrix!(Q, 2.0)
			@test isapprox(scaled_Q[1, 1], 1.0)
		end
	end

	@testset "OCFE Mesh Tests" begin
		num_elements_2d = [5, 10]
		num_elements_3d = [5, 10, 5]
		poly_degree_2d = [5, 5]
		poly_degree_3d = [5, 5, 5]
		derivative_order = 1
		continuity_order = 1
		cartesian_2d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
        cartesian_3d = KSCartesianCoordinates(((0.0, 1.0), (-1, 1), (0, 2.0)))

		cylindrical_3d = KSCylindricalCoordinates((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
		polar_2D = KSPolarCoordinates((0, 1), (0, 2 * pi))
		spherical_3D = KSSphericalCoordinates((0, 1), (0, pi), (0, 2 * pi))

		# Test valid cases with different coordinate systems
		@testset "Valid Cases with Coordinate Systems" begin
			# Test case 1: Cartesian 2D case
			mesh = create_ocfe_mesh(cartesian_2d, num_elements_2d, poly_degree_2d, continuity_order, 2)
			@test mesh isa KSMesh
			@test length(mesh.elements) == prod(num_elements_2d)
			@test all(el -> el isa KSElement, mesh.elements)

			# Test case 2: Cartesian 3D case
			mesh = create_ocfe_mesh(cartesian_3d, num_elements_3d, poly_degree_3d, continuity_order, 3)
			@test mesh isa KSMesh
			@test length(mesh.elements) == prod(num_elements_3d)
			@test all(el -> el isa KSElement, mesh.elements)

			# Test case 3: Cylindrical 3D case
			mesh = create_ocfe_mesh(cylindrical_3d, num_elements_3d, poly_degree_3d, continuity_order, 3)
			@test mesh isa KSMesh
			@test length(mesh.elements) == prod(num_elements_3d)
			@test all(el -> el isa KSElement, mesh.elements)
		end

		# Test invalid cases with incorrect domain and elements
		@testset "Invalid Cases" begin
			# Test case 1: Domain dimensionality mismatch
			@test_throws MethodError create_ocfe_mesh([(0.0, 1.0)], num_elements_2d, poly_degree_3d, continuity_order, derivative_order, 3)

			# Test case 2: Mismatched number of elements
			@test_throws MethodError create_ocfe_mesh(polar_2D, [2], poly_degree_2d, continuity_order, derivative_order, 2)

			# Test case 3: Non-positive number of elements
			@test_throws MethodError create_ocfe_mesh(cartesian_2d, [0, 2], poly_degree_2d, continuity_order, derivative_order, 2)
			# Test case 4: Invalid polynomial degree
			@test_throws MethodError create_ocfe_mesh(cartesian_2d, num_elements_2d, 0, continuity_order, derivative_order, 2)

			# Test case 5: Unknown coordinate system
			@test_throws MethodError create_ocfe_mesh(cartesian_2d, num_elements_2d, poly_degree_2d, continuity_order, derivative_order, 2)
		end
	end
end
