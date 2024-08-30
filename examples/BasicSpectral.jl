using LinearAlgebra, SparseArrays, FastGaussQuadrature, Plots
using ..SpectralMethods

# Test Problem 1: 1D Polynomial Function
function test_problem_1()
	# Define the function and its analytical derivative
	f(x) = x^3 - 2x^2 + x
	df_analytical(x) = 3x^2 - 4x + 1

	# Generate Legendre nodes
	p = 5  # Polynomial degree for refinement
	nodes, _, _, _ = create_legendre_nodes_and_weights(p)

	# Compute numerical derivative using the provided derivative matrix function
	D = derivative_matrix!(nodes)

	# Evaluate the function values at the nodes
	f_values = f.(nodes)

	# Compute the numerical derivative
	df_numerical = D * f_values

	# Compute the analytical derivatives at the nodes
	df_analytical_values = df_analytical.(nodes)

	# Calculate point-wise error
	pointwise_error = abs.(df_numerical - df_analytical_values)

	# Calculate relative error
	error = norm(df_numerical - df_analytical_values) / norm(df_analytical_values)

	# Plotting the function and its derivatives
	p1 = scatter(nodes, f_values, label = "Function f(x)", legend = :top, title = "1D Polynomial Function and Derivatives")
	plot!(p1, nodes, df_numerical, label = "Numerical Derivative")
	plot!(p1, nodes, df_analytical_values, label = "Analytical Derivative", linestyle = :dash)
	xlabel!("x")
	ylabel!("f(x) and Derivatives")
	display(p1)

	println("1D Problem Relative Error: ", error)

	# Save results in a dictionary
	results = Dict("nodes" => nodes,
				   "f_values" => f_values,
				   "df_numerical" => df_numerical,
				   "df_analytical_values" => df_analytical_values,
				   "pointwise_error" => pointwise_error,
				   "error" => error)

	return results
end

# Test Problem 2: 2D Trigonometric Function
function test_problem_2()
	# Define the function and its analytical partial derivatives
	f(x, y) = sin(pi * x) * cos(pi * y)
	df_dx_analytical(x, y) = pi * cos(pi * x) * cos(pi * y)
	df_dy_analytical(x, y) = -pi * sin(pi * x) * sin(pi * y)

	# Generate 2D Legendre nodes
	p = 5  # Polynomial degree for refinement
	nodes_x, _, _, _ = create_legendre_nodes_and_weights(p)
	nodes_y, _, _, _ = create_legendre_nodes_and_weights(p)
	points = [nodes_x, nodes_y]

	# Compute numerical derivative matrices for each dimension
	D_matrices = derivative_matrix_nd(points, 2)

	# Evaluate the function values on the 2D grid
	f_values = [f(x, y) for x in nodes_x, y in nodes_y]

	# Compute numerical partial derivatives
	df_dx_numerical = D_matrices[1] * f_values
	df_dy_numerical = f_values * D_matrices[2]'

	# Compute analytical partial derivatives
	df_dx_analytical_values = [df_dx_analytical(x, y) for x in nodes_x, y in nodes_y]
	df_dy_analytical_values = [df_dy_analytical(x, y) for x in nodes_x, y in nodes_y]

	# Calculate point-wise errors
	pointwise_error_dx = abs.(df_dx_numerical - df_dx_analytical_values)
	pointwise_error_dy = abs.(df_dy_numerical - df_dy_analytical_values)

	# Calculate relative errors
	error_dx = norm(df_dx_numerical - df_dx_analytical_values) / norm(df_dx_analytical_values)
	error_dy = norm(df_dy_numerical - df_dy_analytical_values) / norm(df_dy_analytical_values)

	# Plotting numerical vs analytical derivatives
	p2 = heatmap(nodes_x, nodes_y, df_dx_numerical, title = "Numerical ∂f/∂x", xlabel = "x", ylabel = "y")
	display(p2)
	p3 = heatmap(nodes_x, nodes_y, df_dx_analytical_values, title = "Analytical ∂f/∂x", xlabel = "x", ylabel = "y")
	display(p3)
	p4 = heatmap(nodes_x, nodes_y, pointwise_error_dx, title = "Error ∂f/∂x", xlabel = "x", ylabel = "y")
	display(p4)

	p5 = heatmap(nodes_x, nodes_y, df_dy_numerical, title = "Numerical ∂f/∂y", xlabel = "x", ylabel = "y")
	display(p5)
	p6 = heatmap(nodes_x, nodes_y, df_dy_analytical_values, title = "Analytical ∂f/∂y", xlabel = "x", ylabel = "y")
	display(p6)
	p7 = heatmap(nodes_x, nodes_y, pointwise_error_dy, title = "Error ∂f/∂y", xlabel = "x", ylabel = "y")
	display(p7)

	println("2D Problem Relative Error (dx): ", error_dx)
	println("2D Problem Relative Error (dy): ", error_dy)

	# Save results in a dictionary
	results = Dict("nodes_x" => nodes_x,
				   "nodes_y" => nodes_y,
				   "f_values" => f_values,
				   "df_dx_numerical" => df_dx_numerical,
				   "df_dy_numerical" => df_dy_numerical,
				   "df_dx_analytical_values" => df_dx_analytical_values,
				   "df_dy_analytical_values" => df_dy_analytical_values,
				   "pointwise_error_dx" => pointwise_error_dx,
				   "pointwise_error_dy" => pointwise_error_dy,
				   "error_dx" => error_dx,
				   "error_dy" => error_dy)

	return results
end

# Run the test problems and save the results
results_1d = test_problem_1()
results_2d = test_problem_2()

results_1d

results_2d
