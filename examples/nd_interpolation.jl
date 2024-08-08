using KitchenSink.SpectralMethods

# Generate nodes and weights for 2D
degree = 5
a, b = -1.0, 1.0
nodes_x, weights_x = create_nodes(degree, a, b)
nodes_y, weights_y = create_nodes(degree, a, b)
nodes = [nodes_x, nodes_y]

# Define the function to interpolate
f(x, y) = sin(pi * x) * cos(pi * y)
values = [f(x, y) for x in nodes_x, y in nodes_y]

# Point to interpolate
point = [0.5, -0.5]

# Perform interpolation
interpolated_value = interpolate_nd(nodes, values, point)

# Print the result
println("Nodes X: ", nodes_x)
println("Nodes Y: ", nodes_y)
println("Interpolated value at ", point, ": ", interpolated_value)
