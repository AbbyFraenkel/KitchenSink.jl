using KitchenSink.SpectralMethods
using SparseArrays, LinearAlgebra

# Generate nodes and weights for 2D
degree = 5
a, b = -1.0, 1.0
nodes_x, weights_x = create_nodes(degree, a, b)
nodes_y, weights_y = create_nodes(degree, a, b)
nodes = [nodes_x, nodes_y]

# Compute the first derivative matrices for each dimension
D_nd = derivative_matrix_nd(nodes)

# Define the source term and boundary conditions
f(x, y) = sin(pi * x) * sin(pi * y)  # Source term
u_a, u_b = 0.0, 0.0  # Boundary conditions

# Number of nodes in each dimension
nx, ny = length(nodes_x), length(nodes_y)

# Initialize system matrix and RHS
A = spzeros(nx * ny, nx * ny)
b = zeros(nx * ny)
