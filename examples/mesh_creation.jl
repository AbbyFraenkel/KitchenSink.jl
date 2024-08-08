# # Mesh Creation in KitchenSink

# This document provides examples and explanations for creating meshes using the improved mesh creation functions in KitchenSink.

# ## Creating a Mesh

# To create a mesh, you need to specify the domain, coordinate system, number of elements, and polynomial degree. Here's an example:

# ```julia
# using KitchenSink

# # Define the domain (2D in this case)
# domain = ((0.0, 1.0), (0.0, 1.0))

# # Define the coordinate system
# coord_system = KSCartesianCoordinates{2, Float64}((0.0, 0.0))

# # Specify the number of elements in each dimension
# num_elements = (4, 4)

# # Set the polynomial degree
# polynomial_degree = 3

# # Create the mesh
# mesh = create_mesh(domain, coord_system, num_elements, polynomial_degree)
# ```

# This will create a 2D mesh with 4x4 elements, each using polynomial degree 3, over the domain [0,1] x [0,1] in Cartesian coordinates.

# ## Accessing Mesh Properties

# Once you have created a mesh, you can access its properties:

# ```julia
# # Number of elements
# println("Number of elements: ", length(mesh.elements))

# # Dimensions of the mesh
# println("Mesh dimensions: ", mesh.dimensions)

# # Polynomial degree used
# println("Polynomial degree: ", mesh.elements[1].polynomial_degree)
# ```

# ## Working with Elements

# You can access individual elements and their properties:

# ```julia
# # Access the first element
# element = mesh.elements[1]

# # Get the nodes of the element
# println("Element nodes: ", element.points)

# # Get the collocation points of the element
# println("Collocation points: ", element.collocation_points.points)

# # Access the differentiation matrices
# println("Differentiation matrix for x-direction: ", element.differentiation_matrices[1])
# println("Differentiation matrix for y-direction: ", element.differentiation_matrices[2])
# ```

# ## Creating Meshes in Different Coordinate Systems

# KitchenSink supports different coordinate systems. Here's an example of creating a mesh in polar coordinates:

# ```julia
# # Define a domain in polar coordinates (r, θ)
# polar_domain = ((0.0, 1.0), (0.0, 2π))

# # Define the polar coordinate system
# polar_coord_system = KSPolarCoordinates{Float64}(1.0, 0.0)

# # Create the mesh
# polar_mesh = create_mesh(polar_domain, polar_coord_system, (5, 8), 3)
# ```

# This creates a polar mesh with 5 radial elements and 8 angular elements, each using polynomial degree 3.

# ## Advanced Usage: Creating a 3D Mesh

# For 3D problems, you can create a mesh as follows:

# ```julia
# # Define a 3D domain
# domain_3d = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

# # Define the 3D Cartesian coordinate system
# coord_system_3d = KSCartesianCoordinates{3, Float64}((0.0, 0.0, 0.0))

# # Specify the number of elements in each dimension
# num_elements_3d = (3, 3, 3)

# # Create the 3D mesh
# mesh_3d = create_mesh(domain_3d, coord_system_3d, num_elements_3d, 2)

# println("Number of 3D elements: ", length(mesh_3d.elements))
# println("3D mesh dimensions: ", mesh_3d.dimensions)
# ```

# This creates a 3D mesh with 3x3x3 elements, each using polynomial degree 2, over the domain [0,1] x [0,1] x [0,1] in Cartesian coordinates.

# Remember that when working with higher-dimensional meshes, the computational complexity increases significantly. Always consider the trade-off between accuracy and computational cost when choosing the number of elements and polynomial degree.
