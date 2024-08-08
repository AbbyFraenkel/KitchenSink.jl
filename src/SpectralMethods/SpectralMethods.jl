module SpectralMethods

using LinearAlgebra, FastGaussQuadrature, SparseArrays
using ..KSTypes

# Export functions from collocation_points.jl
export create_nodes, gauss_legendre_with_boundary_nd
include("collocation_points.jl")


# Export functions from basis_functions.jl
export barycentric_weights, Lagrange_polynomials, barycentric_interpolation
export interpolate_nd
include("basis_functions.jl")


# Export functions from differentiation_matrices.jl
export derivative_matrix, kth_derivative_matrix, derivative_matrix_nd
export enforce_ck_continuity
include("differentiation_matrices.jl")


end # module SpectralMethods
