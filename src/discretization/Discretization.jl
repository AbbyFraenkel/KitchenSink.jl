module Discretization

using .Types
using FastGaussQuadrature, LinearAlgebra, SparseArrays, DocStringExtensions

# Export OCFE
include("hierarchical_basis.jl")

export hierarchical_Legendre_interpolation, legendreP, hierarchical_Legendre


# Export Spectral Methods

include("spectral_methods.jl")
export legendre_basis_functions, EO_matrix_derivative, kth_derivative_matrix
export barycentric_weights, shifted_adapted_gauss_legendre, barycentric_interpolation


end
