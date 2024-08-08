module CommonMethods

using LinearAlgebra, SparseArrays
using ..KSTypes

export estimate_error, compute_residual, compute_error_indicator,apply_differential_operator
export compute_expansion_coefficients

include("error_estimation.jl")

export assemble_system_matrix, assemble_rhs_vector, assemble_local_system_matrix, element_size
export estimate_smoothness, get_active_indices, inner_product,total_dofs
export element_dofs

include("assembly.jl")

export solve_equation, solve_multi_level, converged, initialize_solution, solve_coarse_problem
export smooth, gauss_seidel_iteration, apply_boundary_conditions!, enforce_continuity!, is_boundary_node

include("solve_implementation.jl")

end # module CommonMethods
