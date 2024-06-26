module SystemFormulation

using .Types
using FastGaussQuadrature, LinearAlgebra, SparseArrays, DocStringExtensions

# Export DAE Index Reduction
include("IndexReduction.jl")
export reduce_index_numerically

# Export
include("systemUtils.jl")
export validate_inputs, initial_state, assemble_system, assemble_system_with_delay
export assemble_system_with_integral, linear_index, integrate_basis, integrate_forcing
export integrate_forcing_with_delay, integrate_forcing_with_integral, interpolate_solution
export apply_boundary_conditions!

end
