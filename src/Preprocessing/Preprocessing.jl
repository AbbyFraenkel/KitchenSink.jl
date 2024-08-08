
module Preprocessing

using LinearAlgebra, SparseArrays, IncompleteLU, AlgebraicMultigrid
using ..KSTypes, ..SpectralMethods, ..CoordinateSystems, ..CommonMethods

export preprocess_mesh, generate_initial_mesh, refine_mesh, estimate_mesh_error
export create_OCFE_discretization
include("discretization.jl")

export create_mesh, create_nodes, create_elements, create_collocation_points
include("mesh_creation.jl")


export update_tensor_product_masks!, restore_interface_compatibility!
export deactivate_internal_boundary!, update_location_matrices!, linear_index
export update_tensor_product_masks_with_trunk!

include("tensor_product_masks.jl")

end
