module Refinement

# Export mesh Refinement
include("mesh_refinement.jl")
export goal_oriented_refinement, refine_element, refine_mesh, coarsen_mesh, split_element!
export merge_elements!, update_neighbors!, find_children, refine_by_superposition
export refine_grid

# Export refinement indicators
include("indicators.jl")
export estimate_residual, compute_residuals, evaluate_at_superconvergent_points
export recover_derivatives, superconvergence_recovery_derivatives, superconvergence_recovery
export compute_smoothness_indicator, determine_refinement_strategy


end
