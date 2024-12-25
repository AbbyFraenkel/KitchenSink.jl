module ProblemTypes

using LinearAlgebra, SparseArrays, Statistics, Base.Iterators
using ..KSTypes, ..CoordinateSystems, ..NumericUtilities
using ..CacheManagement, ..SpectralMethods, ..BoundaryConditions
using ..Transforms, ..Preconditioners, ..LinearSolvers
using Base.Threads

# Initialize cache constants
# const SOLVER_CACHE = CacheManager{Any}(10000)
# const STANDARD_CELL_CACHE = CacheManager{Any}(5000)
export
    # Core DOF functions
    has_boundary_condition,
    get_system_dof,
    get_mesh_dof,
    get_physical_mesh_dof,
    get_boundary_mesh_dof,
    compute_boundary_dofs,

    # Temporal DOF functions
    get_temporal_dof,
    has_temporal_dof,

    # Combined calculations
    get_total_problem_dof,
    compute_kernel_complexity,
    compute_coupling_points

include("dof.jl")

# Core validation functions
export validate_problem_setup,
    validate_temporal_dofs,
    validate_problem_specific_config,
    validate_time_span,
    verify_solution_properties,
    verify_solution_specific,
    validate_domains,
    are_coordinate_systems_compatible,
    is_valid_domain,
    get_coordinate_system,
    get_problem_dimension,
    get_coordinate_domains,
    modify_problem,
    validate_boundary_conditions,
    strict_domain_overlap

# Keep internal helper functions commented out
# export determine_coordinate_system_compatibility,
#     validate_mesh_compatibility,
#     verify_solution_dimensions

include("validation.jl")
export create_system_matrix_and_vector,
	generate_cache_key,
	assemble_cell_contributions!,
	initialize_local_matrices,
	add_coupling_terms!,
	add_coupling_block!,
	add_to_global_system!,
	finalize_system,
	ensure_diagonal_dominance!,
	apply_boundary_conditions

include("assemble_problems.jl")
export add_problem_contributions!,
	handle_point_arithmetic,
	euclidean_distance,
	evaluate_at_point,
	validate_coupled_problem_domains,
	apply_differential_operators  # Add this export

include("problem_implementations.jl")

# Add after imports, before const declaration

export update_mesh_connectivity!,
	map_nodes_to_computational_domain,
	update_cell_neighbors!,
	update_boundary_cells!,
	create_node_cell_map,
	get_cell_center,
	is_on_boundary,
	get_opposite_direction

include("mesh_operations.jl")

export solve_problem,
	solve_stabilized_system,
	transform_and_verify,
	reshape_temporal_solution,
	transform_to_physical_space,
	needs_transform,
	get_expected_system_size,
	validate_dof_consistency,
	get_cell_operators
include("basic_solve.jl")

# Add to exports

end # module ProblemTypes
