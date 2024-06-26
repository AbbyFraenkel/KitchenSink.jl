module AdaptiveMultiGrid

#Export iteratorators
include("iterators.jl")
export smooth, gauss_seidel, sor, restrict, prolong, v_cycle

# Export multigrid implementation
include("implementation.jl")
export create_grid_hierarchy, create_operators, adaptive_multigrid_solver
end
