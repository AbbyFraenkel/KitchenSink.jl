module OptimalControlPromblems
using .Types, .SystemFormulation

# Export OCFE
include("OCFE.jl")
export ocfe_time_step_with_delay, ocfe_time_step_with_integral, solve_state_equations
export solve_adjoint_equations, update_control, compute_gradient

end
