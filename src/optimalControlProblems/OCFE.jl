
function ocfe_time_step_with_delay(
    u::Vector{Float64},
    control::Vector{Float64},
    t::Float64,
    dt::Float64,
    state_eq::Function,
    ocfe_config::Any,
    τ::Float64,
    history::Function,
    past_times::Vector{Float64},
    past_values::Vector{Float64},
    interpolation_method::Symbol = :spline,
)::Vector{Float64}
    u_next = copy(u)
    for (c, w) in zip(ocfe_config.collocation_points, ocfe_config.weights)
        t_delayed = t + c * dt - τ
        u_delayed =
            t_delayed >= 0 ?
            interpolate_solution(t_delayed, past_times, past_values, interpolation_method) :
            history(t_delayed)
        u_next += state_eq(u, control, t + c * dt, u_delayed) * w * dt
    end
    return u_next
end

function ocfe_time_step_with_integral(
    u::Vector{Float64},
    control::Vector{Float64},
    t::Float64,
    dt::Float64,
    state_eq::Function,
    ocfe_config::Any,
    K::Function,
)::Vector{Float64}
    u_next = copy(u)
    for (c, w) in zip(ocfe_config.collocation_points, ocfe_config.weights)
        integral_term = 0.0
        for s in ocfe_config.collocation_points
            integral_term +=
                K(t + c * dt, s) *
                interpolate_solution(s, past_times, past_values, interpolation_method)
        end
        u_next += state_eq(u, control, t + c * dt, integral_term) * w * dt
    end
    return u_next
end

function solve_state_equations(
    control::Vector{Float64},
    state_eq::Function,
)::Vector{Float64}
    u = initial_state(control)
    tspan = 0:0.1:1.0

    for t in tspan
        u = ocfe_time_step(u, control, t, state_eq)
    end

    return u
end
function solve_adjoint_equations(
    state_solution::Vector{Float64},
    adjoint_eq::Function,
    tmesh::Vector{Float64},
    initial_adjoint::Function,
)::Vector{Float64}
    lambda = initial_adjoint(state_solution)
    for t in reverse(tmesh)
        lambda = ocfe_time_step(lambda, state_solution, t, adjoint_eq)
    end
    return lambda
end


function update_control(
    adjoint_solution::Vector{Float64},
    current_control::Vector{Float64},
    α::Float64,
)::Vector{Float64}
    gradient = compute_gradient(adjoint_solution, current_control)
    new_control = current_control .- α .* gradient
    return new_control
end

@inline function compute_gradient(
    adjoint_solution::Vector{Float64},
    control::Vector{Float64},
)::Vector{Float64}
    return adjoint_solution .* control
end
