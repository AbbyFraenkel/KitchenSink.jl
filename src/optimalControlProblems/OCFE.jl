
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

"""
    create_provisional_ocfe_mesh(n_nodes, n_elements, a = 0, b = 1)

Creates a provisional mesh for the Orthogonal Collocation Finite Element (OCFE) method.

# Arguments
- `n_nodes::Int`: The number of nodes per element.
- `n_elements::Int`: The number of elements in the mesh.
- `a::Float64 = 0`: The lower bound of the interval.
- `b::Float64 = 1`: The upper bound of the interval.

# Returns
- `mesh::Matrix{Float64}`: A matrix representing the mesh with dimensions `n_elements x n_nodes`.

# Notes
- The mesh is created using a uniform distribution of collocation points and the Gauss-Legendre quadrature nodes.
- The elements are defined by the interval `[a, b]` and the number of elements `n_elements`.
"""
function create_provisional_ocfe_mesh(n_nodes, n_elements, a = 0, b = 1)
    xn, wn = ShiftedAdaptedGaussLegendre(n_nodes)

    # Initialize mesh with uniform distribution of collocation points
    # elements = LinRange(a, b, n_elements)
    elements = collect(range(a, b, length = n_elements + 1))
    h = elements[2] - elements[1]
    xn_e = h.* xn

    # Create elements based on the mesh nodes
    mesh = zeros(n_elements, n_nodes)
    for i in 1:n_elements
        for j in 1:n_nodes
            mesh[i, j] = h.* (i.- 1).+ (xn_e[j])
        end
    end

    return mesh
end
