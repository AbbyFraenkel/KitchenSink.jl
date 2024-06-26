module TimeStepping

export solve_pde, rk4_step, euler_step, rk45_step, solve_sde, solve_bvdae_advanced

using LinearAlgebra
using .Types

"""
    euler_step(f::Function, u::Vector{Float64}, t::Float64, dt::Float64)::Vector{Float64}

Perform a single Euler step.
"""
@inline function euler_step(
    f::Function,
    u::Vector{Float64},
    t::Float64,
    dt::Float64,
)::Vector{Float64}
    return u .+ dt .* f(u, t)
end

"""
    rk4_step(f::Function, u::Vector{Float64}, t::Float64, dt::Float64)::Vector{Float64}

Perform a single Runge-Kutta 4th order (RK4) step.
"""
@inline function rk4_step(
    f::Function,
    u::Vector{Float64},
    t::Float64,
    dt::Float64,
)::Vector{Float64}
    k1 = dt .* f(u, t)
    k2 = dt .* f(u .+ 0.5 .* k1, t + 0.5 * dt)
    k3 = dt .* f(u .+ 0.5 .* k2, t + 0.5 * dt)
    k4 = dt .* f(u .+ k3, t + dt)
    return u .+ (k1 .+ 2k2 .+ 2k3 .+ k4) ./ 6
end

"""
    rk45_step(f::Function, u::Vector{Float64}, t::Float64, dt::Float64)::Tuple{Vector{Float64}, Vector{Float64}}

Perform a single Runge-Kutta-Fehlberg 45 (RK45) step.
"""
@inline function rk45_step(
    f::Function,
    u::Vector{Float64},
    t::Float64,
    dt::Float64,
)::Tuple{Vector{Float64},Vector{Float64}}
    k1 = dt .* f(u, t)
    k2 = dt .* f(u .+ 0.25 .* k1, t + 0.25 * dt)
    k3 = dt .* f(u .+ 3 / 32 .* k1 .+ 9 / 32 .* k2, t + 3 / 8 * dt)
    k4 =
        dt .* f(
            u .+ 1932 / 2197 .* k1 .- 7200 / 2197 .* k2 .+ 7296 / 2197 .* k3,
            t + 12 / 13 * dt,
        )
    k5 =
        dt .*
        f(u .+ 439 / 216 .* k1 .- 8 .* k2 .+ 3680 / 513 .* k3 .- 845 / 4104 .* k4, t + dt)
    k6 =
        dt .* f(
            u .- 8 / 27 .* k1 .+ 2 .* k2 .- 3544 / 2565 .* k3 .+ 1859 / 4104 .* k4 .-
            11 / 40 .* k5,
            t + 0.5 * dt,
        )

    u4 = u .+ 25 / 216 .* k1 .+ 1408 / 2565 .* k3 .+ 2197 / 4104 .* k4 .- 1 / 5 .* k5
    u5 =
        u .+ 16 / 135 .* k1 .+ 6656 / 12825 .* k3 .+ 28561 / 56430 .* k4 .- 9 / 50 .* k5 .+
        2 / 55 .* k6

    return u4, u5
end

"""
    solve_pde(D::Matrix{Float64}, x::Vector{Float64}, f::Function, u0::Vector{Float64}, tspan::Vector{Float64}, method::Symbol)::Vector{Float64}

Solve a PDE using the specified method.
"""
function solve_pde(
    D::Matrix{Float64},
    x::Vector{Float64},
    f::Function,
    u0::Vector{Float64},
    tspan::Vector{Float64},
    method::Symbol,
)::Vector{Float64}
    dt = tspan[2] - tspan[1]
    u = copy(u0)

    if method == :rk4
        for t in tspan
            u = rk4_step(f, u, t, dt)
        end
    elseif method == :euler
        for t in tspan
            u = euler_step(f, u, t, dt)
        end
    elseif method == :rk45
        for t in tspan
            u, _ = rk45_step(f, u, t, dt)
        end
    else
        error("Unknown method: $method")
    end

    return u
end

"""
    solve_sde(D::Matrix{Float64}, x::Vector{Float64}, f::Function, g::Function, u0::Vector{Float64}, tspan::Vector{Float64}, dt::Float64)::Matrix{Float64}

Solve a stochastic differential equation using the Euler-Maruyama method.
"""
function solve_sde(
    D::Matrix{Float64},
    x::Vector{Float64},
    f::Function,
    g::Function,
    u0::Vector{Float64},
    tspan::Vector{Float64},
    dt::Float64,
)::Matrix{Float64}
    u = copy(u0)
    t = tspan[1]
    solution = [copy(u0)]

    while t < tspan[end]
        dW = sqrt(dt) .* randn(length(u))  # Generate normally distributed random variables
        u .= u .+ dt .* (D * u + f(x, u, t)) .+ g(x, u, t) .* dW
        t += dt
        push!(solution, copy(u))
    end

    return hcat(solution...)
end

"""
    solve_bvdae_advanced(problem::BVDAEProblem)::Vector{Float64}

Solves a Boundary Value Differential-Algebraic Equation (BVDAE) problem using advanced solvers.

# Arguments
- `problem::BVDAEProblem`: The BVDAE problem to solve.

# Returns
- `sol::Vector{Float64}`: Solution vector.
"""
function solve_bvdae_advanced(problem::BVDAEProblem)::Vector{Float64}
    f, g, bc, tspan, y0 = problem.f, problem.g, problem.bc, problem.tspan, problem.y0
    differential_vars = ones(Bool, length(y0))

    function dae!(res, dy, y, p, t)
        res .= f(y, t) .- dy
        res[end-length(g(y, t))+1:end] .= g(y, t)
    end

    function bc_residual!(res, y, p, t)
        res .= bc(y, t)
    end

    prob = DAEProblem(
        dae!,
        y0,
        zeros(length(y0)),
        tspan,
        differential_vars = differential_vars,
    )
    sol = solve(prob, IDA())

    return sol
end

end # module
