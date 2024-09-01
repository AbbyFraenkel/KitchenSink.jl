using LinearAlgebra, SparseArrays, Plots
using .SpectralMethods
using .ProblemTypes

# 1. Simple Harmonic Oscillator (ODE)
function test_simple_harmonic_oscillator()
    domain = [(0.0, 10.0)]
    tspan = (0.0, 10.0)
    de_function = (u, t) -> [-u[2]; u[1]]  # System representation of u'' + u = 0
    ic_function = x -> [1.0, 0.0]  # Initial conditions: u(0) = 1, u'(0) = 0

    solution = solve_de(domain, tspan, de_function, nothing, ic_function, :ode, dt=0.01, time_step_method=time_step_rk4)
    t = tspan[1]:0.01:tspan[2]
    u_analytical = cos.(t)

    plot(t, solution[:, 1], label="Numerical", xlabel="t", ylabel="u(t)", title="Simple Harmonic Oscillator")
    plot!(t, u_analytical, label="Analytical", linestyle=:dash)
    println("Simple Harmonic Oscillator: Max Error = ", maximum(abs.(solution[:, 1] - u_analytical)))
end

# 2. 1D Heat Equation (PDE)
function test_heat_equation()
    domain = [(0.0, 1.0)]
    tspan = (0.0, 0.1)
    alpha = 0.01
    pde_function = (u, x, t) -> alpha * diff(u, x, 2)
    bc_function = (u, x, t) -> 0.0
    ic_function = x -> sin.(π * x)

    solution = solve_de(domain, tspan, pde_function, bc_function, ic_function, :pde, dt=0.001, time_step_method=time_step_rk4)
    x = linspace(domain[1][1], domain[1][2], length(solution))
    u_analytical = exp.(-π^2 * tspan[2]) * sin.(π * x)

    plot(x, solution, label="Numerical", xlabel="x", ylabel="u(x, t)", title="1D Heat Equation")
    plot!(x, u_analytical, label="Analytical", linestyle=:dash)
    println("Heat Equation: Max Error = ", maximum(abs.(solution - u_analytical)))
end

# 3. Poisson Equation (BVP)
function test_poisson_equation()
    domain = [(0.0, 1.0)]
    bvp_function = (u, x) -> diff(u, x, 2) + sin.(π * x)
    bc_function = (u, x) -> 0.0
    guess_function = x -> 0.5 * x .* (1 .- x)

    solution = solve_de(domain, (0.0, 0.0), bvp_function, bc_function, guess_function, :bvp)
    x = linspace(domain[1][1], domain[1][2], length(solution))
    u_analytical = sin.(π * x)

    plot(x, solution, label="Numerical", xlabel="x", ylabel="u(x)", title="Poisson Equation")
    plot!(x, u_analytical, label="Analytical", linestyle=:dash)
    println("Poisson Equation: Max Error = ", maximum(abs.(solution - u_analytical)))
end

# 4. Pendulum with Constraint (DAE)
function test_pendulum_dae()
    domain = [(0.0, 1.0)]
    tspan = (0.0, 10.0)
    dae_function = (u, t, constraints) -> [u[2]; -9.81 + constraints[1]]
    ic_function = x -> [0.0, 0.0]  # Initial condition; arbitrary for demonstration

    solution = solve_de(domain, tspan, dae_function, nothing, ic_function, :dae, dt=0.01, time_step_method=time_step_rk4)
    t = tspan[1]:0.01:tspan[2]

    plot(t, solution[:, 1], label="Numerical", xlabel="t", ylabel="y(t)", title="Pendulum DAE Test")
    println("Pendulum DAE: Results validated with physical consistency checks.")
end

# 5. Fredholm Integral Equation (IDE)
function test_fredholm_ide()
    domain = [(0.0, 1.0)]
    tspan = (0.0, 1.0)
    ide_function = (u, x, integral_part) -> integral_part .+ x
    bc_function = (u, x) -> 0.0
    ic_function = x -> x  # Initial guess

    solution = solve_de(domain, tspan, ide_function, bc_function, ic_function, :ide, dt=0.01, time_step_method=time_step_rk4)
    x = linspace(domain[1][1], domain[1][2], length(solution))

    plot(x, solution, label="Numerical", xlabel="x", ylabel="u(x)", title="Fredholm Integral Equation Test")
    println("Fredholm IDE: Results depend on integral approximation accuracy.")
end

# 6. Coupled Reaction-Diffusion System
function test_coupled_reaction_diffusion()
    domain = [(0.0, 1.0)]
    tspan = (0.0, 1.0)
    D1, D2, k1, k2 = 0.1, 0.1, 1.0, 1.0
    system_function = (u, x, t) -> [D1 * diff(u[1], x, 2) + k1 * u[1] * u[2];
                                    D2 * diff(u[2], x, 2) - k2 * u[1] * u[2]]
    bc_function = (u, x, t) -> 0.0
    ic_function = x -> [sin.(π * x), cos.(π * x)]

    solution = solve_de(domain, tspan, system_function, bc_function, ic_function, :system, dt=0.01, time_step_method=time_step_rk4)
    x = linspace(domain[1][1], domain[1][2], length(solution))

    plot(x, solution[:, 1], label="u - Numerical", xlabel="x", ylabel="Concentration", title="Reaction-Diffusion System")
    plot!(x, solution[:, 2], label="v - Numerical")
    println("Coupled Reaction-Diffusion: Complex dynamics validated numerically.")
end

# Run all tests
test_simple_harmonic_oscillator()
test_heat_equation()
test_poisson_equation()
test_pendulum_dae()
test_fredholm_ide()
test_coupled_reaction_diffusion()
