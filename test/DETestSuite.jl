module ComprehensiveTestSuite

using Test
using ..KSTypes
using ..SpectralMethods
using ..CoordinateSystems
using ..BoundaryConditions
using ..ProblemTypes
using ..LinearSolvers
using ..AnalyticalTestProblems

# ==============================================================================
# Helper Functions and Mesh Creation
# ==============================================================================

function create_simple_1d_mesh()
    cells = [
        KSCell(
        id = 1,
        p = (2,),
        level = 1,
        continuity_order = (1,),
        standard_cell_key = ((2,), 1),
        neighbors = Dict{Symbol, Int}(),
        node_map = Dict{NTuple{1, Int}, Int}((1,) => 1, (2,) => 2, (3,) => 3),
        tensor_product_mask = (true,),
        boundary_connectivity = Dict{Symbol, Int}(),
        error_estimate = 0.0,
        legendre_decay_rate = 0.0,
        is_leaf = true,
        is_fictitious = false,
        refinement_options = nothing,
        parent_id = nothing,
        child_ids = nothing
    )
    ]
    global_node_coords = Dict{Int, NTuple{1, Float64}}(
        1 => (0.0,), 2 => (0.5,), 3 => (1.0,))
    return KSMesh(
        cells = cells,
        global_node_coords = global_node_coords,
        boundary_cells = Dict{Symbol, Vector{Int}}(),
        physical_domain = x -> (0.0 <= x[1] <= 1.0)
    )
end

function create_simple_2d_mesh()
    cells = [
        KSCell(
        id = 1,
        p = (2, 2),
        level = 1,
        continuity_order = (1, 1),
        standard_cell_key = ((2, 2), 1),
        neighbors = Dict{Symbol, Int}(),
        node_map = Dict{NTuple{2, Int}, Int}(
            (1, 1) => 1,
            (1, 2) => 2,
            (1, 3) => 3,
            (2, 1) => 4,
            (2, 2) => 5,
            (2, 3) => 6,
            (3, 1) => 7,
            (3, 2) => 8,
            (3, 3) => 9
        ),
        tensor_product_mask = (true, true),
        boundary_connectivity = Dict{Symbol, Int}(),
        error_estimate = 0.0,
        legendre_decay_rate = 0.0,
        is_leaf = true,
        is_fictitious = false,
        refinement_options = nothing,
        parent_id = nothing,
        child_ids = nothing
    )
    ]
    global_node_coords = Dict{Int, NTuple{2, Float64}}(
        1 => (0.0, 0.0),
        2 => (0.0, 0.5),
        3 => (0.0, 1.0),
        4 => (0.5, 0.0),
        5 => (0.5, 0.5),
        6 => (0.5, 1.0),
        7 => (1.0, 0.0),
        8 => (1.0, 0.5),
        9 => (1.0, 1.0)
    )
    return KSMesh(
        cells = cells,
        global_node_coords = global_node_coords,
        boundary_cells = Dict{Symbol, Vector{Int}}(),
        physical_domain = x -> (0.0 <= x[1] <= 1.0) && (0.0 <= x[2] <= 1.0)
    )
end

# ==============================================================================
# Test for Moving Boundary PDE (Stefan Problem)
# ==============================================================================
function test_stefan_problem()
    println("Testing Moving Boundary PDE (Stefan Problem)")

    α, T_0, T_s = 0.1, 0.0, 1.0
    s_analytical(t) = 2 * sqrt(α * t)
    u_analytical(x, t) = T_0 + (T_s - T_0) * (x / s_analytical(t))

    stefan_problem, _ = AnalyticalTestProblems.create_stefan_problem()

    # Solve the Stefan problem numerically
    u_numerical, s_numerical = moving_boundary_solver(stefan_problem)

    for i in 1:length(u_numerical)
        @test isapprox(
            u_numerical[i],
            u_analytical(mesh[i], stefan_problem.tspan[2]),
            atol = 1e-2
        )
    end
    @test isapprox(s_numerical[end], s_analytical(stefan_problem.tspan[2]), atol = 1e-2)
end

# ==============================================================================
# Test for Coupled Heat and Wave Equation
# ==============================================================================
function test_coupled_heat_wave()
    println("Testing Coupled Heat and Wave Equation")

    α, c = 0.1, 1.0
    function analytical_solution_u(x, y, t)
        exp(-2 * α * π^2 * t) * sin(π * x) * sin(π * y) +
        cos(c * π * t) * sin(π * x) * sin(π * y)
    end
    analytical_solution_v(x, y, t) = cos(c * π * t) * sin(π * x) * sin(π * y)

    coupled_problem, _ = AnalyticalTestProblems.create_coupled_heat_wave()

    # Solve the coupled system numerically
    u_numerical, v_numerical = coupled_numerical_solver(coupled_problem)

    for i in 1:length(u_numerical)
        @test isapprox(
            u_numerical[i],
            analytical_solution_u(mesh[i]..., coupled_problem.tspan[2]),
            atol = 1e-2
        )
        @test isapprox(
            v_numerical[i],
            analytical_solution_v(mesh[i]..., coupled_problem.tspan[2]),
            atol = 1e-2
        )
    end
end

# ==============================================================================
# Test for Coupled Mechanical DAE and Diffusion PDE
# ==============================================================================
function test_coupled_dae_diffusion()
    println("Testing Coupled Mechanical DAE and Diffusion PDE")

    γ, k = 0.1, 1.0
    ω = sqrt(k - γ^2 / 4)
    analytical_solution_x(t) = exp(-γ * t / 2) * cos(ω * t)
    D = 0.1
    analytical_solution_u(x, t) = sin(π * x) * exp(-D * π^2 * t)

    coupled_problem, _ = AnalyticalTestProblems.create_coupled_dae_diffusion()

    # Solve the coupled system numerically
    x_numerical, u_numerical = coupled_dae_pde_solver(coupled_problem)

    @test isapprox(
        x_numerical[end],
        analytical_solution_x(coupled_problem.tspan[2]),
        atol = 1e-2
    )
    for i in 1:length(u_numerical)
        @test isapprox(
            u_numerical[i],
            analytical_solution_u(mesh[i], coupled_problem.tspan[2]),
            atol = 1e-2
        )
    end
end

# ==============================================================================
# Test for Black-Scholes PIDE with Integral Term
# ==============================================================================
function test_black_scholes_pide()
    println("Testing Black-Scholes PIDE")

    pide_problem, _ = AnalyticalTestProblems.create_black_scholes_pide()

    # Solve the PIDE numerically
    u_numerical = pide_numerical_solver(pide_problem)

    # Placeholder test for qualitative behavior
    @test norm(u_numerical) < 1e6  # Ensure solution doesn't diverge
end

# ==============================================================================
# Test for Van der Pol Oscillator (Nonlinear ODE)
# ==============================================================================
function test_van_der_pol_oscillator()
    println("Testing Van der Pol Oscillator")

    μ = 0.1
    analytical_solution_x(t) = 2 * cos(t)  # Approximation for small μ

    van_der_pol_problem, _ = AnalyticalTestProblems.create_van_der_pol()

    # Solve numerically
    x_numerical = van_der_pol_solver(van_der_pol_problem)

    @test isapprox(
        x_numerical[end],
        analytical_solution_x(van_der_pol_problem.tspan[2]),
        atol = 1e-2
    )
end

# ==============================================================================
# Test for First-Order Linear ODE
# ==============================================================================
function test_first_order_linear_ode()
    println("Testing First-Order Linear ODE")

    analytical_solution(x) = exp(-x)

    ode_problem, _ = AnalyticalTestProblems.create_first_order_linear_ode()

    # Solve the ODE numerically using Forward Euler or similar method
    u_numerical = ode_numerical_solver(ode_problem)

    # Compare numerical and analytical solution
    @test isapprox(u_numerical[end], analytical_solution(ode_problem.tspan[2]), atol = 1e-2)
end

# ==============================================================================
# Test for Nonlinear ODE (y' = y^2)
# ==============================================================================
function test_nonlinear_ode()
    println("Testing Nonlinear ODE (y' = y^2)")

    analytical_solution(x) = 1 / (1 - x)

    ode_problem, _ = AnalyticalTestProblems.create_nonlinear_ode()

    # Solve the ODE numerically
    u_numerical = ode_numerical_solver(ode_problem)

    # Compare numerical and analytical solution
    @test isapprox(u_numerical[end], analytical_solution(ode_problem.tspan[2]), atol = 1e-2)
end

# ==============================================================================
# Test for Burgers' Equation
# ==============================================================================
function test_burgers_equation()
    println("Testing Burgers' Equation")

    burgers_problem, _ = AnalyticalTestProblems.create_burgers_equation()

    # Solve numerically (use Cole-Hopf transformation or other methods)
    u_numerical = burgers_numerical_solver(burgers_problem)

    # Placeholder test to check qualitative behavior
    @test norm(u_numerical) < 1e3  # Ensure solution doesn't diverge
end

# ==============================================================================
# Test for Robertson Problem (Stiff ODE)
# ==============================================================================
function test_robertson_problem()
    println("Testing Robertson Problem (Stiff ODE)")

    analytical_y1(t) = 1 - 0.04 * t
    analytical_y2(t) = t / (2.5 * 10^4)
    analytical_y3(t) = t / (1.5 * 10^7)

    robertson_problem, _ = AnalyticalTestProblems.create_robertson_problem()

    # Solve stiff ODE numerically
    y1_numerical, y2_numerical, y3_numerical = stiff_ode_solver(robertson_problem)

    # Compare numerical and analytical solutions
    @test isapprox(
        y1_numerical[end],
        analytical_y1(robertson_problem.tspan[2]),
        atol = 1e-2
    )
    @test isapprox(
        y2_numerical[end],
        analytical_y2(robertson_problem.tspan[2]),
        atol = 1e-2
    )
    @test isapprox(
        y3_numerical[end],
        analytical_y3(robertson_problem.tspan[2]),
        atol = 1e-2
    )
end

# ==============================================================================
# Test for Integro-Differential Equation (Volterra IDE)
# ==============================================================================
function test_volterra_ide()
    println("Testing Volterra Integro-Differential Equation")

    # Analytical solution: u(t) = e^(-t)
    analytical_solution(t) = exp(-t)

    ide_problem, _ = AnalyticalTestProblems.create_volterra_ide()

    # Solve the IDE numerically
    u_numerical = ide_numerical_solver(ide_problem)

    # Compare numerical and analytical solution
    @test isapprox(u_numerical[end], analytical_solution(ide_problem.tspan[2]), atol = 1e-2)
end

# ==============================================================================
# Test for Delay Differential Equation (DDE)
# ==============================================================================
function test_dde()
    println("Testing Delay Differential Equation")

    a, b = 1.0, 0.5
    analytical_solution(t) = exp((b - a) * t)

    dde_problem, _ = AnalyticalTestProblems.create_dde()

    # Solve the DDE numerically
    u_numerical = dde_numerical_solver(dde_problem)

    # Compare numerical and analytical solution
    @test isapprox(u_numerical[end], analytical_solution(dde_problem.tspan[2]), atol = 1e-2)
end

# ==============================================================================
# Test for Boundary Value DAE (BVDAE)
# ==============================================================================
function test_bvdae()
    println("Testing Boundary Value DAE")

    analytical_solution(t) = t * (t - 1) / 2

    bvdae_problem, _ = AnalyticalTestProblems.create_bvdae()

    # Solve the BVDAE numerically
    u_numerical = bvdae_numerical_solver(bvdae_problem)

    # Compare numerical and analytical solution
    @test isapprox(
        u_numerical[end],
        analytical_solution(bvdae_problem.tspan[2]),
        atol = 1e-2
    )
end

# ==============================================================================
# Run all tests
# ==============================================================================
f

end # module ComprehensiveTestSuite
