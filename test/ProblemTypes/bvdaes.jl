L = 1.0  # Pendulum length
dae_problem = KSDAEProblem(
    (D, u, nodes) -> [u[2, :]; -u[3, :] .* u[1, :]],  # Differential equations
    (u, nodes) -> u[1, :] .^ 2 + u[2, :] .^ 2 .- L^2,  # Algebraic constraint
    (nodes) -> [L * ones(size(nodes, 1)); zeros(size(nodes, 1)); zeros(size(nodes, 1))]  # Initial conditions
)

analytical_dae(t) = [L * cos(t), L * sin(t), -L * sin(t)]


a = 0.25  # Parameter
bvdae_problem = KSBVDAEProblem(
    (D, u, nodes) -> D[1]^2 * u,  # d²y/dx²
    (u, nodes) -> (1 .+ (D[1] * u) .^ 2) .^ (3 / 2) / a,  # Right-hand side
    (nodes) -> [0.0, 0.0]  # Boundary conditions
)

analytical_bvdae(x) = a * cosh((x - 0.5) / a) - a * cosh(0.5 / a)

@testset "BVDAEs" begin
    @testset "DAE Problem" begin
        mesh = create_mesh([0, 1], 100, 3)
        numerical_solution = solve_equation(dae_problem, mesh)

        # Check solution at specific times
        for t in [0.0, 0.5, 1.0]
            node_index = findfirst(x -> isapprox(x, t, atol=1e-2), mesh.nodes)
            @test all(isapprox.(numerical_solution[:, node_index], analytical_dae(t), rtol=1e-2))
        end

        # Check constraint satisfaction
        @test all(isapprox.(numerical_solution[1, :].^2 + numerical_solution[2, :].^2, L^2, rtol=1e-2))
    end

    @testset "BVDAE Problem" begin
        mesh = create_mesh([0, 1], 100, 3)
        numerical_solution = solve_equation(bvdae_problem, mesh)

        # Check boundary conditions
        @test isapprox(numerical_solution[1], 0.0, atol=1e-6)
        @test isapprox(numerical_solution[end], 0.0, atol=1e-6)

        # Check solution at midpoint
        midpoint_index = div(length(mesh.nodes), 2)
        @test isapprox(numerical_solution[midpoint_index], analytical_bvdae(0.5), rtol=1e-2)

        # Check overall error
        max_error = maximum(abs.(numerical_solution .- analytical_bvdae.(mesh.nodes)))
        @test max_error < 1e-2
    end
end
