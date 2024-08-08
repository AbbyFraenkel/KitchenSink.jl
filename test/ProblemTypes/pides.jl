pide_problem = KSPIDEProblem{Float64,1}(
    (D, u, nodes) -> D[1]^2 * u,  # Spatial operator (∂²u/∂x²)
    (u, nodes, solution, mesh) -> [sum(u .* mesh.weights) for _ in 1:length(u)],  # Integral operator
    (u, nodes) -> u,  # Time derivative (∂u/∂t)
    (nodes) -> zeros(size(nodes, 1)),  # Source term
    (nodes) -> [0.0, 0.0]  # Boundary conditions
)

analytical_pide(x, t) = sin(π*x) * exp(t * (1 - π^2))

@testset "PIDEs" begin
    @testset "KSPIDEProblem" begin
        @test pide_problem isa KSPIDE{Float64,1}

        mesh = create_mesh([0, 1], 50, 3)
        t_final = 1.0
        dt = 0.01
        numerical_solution = solve_equation(pide_problem, mesh, t_final, dt)

        # Check boundary conditions
        @test all(isapprox.(numerical_solution[1, :], 0.0, atol=1e-6))
        @test all(isapprox.(numerical_solution[end, :], 0.0, atol=1e-6))

        # Check solution at final time
        for (i, x) in enumerate(mesh.nodes)
            @test isapprox(numerical_solution[i, end], analytical_pide(x, t_final), rtol=1e-2)
        end

        # Check overall error
        max_error = maximum(abs.(numerical_solution[:, end] .- analytical_pide.(mesh.nodes, t_final)))
        @test max_error < 1e-2
    end
end
