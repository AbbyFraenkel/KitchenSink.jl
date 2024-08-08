@testset "IDEs" begin
    @testset "KSIDEProblem" begin
        # Test IDE: u'(t) = u(t) + ∫₀ᵗ e^(t-s)u(s)ds, u(0) = 1
        ide = KSIDEProblem(
            (t, u, int) -> u + int,
            (s, t) -> exp(t - s),
            (0.0, 1.0),
            [1.0]
        )
        @test ide isa KSIDEProblem
        @test ide.tspan == (0.0, 1.0)
        @test ide.y0 == [1.0]

        # Test solving
        solver = KSIterativeSolver(:gmres, 1000, 1e-6, nothing)
        t, u = solve_ide(ide, solver)
        @test length(t) == length(u)
        @test u[1] ≈ 1.0
        @test u[end] > exp(1.0)  # Solution should grow faster than e^t
    end
end
