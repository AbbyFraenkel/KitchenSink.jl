@testset "Coupled Problems" begin
    @testset "KSCoupledProblem" begin
        # Test coupled ODE-PDE problem
        ode = KSODEProblem((t, y) -> -y, (0.0, 1.0), [1.0])
        pde = KSPDEProblem(
            (u, x, t) -> sum(∇²(u)),
            (u, x) -> 0.0,
            (0.0, 1.0),
            ((0.0, 1.0), (0.0, 1.0))
        )
        coupling = (ode_sol, pde_sol) -> ode_sol[end] * pde_sol
        coupled_problem = KSCoupledProblem([ode, pde], [nothing coupling; nothing nothing])

        @test coupled_problem isa KSCoupledProblem
        @test length(coupled_problem.problems) == 2
        @test size(coupled_problem.coupling_terms) == (2, 2)

        # Test solving (simplified)
        mesh = Preprocessing.create_mesh(pde.domain, (10, 10), 2, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))
        ode_solver = KSIterativeSolver(:cg, 1000, 1e-6, nothing)
        pde_solver = KSDirectSolver(:lu)
        t, y = solve_coupled_problem(coupled_problem, mesh, [ode_solver, pde_solver])
        @test length(t) > 0
        @test all(length.(y) .> 0)
    end
end
