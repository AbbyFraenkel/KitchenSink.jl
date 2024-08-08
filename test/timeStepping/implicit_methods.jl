@testset "Implicit methods" begin
using Test, LinearAlgebra, SparseArrays
using ..KSTypes, ..ProblemTypes, ..TimeStepping, ..SpectralMethods, ..Preprocessing, ..CoordinateSystems

# Helper functions
function mock_solver(residual, u_init)
    u_new = u_init
    for _ in 1:10
        u_new = residual(u_new) + u_new
    end
    return u_new
end

f_linear(u, t; a=1.0, b=0.0) = a * u + b
f_nonlinear(u, t) = u^2 + t

@testset "Implicit Methods" begin
    @testset "backward_euler!" begin
        @testset "Linear function" begin
            u0, t0, dt = 1.0, 0.0, 0.1
            a, b = 1.0, 0.0
            u_expected = (u0 + dt * b) / (1 - dt * a)
            u_computed = backward_euler!([u0], t0, dt, (u, t) -> f_linear(u, t, a=a, b=b), mock_solver)
            @test u_computed[1] ≈ u_expected
        end

        @testset "Nonlinear function" begin
            u0, t0, dt = 1.0, 0.0, 0.1
            u_expected = 1.1 # Approximated value
            u_computed = backward_euler!([u0], t0, dt, f_nonlinear, mock_solver)
            @test u_computed[1] ≈ u_expected atol = 1e-2
        end

        @testset "Stiff ODE" begin
            problem = KSODEProblem((t, y) -> -1000*y + 3000 - 2000*exp(-t), (0.0, 0.1), [0.0])
            t, y = 0.0, [0.0]
            for _ in 1:10
                y = backward_euler!(y, t, 0.01, problem.ode, (F, x) -> x .- F(x))
                t += 0.01
            end
            exact = 3 - 0.998*exp(-1000*0.1) - 2.002*exp(-0.1)
            @test y[1] ≈ exact atol=1e-2
        end
    end

    @testset "implicit_midpoint!" begin
        @testset "Linear function" begin
            u0, t0, dt = 1.0, 0.0, 0.1
            a, b = 1.0, 0.0
            u_expected = (u0 + dt * b) / (1 - dt * a)
            u_computed = implicit_midpoint!([u0], t0, dt, (u, t) -> f_linear(u, t, a=a, b=b), mock_solver)
            @test u_computed[1] ≈ u_expected
        end

        @testset "Nonlinear function" begin
            u0, t0, dt = 1.0, 0.0, 0.1
            u_expected = 1.1 # Approximated value
            u_computed = implicit_midpoint!([u0], t0, dt, f_nonlinear, mock_solver)
            @test u_computed[1] ≈ u_expected atol = 1e-2
        end

        @testset "Simple ODE" begin
            problem = KSODEProblem((t, y) -> -y, (0.0, 1.0), [1.0])
            t, y = 0.0, [1.0]
            for _ in 1:10
                y = implicit_midpoint!(y, t, 0.1, problem.ode, (F, x) -> x .- F(x))
                t += 0.1
            end
            @test y[1] ≈ exp(-1.0) atol=1e-3
        end
    end

    @testset "PDE with Implicit Euler" begin
        pde = KSPDEProblem(
            (u, x, t) -> [sum(∇²(u))],
            (u, x) -> 0.0,
            (0.0, 1.0),
            ((0.0, 1.0),)
        )
        mesh = Preprocessing.create_mesh(pde.domain, (20,), 2, KSCartesianCoordinates{1,Float64}((0.0,)))
        u0 = [sin(π * p.coordinates[1]) for p in mesh.nodes]
        solution = solve_pde(pde, mesh, u0, KSTimeSteppingSolver(:implicit_euler, 0.01, 1.0, 1e-6))
        @test maximum(abs.(solution[end])) < maximum(abs.(u0))
    end

    @testset "DAE with Implicit Euler" begin
        L = 1.0
        dae = KSDAEProblem(
            (t, y, yp) -> [yp[1] - y[2], yp[2] + y[4] * y[1], y[1]^2 + y[3]^2 - L^2],
            (0.0, 1.0),
            [L, 0.0, 0.0, 0.0]
        )
        solution = solve_dae(dae, KSTimeSteppingSolver(:implicit_euler, 0.01, 1.0, 1e-6))
        @test isapprox(solution[end][1]^2 + solution[end][3]^2, L^2, atol=1e-3)
    end
end

end
