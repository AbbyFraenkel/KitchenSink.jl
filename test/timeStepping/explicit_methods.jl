@testset "Explicit methods" begin
using Test, LinearAlgebra, SparseArrays
using ..KSTypes, ..ProblemTypes, ..TimeStepping, ..SpectralMethods, ..Preprocessing, ..CoordinateSystems

# Helper functions
function simple_decay(t, y)
    -y
end

function lotka_volterra(t, y)
    α, β, δ, γ = 1.5, 1.0, 3.0, 1.0
    [α * y[1] - β * y[1] * y[2], δ * y[1] * y[2] - γ * y[2]]
end

function harmonic_oscillator(t, y)
    [y[2], -y[1]]
end

@testset "Explicit Methods" begin
    @testset "Simple Decay ODE" begin
        problem = KSODEProblem(simple_decay, (0.0, 1.0), [1.0])
        exact_solution(t) = exp(-t)

        @testset "Forward Euler" begin
            t, y = 0.0, [1.0]
            dt = 0.001
            for _ in 1:1000
                y = forward_euler!(y, t, dt, problem.ode)
                t += dt
            end
            @test y[1] ≈ exact_solution(1.0) atol = 1e-2
        end

        @testset "RK4" begin
            t, y = 0.0, [1.0]
            dt = 0.1
            for _ in 1:10
                y = rk4!(y, t, dt, problem.ode)
                t += dt
            end
            @test y[1] ≈ exact_solution(1.0) atol = 1e-4
        end

        @testset "RK45" begin
            t, y = 0.0, [1.0]
            dt = 0.1
            y, error_estimate = rk45!(y, t, dt, problem.ode)
            @test y[1] ≈ exact_solution(dt) atol = 1e-5
            @test error_estimate < 1e-5
        end
    end

    @testset "Lotka-Volterra System" begin
        problem = KSODEProblem(lotka_volterra, (0.0, 10.0), [1.0, 1.0])
        solution = solve_ode(problem, KSTimeSteppingSolver(:rk4, 0.01, 10.0, 1e-6))

        @test length(solution) == 2
        @test all(x -> x > 0, solution[end])

        # Check if the solution is oscillatory
        peaks = findlocalmaxima(solution.u[1])
        @test length(peaks) > 1
    end

    @testset "Symplectic Euler for Harmonic Oscillator" begin
        problem = KSODEProblem(harmonic_oscillator, (0.0, 1000.0), [1.0, 0.0])
        solution = solve_ode(problem, KSTimeSteppingSolver(:symplectic_euler, 0.1, 1000.0, 1e-6))

        initial_energy = 0.5 * (problem.initial_conditions[1]^2 + problem.initial_conditions[2]^2)
        final_energy = 0.5 * (solution[end][1]^2 + solution[end][2]^2)
        @test isapprox(initial_energy, final_energy, rtol=1e-3)

        # Check if the solution is periodic
        @test isapprox(solution[1], solution[end], rtol=1e-2)
    end

    @testset "Stability Test" begin
        stiff_problem = KSODEProblem((t, y) -> -50 * y, (0.0, 1.0), [1.0])

        @testset "Forward Euler Instability" begin
            t, y = 0.0, [1.0]
            dt = 0.1
            @test_throws DomainError begin
                for _ in 1:10
                    y = forward_euler!(y, t, dt, stiff_problem.ode)
                    @assert !isnan(y[1]) && abs(y[1]) < 1e10 "Solution blew up"
                    t += dt
                end
            end
        end

        @testset "RK4 Stability" begin
            t, y = 0.0, [1.0]
            dt = 0.1
            for _ in 1:10
                y = rk4!(y, t, dt, stiff_problem.ode)
                t += dt
            end
            @test !isnan(y[1]) && abs(y[1]) < 1.0
        end
    end
end

end
