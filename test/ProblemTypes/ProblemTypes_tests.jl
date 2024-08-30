using Test
using LinearAlgebra, SparseArrays
using KitchenSink.ProblemTypes

@testset "ProblemTypes Tests" begin

    @testset "PDEs" begin
        @testset "Valid Cases" begin
            @testset "1D Heat Equation" begin
                domain = ((0.0, 1.0),)
                tspan = (0.0, 1.0)
                pde = (u, x, t) -> ∇²(u)[1]
                bc = (u, x, t) -> 0.0
                ic = (x) -> sin(π * x[1])

                problem = KSPDEProblem(pde, bc, tspan, domain, ic)
                @test problem isa KSPDEProblem{Float64,1}
                @test problem.domain == domain
                @test problem.tspan == tspan

                mesh = create_mesh(domain, (10,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))
                A, b = create_pde_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "2D Poisson Equation" begin
                domain = ((0.0, 1.0), (0.0, 1.0))
                tspan = (0.0, 1.0)
                pde = (u, x, t) -> -sum(∇²(u))
                bc = (u, x, t) -> 0.0
                ic = (x) -> 0.0

                problem = KSPDEProblem(pde, bc, tspan, domain, ic)
                @test problem isa KSPDEProblem{Float64,2}
                @test problem.domain == domain
                @test problem.tspan == tspan

                mesh = create_mesh(domain, (10, 10), 3, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))
                A, b = create_pde_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "3D Heat Equation" begin
                domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
                tspan = (0.0, 1.0)
                pde = (u, x, t) -> sum(∇²(u))
                bc = (u, x, t) -> 0.0
                ic = (x) -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])

                problem = KSPDEProblem(pde, bc, tspan, domain, ic)
                @test problem isa KSPDEProblem{Float64,3}
                @test problem.domain == domain
                @test problem.tspan == tspan

                mesh = create_mesh(domain, (5, 5, 5), 2, KSCartesianCoordinates{3,Float64}((0.0, 0.0, 0.0)))
                A, b = create_pde_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid PDE Function" begin
                @test_throws MethodError KSPDEProblem(
                    "not a function",
                    (u, x, t) -> 0.0,
                    (0.0, 1.0),
                    ((0.0, 1.0),),
                    (x) -> 0.0
                )
            end

            @testset "Mismatched Dimensions" begin
                @test_throws DimensionMismatch KSPDEProblem(
                    (u, x, t) -> sum(∇²(u)),
                    (u, x, t) -> 0.0,
                    (0.0, 1.0),
                    ((0.0, 1.0), (0.0, 1.0)),  # 2D domain
                    (x) -> 0.0  # 1D initial condition
                )
            end
        end

        @testset "Edge Cases" begin
            @testset "Zero Time Span" begin
                @test_throws ArgumentError KSPDEProblem(
                    (u, x, t) -> ∇²(u)[1],
                    (u, x, t) -> 0.0,
                    (0.0, 0.0),  # Zero time span
                    ((0.0, 1.0),),
                    (x) -> 0.0
                )
            end

            @testset "Empty Domain" begin
                @test_throws ArgumentError KSPDEProblem(
                    (u, x, t) -> ∇²(u)[1],
                    (u, x, t) -> 0.0,
                    (0.0, 1.0),
                    (),  # Empty domain
                    (x) -> 0.0
                )
            end

            @testset "Single Point Domain" begin
                @test_throws ArgumentError KSPDEProblem(
                    (u, x, t) -> ∇²(u)[1],
                    (u, x, t) -> 0.0,
                    (0.0, 1.0),
                    ((0.0, 0.0),),  # Single point domain
                    (x) -> 0.0
                )
            end
        end

        @testset "Complex Cases" begin
            @testset "Nonlinear PDE" begin
                domain = ((0.0, 1.0),)
                tspan = (0.0, 1.0)
                pde = (u, x, t) -> u * ∇²(u)[1] + u^2
                bc = (u, x, t) -> 0.0
                ic = (x) -> sin(π * x[1])

                problem = KSPDEProblem(pde, bc, tspan, domain, ic)
                mesh = create_mesh(domain, (20,), 4, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_pde_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "Mixed Boundary Conditions" begin
                domain = ((0.0, 1.0), (0.0, 1.0))
                tspan = (0.0, 1.0)
                pde = (u, x, t) -> sum(∇²(u))
                bc_dirichlet = KSDirichletBC((u, x) -> 0.0, (x) -> x[1] == 0.0 || x[1] == 1.0)
                bc_neumann = KSNeumannBC((u, x) -> 1.0, (x) -> x[2] == 0.0)
                bc_robin = KSRobinBC((u, x) -> 1.0, (u, x) -> 2.0, (u, x) -> 3.0, (x) -> x[2] == 1.0)
                ic = (x) -> 0.0

                problem = KSPDEProblem(pde, [bc_dirichlet, bc_neumann, bc_robin], tspan, domain, ic)
                mesh = create_mesh(domain, (10, 10), 3, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))

                A, b = create_pde_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "PDE with Polar Coordinates" begin
                domain = ((0.0, 1.0), (0.0, 2π))
                tspan = (0.0, 1.0)
                pde = (u, r, theta, t) -> 1 / r * ∂r(r * ∂r(u)) + 1 / r^2 * ∂θ²(u)
                bc = (u, r, theta, t) -> 0.0
                ic = (r, theta) -> r * sin(θ)

                problem = KSPDEProblem(pde, bc, tspan, domain, ic)
                mesh = create_mesh(domain, (10, 20), 3, KSPolarCoordinates{Float64}(1.0, 0.0))

                A, b = create_pde_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end
        end
    end

    @testset "ODEs" begin
        @testset "Valid Cases" begin
            @testset "Simple ODE" begin
                tspan = (0.0, 1.0)
                ode = ( y) -> -y
                ic = [1.0]

                problem = KSODEProblem(ode, tspan, ic)
                @test problem isa KSODEProblem{Float64,1}
                @test problem.tspan == tspan
                @test problem.initial_conditions == ic

                t_mesh = create_mesh(tspan, (10,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))
                A, b = create_ode_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == length(t_mesh.nodes)
                @test length(b) == length(t_mesh.nodes)
            end

            @testset "System of ODEs" begin
                tspan = (0.0, 1.0)
                ode = ( y) -> [-y[1], y[1] - y[2]]
                ic = [1.0, 0.0]

                problem = KSODEProblem(ode, tspan, ic)
                @test problem isa KSODEProblem{Float64,1}
                @test problem.tspan == tspan
                @test problem.initial_conditions == ic

                t_mesh = create_mesh(tspan, (10,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))
                A, b = create_ode_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == 2 * length(t_mesh.nodes)
                @test length(b) == 2 * length(t_mesh.nodes)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid ODE Function" begin
                @test_throws MethodError KSODEProblem(
                    "not a function",
                    (0.0, 1.0),
                    [1.0]
                )
            end

            @testset "Mismatched Dimensions" begin
                @test_throws DimensionMismatch KSODEProblem(
                    ( y) -> [-y[1], -y[2]],
                    (0.0, 1.0),
                    [1.0]  # 1D initial condition for 2D ODE
                )
            end
        end

        @testset "Edge Cases" begin
            @testset "Zero Time Span" begin
                @test_throws ArgumentError KSODEProblem(
                    ( y) -> -y,
                    (0.0, 0.0),  # Zero time span
                    [1.0]
                )
            end

            @testset "Empty Initial Condition" begin
                @test_throws ArgumentError KSODEProblem(
                    ( y) -> -y,
                    (0.0, 1.0),
                    Float64[]  # Empty initial condition
                )
            end

            @testset "Infinite Time Span" begin
                @test_throws ArgumentError KSODEProblem(
                    ( y) -> -y,
                    (0.0, Inf),  # Infinite time span
                    [1.0]
                )
            end
        end

        @testset "Complex Cases" begin
            @testset "Stiff ODE System" begin
                tspan = (0.0, 1000.0)
                function stiff_ode( y)
                    [
                        -0.04 * y[1] + 1e4 * y[2] * y[3],
                        0.04 * y[1] - 1e4 * y[2] * y[3] - 3e7 * y[2]^2,
                        3e7 * y[2]^2
                    ]
                end
                ic = [1.0, 0.0, 0.0]

                problem = KSODEProblem(stiff_ode, tspan, ic)
                t_mesh = create_mesh(tspan, (1000,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_ode_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == 3 * length(t_mesh.nodes)
                @test length(b) == 3 * length(t_mesh.nodes)
            end

            @testset "Non-autonomous ODE" begin
                tspan = (0.0, 10.0)
                ode = ( y) -> sin(t) * y
                ic = [1.0]

                problem = KSODEProblem(ode, tspan, ic)
                t_mesh = create_mesh(tspan, (100,), 4, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_ode_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == length(t_mesh.nodes)
                @test length(b) == length(t_mesh.nodes)
            end
        end
    end

    @testset "DAEs" begin
        @testset "Valid Cases" begin
            @testset "Index-1 DAE" begin
                tspan = (0.0, 1.0)
                dae = ( y, y′) -> [y′[1] + y[2], y[1] + y[2] - 1]
                ic = [1.0, 0.0]

                problem = KSDAEProblem(dae, tspan, ic)
                @test problem isa KSDAEProblem{Float64,1}
                @test problem.tspan == tspan
                @test problem.initial_conditions == ic

                t_mesh = create_mesh(tspan, (10,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))
                A, b = create_dae_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == 2 * length(t_mesh.nodes)
                @test length(b) == 2 * length(t_mesh.nodes)
            end

            @testset "Semi-explicit DAE" begin
                tspan = (0.0, 1.0)
                dae = ( y, y′) -> [y′[1] - y[2], y[1]^2 + y[2] - t]
                ic = [0.0, 0.0]

                problem = KSDAEProblem(dae, tspan, ic)
                @test problem isa KSDAEProblem{Float64,1}
                @test problem.tspan == tspan
                @test problem.initial_conditions == ic

                t_mesh = create_mesh(tspan, (10,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))
                A, b = create_dae_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == 2 * length(t_mesh.nodes)
                @test length(b) == 2 * length(t_mesh.nodes)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid DAE Function" begin
                @test_throws MethodError KSDAEProblem(
                    "not a function",
                    (0.0, 1.0),
                    [1.0, 0.0]
                )
            end

            @testset "Mismatched Dimensions" begin
                @test_throws DimensionMismatch KSDAEProblem(
                    ( y, y′) -> [y′[1] + y[2], y[1] + y[2] - 1],
                    (0.0, 1.0),
                    [1.0]  # 1D initial condition for 2D DAE
                )
            end
        end

        @testset "Edge Cases" begin
            @testset "Zero Time Span" begin
                @test_throws ArgumentError KSDAEProblem(
                    ( y, y′) -> [y′[1] + y[2], y[1] + y[2] - 1],
                    (0.0, 0.0),  # Zero time span
                    [1.0, 0.0]
                )
            end

            @testset "Empty Initial Condition" begin
                @test_throws ArgumentError KSDAEProblem(
                    ( y, y′) -> [y′[1] + y[2], y[1] + y[2] - 1],
                    (0.0, 1.0),
                    Float64[]  # Empty initial condition
                )
            end

            @testset "Infinite Time Span" begin
                @test_throws ArgumentError KSDAEProblem(
                    ( y, y′) -> [y′[1] + y[2], y[1] + y[2] - 1],
                    (0.0, Inf),  # Infinite time span
                    [1.0, 0.0]
                )
            end
        end

        @testset "Complex Cases" begin
            @testset "Higher Index DAE" begin
                tspan = (0.0, 1.0)
                dae = ( y, y′) -> [y′[1] - y[2], y[1]^2 + y[2]^2 - 1]
                ic = [1.0, 0.0]

                problem = KSDAEProblem(dae, tspan, ic)
                t_mesh = create_mesh(tspan, (20,), 4, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_dae_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == 2 * length(t_mesh.nodes)
                @test length(b) == 2 * length(t_mesh.nodes)
            end

            @testset "Non-autonomous DAE" begin
                tspan = (0.0, 10.0)
                dae = ( y, y′) -> [y′[1] - y[2], y[1] + y[2] - sin(t)]
                ic = [0.0, 0.0]

                problem = KSDAEProblem(dae, tspan, ic)
                t_mesh = create_mesh(tspan, (100,), 4, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_dae_system_matrix_and_vector(problem, t_mesh)
                @test size(A, 1) == size(A, 2) == 2 * length(t_mesh.nodes)
                @test length(b) == 2 * length(t_mesh.nodes)
            end
        end
    end

    @testset "BVPs" begin
        @testset "Valid Cases" begin
            @testset "Linear 2-point BVP" begin
                domain = ((0.0, 1.0),)
                bvp = (u, x) -> -∇²(u)[1]
                bc = KSDirichletBC((u, x) -> 0.0, (x) -> x[1] == 0.0 || x[1] == 1.0)

                problem = KSBVPProblem(bvp, bc, domain)
                @test problem isa KSBVPProblem{Float64,1}
                @test problem.domain == domain

                mesh = create_mesh(domain, (10,), 3, KSCartesianCoordinates{1,Float64}((0.0,)))
                A, b = create_bvp_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "2D BVP" begin
                domain = ((0.0, 1.0), (0.0, 1.0))
                bvp = (u, x) -> -sum(∇²(u))
                bc = KSDirichletBC((u, x) -> 0.0, (x) -> x[1] == 0.0 || x[1] == 1.0 || x[2] == 0.0 || x[2] == 1.0)

                problem = KSBVPProblem(bvp, bc, domain)
                @test problem isa KSBVPProblem{Float64,2}
                @test problem.domain == domain

                mesh = create_mesh(domain, (10, 10), 3, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))
                A, b = create_bvp_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end
        end

        @testset "Invalid Cases" begin
            @testset "Invalid BVP Function" begin
                @test_throws MethodError KSBVPProblem(
                    "not a function",
                    KSDirichletBC((u, x) -> 0.0, (x) -> true),
                    ((0.0, 1.0),)
                )
            end

            @testset "Mismatched Dimensions" begin
                @test_throws DimensionMismatch KSBVPProblem(
                    (u, x) -> -sum(∇²(u)),
                    KSDirichletBC((u, x) -> 0.0, (x) -> true),
                    ((0.0, 1.0),)  # 1D domain for 2D BVP
                )
            end
        end

        @testset "Edge Cases" begin
            @testset "Single Point Domain" begin
                @test_throws ArgumentError KSBVPProblem(
                    (u, x) -> -∇²(u)[1],
                    KSDirichletBC((u, x) -> 0.0, (x) -> true),
                    ((0.0, 0.0),)  # Single point domain
                )
            end

            @testset "Empty Domain" begin
                @test_throws ArgumentError KSBVPProblem(
                    (u, x) -> -∇²(u)[1],
                    KSDirichletBC((u, x) -> 0.0, (x) -> true),
                    ()  # Empty domain
                )
            end
        end

        @testset "Complex Cases" begin
            @testset "Nonlinear BVP" begin
                domain = ((0.0, 1.0),)
                bvp = (u, x) -> -∇²(u)[1] + u^2
                bc = KSDirichletBC((u, x) -> 0.0, (x) -> x[1] == 0.0 || x[1] == 1.0)

                problem = KSBVPProblem(bvp, bc, domain)
                mesh = create_mesh(domain, (20,), 4, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_bvp_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "BVP with Mixed Boundary Conditions" begin
                domain = ((0.0, 1.0),)
                bvp = (u, x) -> -∇²(u)[1]
                bc_left = KSDirichletBC((u, x) -> 0.0, (x) -> x[1] == 0.0)
                bc_right = KSNeumannBC((u, x) -> 1.0, (x) -> x[1] == 1.0)

                problem = KSBVPProblem(bvp, [bc_left, bc_right], domain)
                mesh = create_mesh(domain, (20,), 4, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_bvp_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end

            @testset "BVP with Internal Layer" begin
                domain = ((0.0, 1.0),)
                ε = 1e-3
                bvp = (u, x) -> -ε * ∇²(u)[1] + u
                bc = KSDirichletBC((u, x) -> 0.0, (x) -> x[1] == 0.0 || x[1] == 1.0)

                problem = KSBVPProblem(bvp, bc, domain)
                mesh = create_mesh(domain, (100,), 5, KSCartesianCoordinates{1,Float64}((0.0,)))

                A, b = create_bvp_system_matrix_and_vector(problem, mesh)
                @test size(A, 1) == size(A, 2) == length(mesh.nodes)
                @test length(b) == length(mesh.nodes)
                @test issparse(A)
            end
        end
    end
end


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

@testset "Coupled Problems" begin
	@testset "KSCoupledProblem" begin
		# Test coupled ODE-PDE problem
		ode = KSODEProblem((y) -> -y, (0.0, 1.0), [1.0])
		pde = KSPDEProblem((u, x, t) -> sum(∇²(u)),
						   (u, x) -> 0.0,
						   (0.0, 1.0),
						   ((0.0, 1.0), (0.0, 1.0)))
		coupling = (ode_sol, pde_sol) -> ode_sol[end] * pde_sol
		coupled_problem = KSCoupledProblem([ode, pde], [nothing coupling; nothing nothing])

		@test coupled_problem isa KSCoupledProblem
		@test length(coupled_problem.problems) == 2
		@test size(coupled_problem.coupling_terms) == (2, 2)

		# Test solving (simplified)
		mesh = Preprocessing.create_mesh(pde.domain, (10, 10), 2, KSCartesianCoordinates{2, Float64}((0.0, 0.0)))
		ode_solver = KSIterativeSolver(:cg, 1000, 1e-6, nothing)
		pde_solver = KSDirectSolver(:lu)
		t, y = solve_coupled_problem(coupled_problem, mesh, [ode_solver, pde_solver])
		@test length(t) > 0
		@test all(length.(y) .> 0)
	end
end
@testset "IDEs" begin
	@testset "KSIDEProblem" begin
		# Test IDE: u'(t) = u(t) + ∫₀ᵗ e^(t-s)u(s)ds, u(0) = 1
		ide = KSIDEProblem((u, int) -> u + int,
						   (s, t) -> exp(t - s),
						   (0.0, 1.0),
						   [1.0])
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
analytical_pde(x, y) = sin(π * x) * sin(π * y)

mesh = create_mesh([0, 1, 0, 1], (10, 10), 5)  # 10x10 elements, polynomial degree 5
numerical_solution = solve_equation(pde_problem, mesh)

# Calculate error
error = 0.0
for (i, element) in enumerate(mesh.elements)
	for (j, node) in enumerate(element.points)
		x, y = node.coordinates
		error = max(error, abs(numerical_solution[i][j] - analytical_pde(x, y)))
	end
end

@testset "PDEs" begin
	@testset "KSPDEProblem - Elliptic" begin
		# Test PDE: -Δu = f, u = 0 on boundary
		pde = KSPDEProblem((u, x, t) -> -sum(∇²(u)),
						   (u, x) -> 0.0,
						   (0.0, 1.0),
						   ((0.0, 1.0), (0.0, 1.0)))
		@test pde isa KSPDEProblem
		@test pde.domain == ((0.0, 1.0), (0.0, 1.0))

		# Test discretization and solving
		mesh = Preprocessing.create_mesh(pde.domain, (10, 10), 2, KSCartesianCoordinates{2, Float64}((0.0, 0.0)))
		A, b = assemble_PDE_problem(pde, mesh)
		@test size(A, 1) == size(A, 2)
		@test length(b) == size(A, 1)

		solver = KSDirectSolver(:lu)
		u = solve_linear_system(A, b, solver)
		@test length(u) == size(A, 1)
	end

	@testset "KSPDEProblem - Parabolic" begin
		# Test PDE: ∂u/∂t = Δu, u = 0 on boundary, u(x,0) = sin(πx)sin(πy)
		pde = KSPDEProblem((u, x, t) -> sum(∇²(u)),
						   (u, x) -> 0.0,
						   (0.0, 1.0),
						   ((0.0, 1.0), (0.0, 1.0)))
		@test pde isa KSPDEProblem

		# Test time-stepping
		mesh = Preprocessing.create_mesh(pde.domain, (10, 10), 2, KSCartesianCoordinates{2, Float64}((0.0, 0.0)))
		u0 = [sin(π * p.coordinates[1]) * sin(π * p.coordinates[2]) for p in mesh.nodes]
		dt = 0.01
		nsteps = 10
		u = u0
		for _ in 1:nsteps
			A, b = assemble_PDE_problem(pde, mesh, u, dt)
			solver = KSDirectSolver(:lu)
			u = solve_linear_system(A, b, solver)
		end
		@test length(u) == length(u0)
		@test maximum(abs.(u)) < maximum(abs.(u0))  # Solution should decay
	end

	@testset "Different Coordinate Systems" begin
		@testset "Polar Coordinates" begin
			# Test PDE in polar coordinates: Δu = f
			pde = KSPDEProblem((u, r, theta) -> 1 / r * ∂r(r * ∂r(u)) + 1 / r^2 * ∂θ²(u),
							   (u, r, theta) -> 0.0,
							   (0.0, 1.0),
							   ((0.0, 1.0), (0.0, 2π)))
			@test pde isa KSPDEProblem

			mesh = Preprocessing.create_mesh(pde.domain, (10, 20), 2, KSPolarCoordinates{Float64}(1.0, 0.0))
			A, b = assemble_PDE_problem(pde, mesh)
			@test size(A, 1) == size(A, 2)
			@test length(b) == size(A, 1)

			solver = KSDirectSolver(:lu)
			u = solve_linear_system(A, b, solver)
			@test length(u) == size(A, 1)
		end

		@testset "Spherical Coordinates" begin
			# Test PDE in spherical coordinates: Δu = f
			pde = KSPDEProblem((u, r, theta, phi) -> 1 / r^2 * ∂r(r^2 * ∂r(u)) + 1 / (r^2 * sin(θ)) * ∂θ(sin(θ) * ∂θ(u)) + 1 / (r^2 * sin(θ)^2) * ∂φ²(u),
							   (u, r, theta, phi) -> 0.0,
							   (0.0, 1.0),
							   ((0.0, 1.0), (0.0, π), (0.0, 2π)))
			@test pde isa KSPDEProblem

			mesh = Preprocessing.create_mesh(pde.domain, (10, 10, 20), 2, KSSphericalCoordinates{Float64}(1.0, 0.0, 0.0))
			A, b = assemble_PDE_problem(pde, mesh)
			@test size(A, 1) == size(A, 2)
			@test length(b) == size(A, 1)

			solver = KSDirectSolver(:lu)
			u = solve_linear_system(A, b, solver)
			@test length(u) == size(A, 1)
		end
	end
end
pide_problem = KSPIDEProblem{Float64, 1}((D, u, nodes) -> D[1]^2 * u,  # Spatial operator (∂²u/∂x²)
										 (u, nodes, solution, mesh) -> [sum(u .* mesh.weights) for _ in 1:length(u)],  # Integral operator
										 (u, nodes) -> u,  # Time derivative (∂u/∂t)
										 (nodes) -> zeros(size(nodes, 1)),  # Source term
										 (nodes) -> [0.0, 0.0])

analytical_pide(x, t) = sin(π * x) * exp(t * (1 - π^2))

@testset "PIDEs" begin
	@testset "KSPIDEProblem" begin
		@test pide_problem isa KSPIDE{Float64, 1}

		mesh = create_mesh([0, 1], 50, 3)
		t_final = 1.0
		dt = 0.01
		numerical_solution = solve_equation(pide_problem, mesh, t_final, dt)

		# Check boundary conditions
		@test all(isapprox.(numerical_solution[1, :], 0.0, atol = 1e-6))
		@test all(isapprox.(numerical_solution[end, :], 0.0, atol = 1e-6))

		# Check solution at final time
		for (i, x) in enumerate(mesh.nodes)
			@test isapprox(numerical_solution[i, end], analytical_pide(x, t_final), rtol = 1e-2)
		end

		# Check overall error
		max_error = maximum(abs.(numerical_solution[:, end] .- analytical_pide.(mesh.nodes, t_final)))
		@test max_error < 1e-2
	end
end
