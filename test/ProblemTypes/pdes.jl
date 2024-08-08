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
        pde = KSPDEProblem(
            (u, x, t) -> -sum(∇²(u)),
            (u, x) -> 0.0,
            (0.0, 1.0),
            ((0.0, 1.0), (0.0, 1.0))
        )
        @test pde isa KSPDEProblem
        @test pde.domain == ((0.0, 1.0), (0.0, 1.0))

        # Test discretization and solving
        mesh = Preprocessing.create_mesh(pde.domain, (10, 10), 2, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))
        A, b = assemble_PDE_problem(pde, mesh)
        @test size(A, 1) == size(A, 2)
        @test length(b) == size(A, 1)

        solver = KSDirectSolver(:lu)
        u = solve_linear_system(A, b, solver)
        @test length(u) == size(A, 1)
    end

    @testset "KSPDEProblem - Parabolic" begin
        # Test PDE: ∂u/∂t = Δu, u = 0 on boundary, u(x,0) = sin(πx)sin(πy)
        pde = KSPDEProblem(
            (u, x, t) -> sum(∇²(u)),
            (u, x) -> 0.0,
            (0.0, 1.0),
            ((0.0, 1.0), (0.0, 1.0))
        )
        @test pde isa KSPDEProblem

        # Test time-stepping
        mesh = Preprocessing.create_mesh(pde.domain, (10, 10), 2, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))
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
            pde = KSPDEProblem(
                (u, r, θ) -> 1 / r * ∂r(r * ∂r(u)) + 1 / r^2 * ∂θ²(u),
                (u, r, θ) -> 0.0,
                (0.0, 1.0),
                ((0.0, 1.0), (0.0, 2π))
            )
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
            pde = KSPDEProblem(
                (u, r, θ, φ) -> 1 / r^2 * ∂r(r^2 * ∂r(u)) + 1 / (r^2 * sin(θ)) * ∂θ(sin(θ) * ∂θ(u)) + 1 / (r^2 * sin(θ)^2) * ∂φ²(u),
                (u, r, θ, φ) -> 0.0,
                (0.0, 1.0),
                ((0.0, 1.0), (0.0, π), (0.0, 2π))
            )
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
