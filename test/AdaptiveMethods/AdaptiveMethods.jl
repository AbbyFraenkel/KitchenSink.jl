using LinearAlgebra, FastGaussQuadrature, Test
using ..KSTypes, ..AdaptiveMethods, ..SpectralMethods, ..Preprocessing, ..MultiLevelMethods
using ..CoordinateSystems, ..TimeStepping, ..ProblemTypes
function create_test_problem(dim::Int, coord_system::Symbol)
    if dim == 1
        domain = ((0.0, 1.0),)
        f(x) = sin(π * x[1])
        u_exact(x) = sin(π * x[1]) / π^2
        bc(x) = 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{1,Float64}((0.0,)) :
                coord_system == :polar ? KSPolarCoordinates{Float64}(1.0, 0.0) :
                error("Unsupported 1D coordinate system")
    elseif dim == 2
        domain = ((0.0, 1.0), (0.0, 1.0))
        f(x) = 2π^2 * sin(π * x[1]) * sin(π * x[2])
        u_exact(x) = sin(π * x[1]) * sin(π * x[2])
        bc(x) = 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{2,Float64}((0.0, 0.0)) :
                coord_system == :polar ? KSPolarCoordinates{Float64}(1.0, 0.0) :
                error("Unsupported 2D coordinate system")
    elseif dim == 3
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        f(x) = 3π^2 * sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        u_exact(x) = sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        bc(x) = 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{3,Float64}((0.0, 0.0, 0.0)) :
                coord_system == :spherical ? KSSphericalCoordinates{Float64}(1.0, 0.0, 0.0) :
                error("Unsupported 3D coordinate system")
    else
        error("Unsupported dimension")
    end
    return KSProblem(x -> -sum(∇²(x)), domain, bc, coord), f, u_exact
end
@testset "Adaptive Methods" begin



    @testset "Time-Independent Adaptivity" begin
        @testset "Elliptic PDE" begin
            # -Δu = f, u = 0 on boundary
            # f chosen so that u = sin(πx)sin(πy) is the exact solution
            function elliptic_problem()
                f(x) = 2π^2 * sin(π * x[1]) * sin(π * x[2])
                u_exact(x) = sin(π * x[1]) * sin(π * x[2])
                domain = ((0.0, 1.0), (0.0, 1.0))
                bc(x) = 0.0
                return KSPDEProblem((u, x) -> -sum(∇²(u)) - f(x), bc, domain, KSCartesianCoordinates{2,Float64}((0.0, 0.0))), u_exact
            end

            problem, u_exact = elliptic_problem()


            @testset "hp-adaptivity" begin
                initial_mesh = Preprocessing.create_mesh(problem.domain, (5, 5), 2, problem.coordinate_system)

                function adapt_and_solve(mesh, max_iterations)
                    for _ in 1:max_iterations
                        A, b = ProblemTypes.assemble_PDE_problem(problem, mesh)
                        u = LinearSolvers.solve_linear_system(A, b, LinearSolvers.KSDirectSolver(:lu))

                        error_estimator(element) = estimate_error(element, u[get_active_indices(element)], problem)
                        smoothness_indicator(element) = estimate_smoothness(element, u[get_active_indices(element)])

                        new_mesh = hp_adapt!(mesh, error_estimator, smoothness_indicator, 1e-3)

                        if length(new_mesh.elements) == length(mesh.elements)
                            break
                        end
                        mesh = new_mesh
                    end
                    return mesh
                end

                final_mesh = adapt_and_solve(initial_mesh, 5)
                @test length(final_mesh.elements) > length(initial_mesh.elements)

                # Check solution accuracy
                A, b = ProblemTypes.assemble_PDE_problem(problem, final_mesh)
                u = LinearSolvers.solve_linear_system(A, b, LinearSolvers.KSDirectSolver(:lu))
                error = maximum(abs.(u .- [u_exact(p.coordinates) for p in final_mesh.nodes]))
                @test error < 1e-3
            end
        end
    end

    @testset "Time-Dependent Adaptivity" begin
        @testset "Parabolic PDE" begin
            # ∂u/∂t = Δu, u = 0 on boundary, u(x,y,0) = sin(πx)sin(πy)
            function parabolic_problem()
                u_exact(x, t) = exp(-2π^2 * t) * sin(π * x[1]) * sin(π * x[2])
                domain = ((0.0, 1.0), (0.0, 1.0))
                tspan = (0.0, 0.1)
                bc(x) = 0.0
                ic(x) = sin(π * x[1]) * sin(π * x[2])
                return KSPDEProblem((u, x, t) -> sum(∇²(u)), bc, tspan, domain, KSCartesianCoordinates{2,Float64}((0.0, 0.0))), ic, u_exact
            end

            problem, ic, u_exact = parabolic_problem()

            @testset "Space-time hp-adaptivity" begin
                initial_mesh = Preprocessing.create_mesh(problem.domain, (5, 5), 2, problem.coordinate_system)
                u = [ic(p.coordinates) for p in initial_mesh.nodes]

                function adapt_and_solve(mesh, u, t, dt, final_time)
                    while t < final_time
                        # Time stepping
                        A, b = ProblemTypes.assemble_PDE_problem(problem, mesh, u, dt)
                        u_new = LinearSolvers.solve_linear_system(A, b, LinearSolvers.KSDirectSolver(:lu))

                        # Spatial adaptation
                        error_estimator(element) = estimate_error(element, u_new[get_active_indices(element)], problem)
                        smoothness_indicator(element) = estimate_smoothness(element, u_new[get_active_indices(element)])

                        new_mesh = hp_adapt!(mesh, error_estimator, smoothness_indicator, 1e-3)

                        if length(new_mesh.elements) != length(mesh.elements)
                            # Interpolate solution to new mesh
                            u_new = interpolate_solution(u_new, mesh, new_mesh)
                            mesh = new_mesh
                        end

                        u = u_new
                        t += dt

                        # Adaptive time-stepping
                        dt = TimeStepping.adaptive_timestep!(u, t, dt, (u, t) -> ProblemTypes.pde_rhs(problem, u, t), KSSolverOptions(100, 1e-4, true), 1e-4)
                    end
                    return mesh, u
                end

                final_mesh, final_u = adapt_and_solve(initial_mesh, u, 0.0, 0.01, problem.tspan[2])
                @test length(final_mesh.elements) > length(initial_mesh.elements)

                # Check solution accuracy
                error = maximum(abs.(final_u .- [u_exact(p.coordinates, problem.tspan[2]) for p in final_mesh.nodes]))
                @test error < 1e-3
            end
        end
    end

    @testset "Preprocessing with Adaptivity" begin
        @testset "Mesh Preprocessing" begin
            function complex_domain()
                # A domain with a circular hole
                in_domain(x) = 0.1^2 <= x[1]^2 + x[2]^2 <= 1.0
                domain = ((-1.0, 1.0), (-1.0, 1.0))
                return domain, in_domain
            end

            domain, in_domain = complex_domain()

            @testset "Adaptive mesh generation" begin
                function refine_criterion(element)
                    center = sum(p.coordinates for p in element.points) / length(element.points)
                    return abs(sqrt(sum(center .^ 2)) - 0.1) < 0.05  # Refine near the hole boundary
                end

                initial_mesh = Preprocessing.create_mesh(domain, (10, 10), 2, KSCartesianCoordinates{2,Float64}((0.0, 0.0)))

                function adaptive_mesh_generation(mesh, max_iterations)
                    for _ in 1:max_iterations
                        elements_to_refine = findall(refine_criterion, mesh.elements)
                        if isempty(elements_to_refine)
                            break
                        end
                        mesh = refine_mesh_hierarchy([mesh], 1, elements_to_refine)[1]
                    end
                    return mesh
                end

                final_mesh = adaptive_mesh_generation(initial_mesh, 5)
                @test length(final_mesh.elements) > length(initial_mesh.elements)

                # Check that the mesh respects the domain
                @test all(in_domain(p.coordinates) for element in final_mesh.elements for p in element.points)
            end
        end
    end


    @testset "Integration with MultiLevelMethods" begin
        @testset "Adaptive Multi-level Solver" begin
            for dim in 1:3, coord_system in [:cartesian, dim == 3 ? :spherical : :polar]
                problem, f, u_exact = create_test_problem(dim, coord_system)
                base_mesh = Preprocessing.create_mesh(problem.domain, ntuple(_ -> 2, dim), 2, problem.coordinate_system)
                Preprocessing.update_tensor_product_masks!(base_mesh)

                hierarchy = MultiLevelMethods.create_mesh_hierarchy(base_mesh, 3)

                # Perform a few V-cycles
                u = zeros(length(hierarchy[end].elements) * (hierarchy[end].elements[1].polynomial_degree + 1)^dim)
                for _ in 1:3
                    u = MultiLevelMethods.v_cycle(hierarchy, f, u, length(hierarchy))
                end

                # Estimate error and perform adaptive refinement
                error_estimates = [estimate_error(element, u[MultiLevelMethods.get_active_indices(element)], problem) for element in hierarchy[end].elements]
                refined_mesh = adapt_mesh_superposition(hierarchy[end], u, problem, maximum(error_estimates) / 2)

                # Check if refinement occurred
                @test length(refined_mesh.elements) > length(hierarchy[end].elements)

                # Solve on the refined mesh
                refined_hierarchy = MultiLevelMethods.create_mesh_hierarchy(refined_mesh, 3)
                u_refined = MultiLevelMethods.solve_multi_level(refined_hierarchy, f, 3)

                # Check if the solution improved
                error_before = norm(u - [u_exact(p.coordinates) for element in hierarchy[end].elements for p in element.points])
                error_after = norm(u_refined - [u_exact(p.coordinates) for element in refined_mesh.elements for p in element.points])
                @test error_after < error_before
            end
        end
    end

    include("error_estimation.jl")
    include("hp_refinement.jl")
    include("refinement_by_superposition.jl")

end
