using Test, LinearAlgebra
using .KSTypes, .SpectralMethods, .CommonMethods

# Define a test problem
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
    return KSProblem(x -> -sum(x .^ 2), domain, bc, coord), f, u_exact
end


@testset "Common Methods" begin
    include("error_estimation.jl")
    include("assembly.jl")
    include("solve_implementation.jl")

    @testset "Integration Tests" begin
        @testset "Assemble and Solve" begin
            mesh = create_test_mesh_1D()
            problem, f, u_exact = create_test_problem(1, :cartesian)

            @testset "Assemble System Matrix and RHS" begin
                A = assemble_system_matrix(mesh)
                b = assemble_rhs_vector(mesh, f)
                # Update these expected values based on your actual implementation
                expected_A = [2.0 1.0; 1.0 2.0]  # Example expected result
                expected_b = [1.0, 0.0]          # Example expected result
                @test size(A, 1) == 2
                @test size(A, 2) == 2
                @test isapprox(A, expected_A, atol=1e-6)
                @test length(b) == 2
                @test isapprox(b, expected_b, atol=1e-6)
            end

            @testset "Solve Equation" begin
                solution = solve_equation(problem, mesh)
                expected_solution = [u_exact(mesh.elements[1].points[1].coordinates), u_exact(mesh.elements[1].points[2].coordinates)]
                @test length(solution) == length(mesh.elements)
                @test isapprox(solution, expected_solution, atol=1e-6)
            end
        end

        @testset "Multi-level Solve" begin
            mesh = create_test_mesh_1D()
            hierarchy = [mesh]
            problem, f, u_exact = create_test_problem(1, :cartesian)

            @testset "Solve Multi-level" begin
                solution = solve_multi_level(hierarchy, f, 1)
                expected_solution = [u_exact(mesh.elements[1].points[1].coordinates), u_exact(mesh.elements[1].points[2].coordinates)]
                @test length(solution) == sum(length(el.basis_functions) for el in mesh.elements)
                @test isapprox(solution, expected_solution, atol=1e-6)
            end
        end

        @testset "Adaptive Methods Integration" begin
            mesh = create_test_mesh_1D()
            problem, f, u_exact = create_test_problem(1, :cartesian)
            solution = initialize_solution(mesh)

            @testset "Refine and Solve" begin
                refined_elements = h_refine(mesh.elements[1])
                new_mesh = KSMesh(refined_elements, [true for _ in refined_elements], [Dict{Int,Int}() for _ in refined_elements], [bf for el in refined_elements for bf in el.basis_functions], 0.0, 1)
                A = assemble_system_matrix(new_mesh)
                b = assemble_rhs_vector(new_mesh, f)
                refined_solution = solve_equation(problem, new_mesh)
                expected_solution = [u_exact(el.points[1].coordinates) for el in new_mesh.elements]
                @test length(refined_solution) == length(new_mesh.elements)
                @test isapprox(refined_solution, expected_solution, atol=1e-6)
            end
        end

        @testset "Boundary Conditions and Continuity" begin
            mesh = create_test_mesh_1D()
            problem, f, u_exact = create_test_problem(1, :cartesian)
            solution = initialize_solution(mesh)

            @testset "Apply Boundary Conditions and Enforce Continuity" begin
                apply_boundary_conditions!(solution[1], problem, mesh.elements[1], [true, true])
                enforce_continuity!(solution, mesh)
                expected_solution = [u_exact(mesh.elements[1].points[1].coordinates), u_exact(mesh.elements[1].points[2].coordinates)]
                @test isapprox(solution[1], expected_solution, atol=1e-6)
            end
        end
    end
end
