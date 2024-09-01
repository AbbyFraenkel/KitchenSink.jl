using Test
using LinearAlgebra, SparseArrays

using KitchenSink.MultiLevelMethods
@testset "MultiLevelMethods" begin

    @testset "create_mesh_hierarchy" begin
        # Create a simple 1D base mesh
        domain = [(0.0, 1.0)]
        num_elements = [4]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

        @testset "Valid input" begin
            hierarchy = create_mesh_hierarchy(base_mesh, 3)
            @test length(hierarchy) == 3
            @test all(mesh -> mesh isa KSMesh{Float64, 1}, hierarchy)
            @test all(i -> length(hierarchy[i].elements) > length(hierarchy[i-1].elements), 2:3)
        end

        @testset "Invalid input" begin
            @test_throws ArgumentError create_mesh_hierarchy(base_mesh, 0)
            @test_throws ArgumentError create_mesh_hierarchy(base_mesh, -1)
        end
    end

    @testset "coarsen_mesh" begin
        # Create a simple 1D mesh with 4 elements
        domain = [(0.0, 1.0)]
        num_elements = [4]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

        @testset "Valid input" begin
            coarse_mesh = coarsen_mesh(mesh)
            @test coarse_mesh isa KSMesh{Float64, 1}
            @test length(coarse_mesh.elements) == 2
        end

        @testset "Invalid input" begin
            single_element_mesh = CommonMethods.create_mesh(domain, coord_system, [1], polynomial_degree)
            @test_throws ArgumentError coarsen_mesh(single_element_mesh)
        end
    end

    @testset "merge_elements" begin
        # Create two adjacent 1D elements
        domain = [(0.0, 1.0)]
        coord_system = KSCartesianCoordinates(domain)
        el1 = CommonMethods.create_elements([KSPoint([0.0]), KSPoint([0.5])], [1], 2, coord_system)[1]
        el2 = CommonMethods.create_elements([KSPoint([0.5]), KSPoint([1.0])], [1], 2, coord_system)[1]

        merged_element = merge_elements(el1, el2)
        @test merged_element isa KSElement{Float64, 1}
        @test length(merged_element.points) == 3
        @test merged_element.polynomial_degree == max(el1.polynomial_degree, el2.polynomial_degree)
    end

    @testset "refine_mesh_uniformly" begin
        domain = [(0.0, 1.0)]
        num_elements = [4]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

        refined_mesh = refine_mesh_uniformly(mesh)
        @test refined_mesh isa KSMesh{Float64, 1}
        @test length(refined_mesh.elements) == 2 * length(mesh.elements)
    end

    @testset "refine_mesh_hierarchy" begin
        domain = [(0.0, 1.0)]
        num_elements = [4]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)

        @testset "Valid input" begin
            new_hierarchy = refine_mesh_hierarchy(hierarchy, 2, [1, 3])
            @test length(new_hierarchy) == 3
            @test length(new_hierarchy[2].elements) > length(hierarchy[2].elements)
        end

        @testset "Invalid input" begin
            @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 0, [1])
            @test_throws ArgumentError refine_mesh_hierarchy(hierarchy, 4, [1])
            @test refine_mesh_hierarchy(hierarchy, 2, Int[]) == hierarchy
        end
    end

    @testset "refine_marked_elements" begin
        domain = [(0.0, 1.0)]
        num_elements = [4]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

        refined_mesh = refine_marked_elements(mesh, [1, 3])
        @test refined_mesh isa KSMesh{Float64, 1}
        @test length(refined_mesh.elements) == length(mesh.elements) + 2
    end

    @testset "adjust_finer_level" begin
        domain = [(0.0, 1.0)]
        num_elements = [4]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        coarse_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
        fine_mesh = refine_mesh_uniformly(coarse_mesh)

        adjusted_fine_mesh = adjust_finer_level(coarse_mesh, fine_mesh)
        @test adjusted_fine_mesh isa KSMesh{Float64, 1}
        @test length(adjusted_fine_mesh.elements) == length(fine_mesh.elements)
    end

    @testset "Multigrid cycles" begin
        # Create a simple 1D problem
        domain = [(0.0, 1.0)]
        num_elements = [8]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
        hierarchy = create_mesh_hierarchy(base_mesh, 3)

        # Define a simple 1D problem: -u'' = f, u(0) = u(1) = 0
        f(x) = sin(π * x[1])
        u_exact(x) = sin(π * x[1]) / (π^2)

        A = CommonMethods.assemble_system_matrix(hierarchy[end])
        b = CommonMethods.assemble_rhs_vector(hierarchy[end], f)

        @testset "v_cycle" begin
            u = zeros(length(b))
            u = v_cycle(hierarchy, x -> b - A * x, u, length(hierarchy))
            @test norm(A * u - b) < 1e-6
        end

        @testset "w_cycle" begin
            u = zeros(length(b))
            u = w_cycle(hierarchy, x -> b - A * x, u, length(hierarchy))
            @test norm(A * u - b) < 1e-6
        end

        @testset "full_multigrid" begin
            u = full_multigrid(hierarchy, x -> b - A * x)
            @test norm(A * u - b) < 1e-6
        end
    end

    @testset "geometric_multigrid" begin
        # Create a simple 1D problem
        domain = [(0.0, 1.0)]
        num_elements = [16]
        polynomial_degree = 2
        coord_system = KSCartesianCoordinates(domain)
        base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
        hierarchy = create_mesh_hierarchy(base_mesh, 4)

        # Define a simple 1D problem: -u'' = f, u(0) = u(1) = 0
        f(x) = sin(π * x[1])
        u_exact(x) = sin(π * x[1]) / (π^2)

        A = CommonMethods.assemble_system_matrix(hierarchy[end])
        b = CommonMethods.assemble_rhs_vector(hierarchy[end], f)

        @testset "V-cycle" begin
            x = geometric_multigrid(A, b, hierarchy, cycle_type=:v)
            @test norm(A * x - b) < 1e-6
        end

        @testset "W-cycle" begin
            x = geometric_multigrid(A, b, hierarchy, cycle_type=:w)
            @test norm(A * x - b) < 1e-6
        end

        @testset "Full Multigrid" begin
            x = geometric_multigrid(A, b, hierarchy, cycle_type=:fmg)
            @test norm(A * x - b) < 1e-6
        end

        @testset "Invalid cycle type" begin
            @test_throws ErrorException geometric_multigrid(A, b, hierarchy, cycle_type=:invalid)
        end
    end

    @testset "algebraic_multigrid" begin
        # Create a simple sparse matrix problem
        n = 100
        A = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.0, n-1))
        b = ones(n)

        x = algebraic_multigrid(A, b)
        @test norm(A * x - b) < 1e-6
    end

    @testset "Helper functions" begin
        @testset "solve_coarse_problem" begin
            domain = [(0.0, 1.0)]
            num_elements = [4]
            polynomial_degree = 2
            coord_system = KSCartesianCoordinates(domain)
            mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

            f(x) = sin(π * x[1])
            u = solve_coarse_problem(mesh, f)
            @test length(u) == CommonMethods.total_dofs(mesh)
        end

        @testset "smooth and gauss_seidel_iteration" begin
            domain = [(0.0, 1.0)]
            num_elements = [4]
            polynomial_degree = 2
            coord_system = KSCartesianCoordinates(domain)
            mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

            f(x) = sin(π * x[1])
            u = zeros(CommonMethods.total_dofs(mesh))
            u_smoothed = smooth(mesh, f, u)
            @test length(u_smoothed) == length(u)
            @test norm(u_smoothed) > 0
        end

        @testset "compute_residual" begin
            domain = [(0.0, 1.0)]
            num_elements = [4]
            polynomial_degree = 2
            coord_system = KSCartesianCoordinates(domain)
            mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)

            f(x) = sin(π * x[1])
            u = zeros(CommonMethods.total_dofs(mesh))
            r = compute_residual(mesh, f, u)
            @test length(r) == length(u)
        end

        @testset "validate_mesh_hierarchy" begin
            domain = [(0.0, 1.0)]
            num_elements = [4]
            polynomial_degree = 2
            coord_system = KSCartesianCoordinates(domain)
            base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
            hierarchy = create_mesh_hierarchy(base_mesh, 3)

            @test_nowarn validate_mesh_hierarchy(hierarchy)
            @test_throws ArgumentError validate_mesh_hierarchy(KSMesh{Float64, 1}[])
            @test_throws ArgumentError validate_mesh_hierarchy([base_mesh, base_mesh])
        end

        @testset "compute_mesh_complexity" begin
            domain = [(0.0, 1.0)]
            num_elements = [4]
            polynomial_degree = 2
            coord_system = KSCartesianCoordinates(domain)
            base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
            hierarchy = create_mesh_hierarchy(base_mesh, 3)

            complexity = compute_mesh_complexity(hierarchy)
            @test complexity > 1.0
        end

        @testset "estimate_convergence_factor" begin
            domain = [(0.0, 1.0)]
            num_elements = [8]
            polynomial_degree = 2
            coord_system = KSCartesianCoordinates(domain)
            base_mesh = CommonMethods.create_mesh(domain, coord_system, num_elements, polynomial_degree)
            hierarchy = create_mesh_hierarchy(base_mesh, 3)

            f(x) = sin(π * x[1])
            u_exact(x) = sin(π * x[1]) / (π^2)
            u_exact_vec = [u_exact([x]) for x in LinRange(0, 1, CommonMethods.total_dofs(hierarchy[end]))]

            conv_factor = estimate_convergence_factor(hierarchy, f, u_exact_vec, 5)
            @test 0 < conv_factor < 1
        end

        @testset "geometric_mean" begin
            @test isapprox(geometric_mean([1.0, 2.0, 3.0]), 1.8171205928321397)
            @test isapprox(geometric_mean([0.1, 1.0, 10.0]), 1.0)
        end
    end
end
