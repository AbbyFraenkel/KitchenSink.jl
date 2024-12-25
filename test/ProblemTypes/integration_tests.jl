@testset "Complete Problem Workflows" begin
    # Previous test sets remain the same...

    @testset "Cache Management Integration" begin
        dim = 2
        prob = create_test_problem(:pde, dim)
        mesh, coord_sys = create_test_mesh(dim)

        # Clear caches directly
        CacheManagement.clear_cache!(ProblemTypes.SOLVER_CACHE)
        CacheManagement.clear_cache!(ProblemTypes.STANDARD_CELL_CACHE)

        # First solution should populate cache
        sol1 = solve_problem(prob, mesh)

        # Second solution should use cache
        sol2 = solve_problem(prob, mesh)

        # Solutions should be identical when using cache
        @test sol1 ≈ sol2

        # Cache should contain standard cell data
        @test !isempty(STANDARD_CELL_CACHE.items)

        # Modify problem and verify cache updates
        new_prob = create_test_problem(:pde, dim; num_vars=2)
        sol3 = solve_problem(new_prob, mesh)

        # New solution should be different due to different problem
        @test length(sol3) != length(sol1)

        # Cache should have grown
        @test length(STANDARD_CELL_CACHE.items) >= 1
    end

    @testset "Coordinate System Transformations" begin
        for prob_type in TEST_PROBLEM_TYPES
            dim = 2
            prob = create_test_problem(prob_type, dim)
            mesh, coord_sys = create_test_mesh(dim)

            # Get solution in computational space
            sol_comp = solve_problem(prob, mesh)

            # Transform to physical space if needed
            if needs_transform(prob)
                sol_phys = transform_to_physical_space(sol_comp, prob.coordinate_system, mesh)

                # Verify physical solution
                @test length(sol_phys) == length(sol_comp)
                @test all(isfinite, sol_phys)
                @test verify_solution_properties(sol_phys, prob)
            end
        end
    end

    @testset "Memory Management" begin
        dim = 2
        prob = create_test_problem(:pde, dim)
        mesh, coord_sys = create_test_mesh(dim)

        # Track memory allocations
        allocs = @allocated begin
            A, b = create_system_matrix_and_vector(prob, mesh, coord_sys)
            sol = solve_problem(prob, mesh)
        end

        # Solve again to check caching effect
        allocs_cached = @allocated begin
            A, b = create_system_matrix_and_vector(prob, mesh, coord_sys)
            sol = solve_problem(prob, mesh)
        end

        # Cached solution should allocate less
        @test allocs_cached < allocs

        # Clean up for next tests
        GC.gc()
    end

    @testset "Boundary Condition Integration" begin
        dim = 2
        prob = create_test_problem(:pde, dim)
        mesh, coord_sys = create_test_mesh(dim)

        # Get boundary nodes
        boundary_nodes = BoundaryConditions.get_boundary_nodes(mesh)

        # Solve problem
        sol = solve_problem(prob, mesh)

        # Check boundary values
        for node in boundary_nodes
            for bc in prob.boundary_conditions
                if bc.boundary_region(node)
                    expected = if bc.boundary_value isa Function
                        bc.boundary_value(node)
                    else
                        bc.boundary_value
                    end
                    @test sol[node] ≈ expected rtol=1e-10
                end
            end
        end
    end

    @testset "Mesh Refinement Integration" begin
        dim = 2
        for p in TEST_POLYNOMIAL_ORDERS[1:3]
            prob = create_test_problem(:pde, dim)
            mesh, coord_sys = create_test_mesh(dim)

            # Initial solution
            sol = solve_problem(prob, mesh)

            # Update mesh connectivity
            update_mesh_connectivity!(mesh)

            # Solve with updated mesh
            new_sol = solve_problem(prob, mesh)

            # Solutions should be consistent with mesh structure
            @test length(new_sol) == get_total_dof(mesh) * prob.num_vars
            @test verify_solution_properties(new_sol, prob)
        end
    end

    @testset "Linear Solver Integration" begin
        dim = 2
        prob = create_test_problem(:pde, dim)
        mesh, coord_sys = create_test_mesh(dim)

        # Create system
        A, b = create_system_matrix_and_vector(prob, mesh, coord_sys)

        # Test different solver types
        solver_types = [
            KSDirectSolver(:lu),
            KSIterativeSolver(:gmres, 1000, sqrt(eps(Float64))),
            KSAMGSolver(1000, sqrt(eps(Float64)), nothing)
        ]

        for solver in solver_types
            @testset "$(typeof(solver)) solver" begin
                # Solve system
                sol = @test_nowarn LinearSolvers.solve_linear_system(A, b, solver)

                # Verify solution
                @test all(isfinite, sol)
                @test norm(A * sol - b) / norm(b) < sqrt(eps(Float64))
            end
        end
    end

    @testset "Performance Regression" begin
        dim = 2
        prob = create_test_problem(:pde, dim)
        mesh, coord_sys = create_test_mesh(dim)

        # Benchmark standard operations
        time_assembly = @elapsed begin
            A, b = create_system_matrix_and_vector(prob, mesh, coord_sys)
        end

        time_solve = @elapsed begin
            sol = solve_problem(prob, mesh)
        end

        # These thresholds should be adjusted based on the specific hardware
        @test time_assembly < 1.0  # Assembly should be reasonably fast
        @test time_solve < 1.0     # Solve should be reasonably fast

        # Test parallel performance if multiple threads available
        if Threads.nthreads() > 1
            times = Vector{Float64}(undef, Threads.nthreads())
            Threads.@threads for i in 1:Threads.nthreads()
                times[i] = @elapsed solve_problem(prob, mesh)
            end

            # Parallel solutions should have similar timing
            @test maximum(times) / minimum(times) < 2.0
        end
    end

    @testset "Error Propagation" begin
        dim = 2
        prob = create_test_problem(:pde, dim)
        mesh, coord_sys = create_test_mesh(dim)

        # Test error propagation with invalid input
        invalid_mesh = deepcopy(mesh)
        invalid_mesh.cells = KSCell{Float64, dim}[]

        @test_throws ArgumentError solve_problem(prob, invalid_mesh)

        # Test error handling in system creation
        @test_throws ArgumentError create_system_matrix_and_vector(
            prob, invalid_mesh, coord_sys
        )

        # Test boundary condition errors
        invalid_prob = deepcopy(prob)
        invalid_prob.boundary_conditions[1] = KSDirichletBC(
            boundary_value=x -> NaN,
            boundary_region=x -> true,
            coordinate_system=coord_sys
        )

        @test_throws ArgumentError solve_problem(invalid_prob, mesh)
    end
end
