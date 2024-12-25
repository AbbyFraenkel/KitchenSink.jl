@testset "Problem Setup" begin
    @testset "Problem-Mesh Compatibility" begin
        for p in TEST_POLYNOMIAL_ORDERS[1:3]
            for prob_type in TEST_PROBLEM_TYPES
                @testset "p=$p, $prob_type" begin
                    dim = 2
                    prob = create_test_problem(prob_type, dim)
                    # Use compatible mesh creation
                    mesh, coord_sys = create_compatible_test_mesh(prob_type, dim)

                    # Test setup using utilities
                    @test validate_problem_setup(prob, mesh, coord_sys)
                end
            end
        end
    end

    @testset "Coordinate System Compatibility" begin
        dim = 2  # Fixed dimension for tests
        mesh, _ = create_test_mesh(dim)
        cart_sys = KSCartesianCoordinates(ntuple(i -> (-1.0, 1.0), dim))
        polar_sys = KSPolarCoordinates((0.0, 1.0), (0.0, 2π))

        @test all(TEST_PROBLEM_TYPES) do prob_type
            prob = create_test_problem(prob_type, dim)

            if prob isa KSCoupledProblem
                # Test all subproblems at once
                all(prob.problems) do subprob
                    are_coordinate_systems_compatible(subprob.coordinate_system, cart_sys) &&
                    !are_coordinate_systems_compatible(subprob.coordinate_system, polar_sys)
                end
            else
                are_coordinate_systems_compatible(prob.coordinate_system, cart_sys) &&
                !are_coordinate_systems_compatible(prob.coordinate_system, polar_sys)
            end
        end
    end
end

@testset "Solution Process" begin
    @testset "Basic Solution" begin
        # Test all combinations with broadcasting
        @test all(Iterators.product(TEST_POLYNOMIAL_ORDERS[1:3], TEST_PROBLEM_TYPES)) do (p, prob_type)
            dim = 2
            prob = create_test_problem(prob_type, dim)
            mesh, coord_sys = create_test_mesh(dim)

            try
                solution = solve_problem(prob, mesh)
                verify_solution_properties(solution, prob) &&
                length(solution) == get_total_dof(mesh) * prob.num_vars &&
                all(isfinite, solution)
            catch e
                @warn "Solution failed" prob_type dim exception=e
                false
            end
        end
    end
end

@testset "System Size Calculations" begin
    dim = 2
    @testset "DOF Calculations" begin
        # Test all problem types at once
        probs = [create_test_problem(pt, dim) for pt in TEST_PROBLEM_TYPES]
        mesh, _ = create_test_mesh(dim)

        @test all(p -> get_total_dof(p) > 0, probs)
        @test all(p -> validate_dof_consistency(p, mesh), probs)
    end

    @testset "Expected System Size" begin
        mesh, _ = create_test_mesh(dim)
        @test all(TEST_PROBLEM_TYPES) do prob_type
            prob = create_test_problem(prob_type, dim)
            expected_size = get_expected_system_size(prob, mesh)
            A, b = create_system_matrix_and_vector(prob, mesh, prob.coordinate_system)
            size(A, 1) == expected_size && length(b) == expected_size
        end
    end
end

@testset "Cell Operations" begin
    @testset "Cell Finding" begin
        dim = 2
        mesh, _ = create_test_mesh(dim)

        @test all(filter(!isempty, mesh.cells)) do cell
            if cell.is_fictitious
                return true
            end
            center = sum(get_cell_bounds(cell)) ./ 2
            found_idx = find_containing_cell(center, mesh)
            found_idx == cell.id
        end

        # Test point outside mesh
        far_point = fill(1000.0, dim)
        @test isnothing(find_containing_cell(far_point, mesh))
    end
end

@testset "Thread Safety" begin
    @testset "Parallel Solution" begin
        dim = 2
        mesh, _ = create_test_mesh(dim)
        prob = create_test_problem(:pde, dim)

        # Test parallel solutions
        n_threads = 4
        solutions = Vector{Vector{Float64}}(undef, n_threads)

        Threads.@threads for i in 1:n_threads
            solutions[i] = solve_problem(prob, mesh)
        end

        # All solutions should be identical
        @test all(s -> s ≈ solutions[1], solutions[2:end])
    end
end
