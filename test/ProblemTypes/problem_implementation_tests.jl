using Test
using KitchenSink
using KitchenSink.ProblemTypes
using KitchenSink.KSTypes
using KitchenSink.SpectralMethods
using KitchenSink.CoordinateSystems
using LinearAlgebra
using SparseArrays

@testset "ProblemTypes.ProblemImplementations" begin
    @testset "PDE Problems" begin
        @testset "Basic PDE Assembly" begin
            problem = create_test_problem(:pde, 2)
            mesh, coord_sys = create_test_mesh(2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            # Initialize local matrices
            local_A = zeros(length(cell.node_map) * problem.num_vars,
                          length(cell.node_map) * problem.num_vars)
            local_b = zeros(length(cell.node_map) * problem.num_vars)

            # Get differential operators
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)

            # Test contribution addition
            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, diff_ops)

            # Check non-zero contributions
            @test any(!iszero, local_A)
            @test any(!iszero, local_b)
        end

        @testset "Temporal PDE Assembly" begin
            problem = create_test_problem(:pde, 2, tspan=(0.0, 1.0))
            mesh, coord_sys = create_test_mesh(2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            n_nodes = length(cell.node_map)
            n_times = has_temporal_dof(problem) ? get_temporal_dof(problem) : 1
            total_size = n_nodes * problem.num_vars * n_times

            local_A = zeros(total_size, total_size)
            local_b = zeros(total_size)
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)

            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, diff_ops)

            # Check temporal blocks
            for t in 1:n_times
                time_offset = (t-1) * n_nodes * problem.num_vars
                block = view(local_A,
                           (time_offset+1):(time_offset+n_nodes*problem.num_vars),
                           (time_offset+1):(time_offset+n_nodes*problem.num_vars))
                @test any(!iszero, block)
            end
        end
    end

    @testset "ODE Problems" begin
        @testset "Basic ODE Assembly" begin
            problem = create_test_problem(:ode, 2)
            mesh, coord_sys = create_test_mesh(2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            local_A = zeros(length(cell.node_map) * problem.num_vars,
                          length(cell.node_map) * problem.num_vars)
            local_b = zeros(length(cell.node_map) * problem.num_vars)
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)

            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, diff_ops)

            @test any(!iszero, local_A)
        end

        @testset "ODE Time Steps" begin
            problem = create_test_problem(:ode, 2, tspan=(0.0, 1.0))
            mesh, coord_sys = create_test_mesh(2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            n_nodes = length(cell.node_map)
            n_times = has_temporal_dof(problem) ? get_temporal_dof(problem) : 1

            local_A = zeros(n_nodes * problem.num_vars * n_times,
                          n_nodes * problem.num_vars * n_times)
            local_b = zeros(n_nodes * problem.num_vars * n_times)
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)

            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, diff_ops)

            # Check each time step
            for t in 1:n_times
                time_offset = (t-1) * n_nodes * problem.num_vars
                @test any(!iszero, view(local_A,
                    (time_offset+1):(time_offset+n_nodes*problem.num_vars), :))
            end
        end
    end

    @testset "DAE Problems" begin
        @testset "DAE Assembly" begin
            # Create DAE problem with 2 differential and 1 algebraic variable
            problem = create_test_problem(:dae, 2, num_vars=3, num_algebraic_vars=1)
            mesh, coord_sys = create_test_mesh(2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            local_A = zeros(length(cell.node_map) * problem.num_vars,
                          length(cell.node_map) * problem.num_vars)
            local_b = zeros(length(cell.node_map) * problem.num_vars)
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)

            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, diff_ops)

            # Check differential and algebraic parts
            n_nodes = length(cell.node_map)
            n_diff = problem.num_vars - problem.num_algebraic_vars

            for i in 1:n_nodes
                diff_part = view(local_A, (i-1)*problem.num_vars .+ (1:n_diff), :)
                alg_part = view(local_A, (i-1)*problem.num_vars .+ (n_diff+1:problem.num_vars), :)

                @test any(!iszero, diff_part)
                @test any(!iszero, alg_part)
            end
        end
    end

    @testset "Moving Boundary Problems" begin
        @testset "Moving Boundary Assembly" begin
            problem = create_test_problem(:moving_boundary, 2)
            mesh, coord_sys = create_compatible_test_mesh(:moving_boundary, 2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            local_A = zeros(length(cell.node_map) * problem.num_vars,
                          length(cell.node_map) * problem.num_vars)
            local_b = zeros(length(cell.node_map) * problem.num_vars)

            # Create differential operators and boundary velocity
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)
            boundary_velocity = ntuple(d -> 0.1, 2)
            ops = (diff_ops, boundary_velocity)

            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, ops)

            @test any(!iszero, local_A)
            @test any(!iszero, local_b)
        end
    end

    @testset "Coupled Problems" begin
        @testset "Coupled System Assembly" begin
            problem = create_test_problem(:coupled_problem, 2)
            mesh, coord_sys = create_test_mesh(2)
            cell = first(filter(c -> !c.is_fictitious, mesh.cells))
            std_cell = SpectralMethods.get_or_create_standard_cell(cell.standard_cell_key...)

            total_dofs = sum(get_total_problem_dof(p, mesh) for p in problem.problems)
            local_A = zeros(total_dofs, total_dofs)
            local_b = zeros(total_dofs)
            diff_ops = ntuple(d -> std_cell.differentiation_matrix_with_boundary[d], 2)

            @test add_problem_contributions!(local_A, local_b, problem, cell, mesh,
                                          std_cell, diff_ops)

            # Check subproblem blocks and coupling terms
            dims = [get_total_problem_dof(p, mesh) for p in problem.problems]
            offsets = [0; cumsum(dims)[1:end-1]]

            for i in 1:length(problem.problems)
                block_range = (offsets[i]+1):(offsets[i]+dims[i])
                @test any(!iszero, view(local_A, block_range, block_range))
            end
        end
    end

    @testset "Utility Functions" begin
        @testset "Differential Operators" begin
            n_nodes = 5
            n_vars = 2
            u = ones(n_nodes * n_vars)
            diff_ops = ntuple(d -> rand(n_nodes, n_nodes), 2)

            derivs = apply_differential_operators(diff_ops, u)

            @test length(derivs) == 2
            @test all(length(d) == length(u) for d in derivs)
        end

        @testset "Quadrature Contributions" begin
            n = 5
            local_A = zeros(n, n)
            local_b = zeros(n)
            val = (rand(2, 2), rand())
            weight = 1.0
            idx = 2

            @test add_quadrature_contribution!(local_A, local_b, val, weight, idx)

            # Check diagonal dominance
            for i in 1:n
                if i == idx
                    row_sum = sum(abs.(local_A[i,:])) - abs(local_A[i,i])
                    @test abs(local_A[i,i]) > row_sum
                end
            end
        end
    end
end
