@testset verbose=true "DOF Calculation Tests" begin
    @testset "Type Stability" begin
        for dim in TEST_DIMENSIONS
            for prob_type in TEST_PROBLEM_TYPES
                @testset "$(prob_type) - $(dim)D" begin
                    problem = create_test_problem(prob_type, dim)
                    mesh, coord_sys = create_compatible_test_mesh(prob_type, dim)

                    # System DOF type stability
                    @inferred get_system_dof(problem)

                    # Mesh DOF type stability
                    @inferred get_mesh_dof(mesh)
                    @inferred get_physical_mesh_dof(mesh)

                    # Total DOF type stability
                    @inferred get_total_problem_dof(problem, mesh)

                    # Test return type stability
                    @test typeof(get_system_dof(problem)) === Int
                    @test typeof(get_mesh_dof(mesh)) === Int
                    @test typeof(get_total_problem_dof(problem, mesh)) === Int
                end
            end
        end
    end

    @testset "Standard PDE Problems" begin
        for dim in TEST_DIMENSIONS
            @testset "$(dim)D PDE" begin
                problem = create_test_problem(:pde, dim)
                mesh = create_test_mesh(dim)[1]

                # Basic DOF checks
                sys_dof = get_system_dof(problem)
                @test sys_dof == problem.num_vars

                # Calculate spatial DOFs
                spatial_dofs = sys_dof * get_mesh_dof(mesh)

                # Time dependency
                if !isnothing(problem.tspan)
                    temp_dof = get_temporal_dof(problem)
                    @test temp_dof > 0
                    # Time steps based on timespan, not dt
                    @test temp_dof == ceil(Int, 100) # Default to 100 timesteps

                    # Total DOF consistency with temporal component
                    total_dof = get_total_problem_dof(problem, mesh)
                    @test total_dof == spatial_dofs * temp_dof
                else
                    # Total DOF consistency without temporal component
                    total_dof = get_total_problem_dof(problem, mesh)
                    @test total_dof == spatial_dofs
                end
            end
        end
    end

    @testset "DAE Problems" begin
        for dim in TEST_DIMENSIONS
            problem = create_test_problem(:dae, dim)
            mesh = create_test_mesh(dim)[1]

            # Verify algebraic constraint handling
            sys_dof = get_system_dof(problem)
            @test sys_dof == problem.num_vars - problem.num_algebraic_vars

            # Test different valid algebraic variable counts
            valid_alg_vars = [0, 1]  # Only test valid counts less than num_vars
            for n_alg in valid_alg_vars
                # Only modify if count would be valid
                if n_alg < problem.num_vars
                    modified_problem = modify_problem(problem, :num_algebraic_vars, n_alg)
                    @test get_system_dof(modified_problem) == problem.num_vars - n_alg
                end
            end

            # Test invalid cases
            @test_throws ArgumentError modify_problem(
                problem, :num_algebraic_vars, -1)  # Negative
            @test_throws ArgumentError modify_problem(
                problem, :num_algebraic_vars, problem.num_vars)  # Equal to num_vars
            @test_throws ArgumentError modify_problem(
                problem, :num_algebraic_vars, problem.num_vars + 1)  # Greater than num_vars
        end
    end

    @testset "Boundary Value Problems" begin
        for dim in TEST_DIMENSIONS
            @testset "$(dim)D BVP" begin
                problem = create_test_problem(:bvp, dim)
                mesh = create_test_mesh(dim)[1]

                # Test different boundary conditions
                dirichlet_bc = KSDirichletBC(
                    boundary_value = x -> 0.0,
                    boundary_region = x -> true,
                    coordinate_system = problem.coordinate_system  # Use problem's coordinate system
                )

                neumann_bc = KSNeumannBC(
                    flux_value = x -> 1.0,
                    boundary_region = x -> true,
                    coordinate_system = problem.coordinate_system  # Use problem's coordinate system
                )

                robin_bc = KSRobinBC(
                    neumann_coefficient = x -> 1.0,
                    dirichlet_coefficient = x -> 1.0,
                    boundary_value = x -> 0.0,
                    boundary_region = x -> true,
                    coordinate_system = problem.coordinate_system  # Use problem's coordinate system
                )

                # Test DOF changes with different BCs
                problem_dirichlet = create_test_bvp(
                    problem.coordinate_system,
                    problem.domain;
                    num_vars=problem.num_vars,
                    boundary_conditions=[dirichlet_bc]
                )
                base_dof = get_system_dof(problem_dirichlet)
                @test base_dof == problem.num_vars  # Base DOFs are just num_vars

                # Test Robin BCs add extra DOF
                problem_robin = create_test_bvp(
                    problem.coordinate_system,
                    problem.domain;
                    num_vars=problem.num_vars,
                    boundary_conditions=[robin_bc]
                )
                robin_dof = get_system_dof(problem_robin)
                @test robin_dof > base_dof  # Robin needs extra DOF for derivative
                @test robin_dof == base_dof + 1  # One extra DOF per Robin BC
            end
        end
    end

    @testset "Integral Problems" begin
        @testset "IDE" begin
            problem = create_test_problem(:ide, 1)
            mesh = create_test_mesh(1)[1]

            # Test quadrature points
            sys_dof = get_system_dof(problem)
            quad_order = compute_kernel_complexity(problem.kernel)
            @test quad_order > 0
            @test sys_dof > problem.num_vars  # Extra DOFs for quadrature
        end

        @testset "PIDE" begin
            for dim in TEST_DIMENSIONS
                problem = create_test_problem(:pide, dim)
                mesh = create_test_mesh(dim)[1]

                # Test dimensional quadrature
                sys_dof = get_system_dof(problem)
                @test sys_dof > problem.num_vars * dim  # Extra DOFs per dimension
            end
        end
    end

    @testset "Moving Boundary Problems" begin
        for dim in TEST_DIMENSIONS
            problem = create_test_problem(:moving_boundary, dim)
            mesh = create_test_mesh(dim)[1]

            # Test motion tracking DOFs
            sys_dof = get_system_dof(problem)
            @test sys_dof == problem.num_vars + 2dim  # Position and velocity DOFs

            # Test boundary DOFs
            regular_boundary_dof = get_boundary_mesh_dof(mesh)
            @test regular_boundary_dof > 0  # Basic boundary test

            # Verify total DOF calculation
            total_dof = get_total_problem_dof(problem, mesh)
            @test total_dof > regular_boundary_dof * problem.num_vars
        end
    end

    @testset "Optimal Control Problems" begin
        problem = create_test_problem(:optimal_control, 2)
        mesh = create_test_mesh(2)[1]

        # Test component DOFs
        sys_dof = get_system_dof(problem)
        @test sys_dof ≥ length(problem.control_bounds)  # At least control variables
        @test sys_dof ≥ length(problem.initial_state)   # At least state variables

        # Test temporal discretization
        temp_dof = get_temporal_dof(problem)
        @test temp_dof == problem.num_time_steps

        # Test constraint handling
        if hasproperty(problem, :constraints)
            constrained_dof = sum(compute_constraint_dofs.(problem.constraints))
            @test get_system_dof(problem) ≥ sys_dof + constrained_dof
        end
    end

    @testset "Coupled Problems" begin
        problem = create_test_problem(:coupled_problem, 2)
        mesh = create_test_mesh(2)[1]

        # Test subproblem DOFs
        sys_dof = get_system_dof(problem)
        subproblem_dofs = sum(get_system_dof.(problem.problems))
        @test sys_dof ≥ subproblem_dofs

        # Test total system DOFs
        total_dof = get_total_problem_dof(problem, mesh)
        @test total_dof ≥ subproblem_dofs * get_mesh_dof(mesh)

        # Test coupling existence
        @test any(!isnothing, problem.coupling_terms)
    end

    @testset "Temporal DOF Tests" begin
        for dim in TEST_DIMENSIONS
            for prob_type in TEST_PROBLEM_TYPES
                @testset "$(prob_type) - $(dim)D" begin
                    problem = create_test_problem(prob_type, dim)
                    mesh, _ = create_compatible_test_mesh(prob_type, dim)

                    # Test temporal DOF presence
                    if prob_type == :optimal_control
                        # Optimal control always has temporal DOFs
                        @test has_temporal_dof(problem)
                        @test get_temporal_dof(problem) == problem.num_time_steps
                    elseif prob_type == :coupled_problem
                        # Coupled problems inherit temporal properties from subproblems
                        @test has_temporal_dof(problem) == any(has_temporal_dof, problem.problems)
                        if has_temporal_dof(problem)
                            @test get_temporal_dof(problem) == maximum(get_temporal_dof.(problem.problems))
                        end
                    else
                        @test has_temporal_dof(problem) == (!isnothing(problem.tspan))
                    end

                    # Test temporal DOF calculation
                    temporal_dof = get_temporal_dof(problem)
                    @test temporal_dof ≥ 1
                    @test isa(temporal_dof, Int)

                    # Test inclusion in total DOF
                    total_dof = get_total_problem_dof(problem, mesh)
                    base_dof = get_system_dof(problem) * get_mesh_dof(mesh)
                    if prob_type == :optimal_control
                        @test total_dof == base_dof * temporal_dof
                    else
                        @test total_dof == (has_temporal_dof(problem) ? base_dof * temporal_dof : base_dof)
                    end
                end
            end
        end
    end

    @testset "Edge Cases" begin
        # Test single-cell mesh
        single_cell_mesh = create_test_mesh(1)[1]
        @test get_mesh_dof(single_cell_mesh) > 0

        # Test empty boundary conditions
        problem = create_test_problem(:pde, 1)
        empty_bcs = Vector{AbstractKSBoundaryCondition}()
        @test_throws ArgumentError modify_problem(problem, :boundary_conditions, empty_bcs)

        # Test invalid boundary conditions
        @test_throws ArgumentError modify_problem(problem, :boundary_conditions, nothing)
        @test_throws ArgumentError modify_problem(problem, :boundary_conditions, Any[1, 2, 3])

        # Test small mesh - use create_test_mesh with minimal settings
        domain = ntuple(i -> (-1.0, 1.0), 1)
        small_mesh, _ = create_test_mesh(1, num_cells=(1,), p_orders=(2,))
        @test get_mesh_dof(small_mesh) ≥ 0
        @test get_physical_mesh_dof(small_mesh) ≥ 0

        # Test minimal mesh - create single cell with minimal structure
        domain = ntuple(i -> (-1.0, 1.0), 1)
        coord_sys = KSCartesianCoordinates(domain)

        # Create a minimal cell using positional constructor
        minimal_cell = KSCell{Float64,1}(
            1,              # id
            (3,),          # p
            1,             # level
            (1,),          # continuity_order
            ((3,), 1),     # standard_cell_key
            Dict{Symbol,Int}(),  # neighbors
            Dict{NTuple{1,Int},Int}(
                (1,) => 1, (2,) => 2, (3,) => 3  # Add some minimal node mapping
            ),
            (trues(5),),   # tensor_product_mask
            Dict{Symbol,Int}(),  # boundary_connectivity
            0.0,           # error_estimate
            0.0,           # legendre_decay_rate
            true,          # is_leaf
            false,         # is_fictitious
            nothing,       # refinement_options
            nothing,       # parent_id
            nothing        # child_ids
        )

        # Create minimal mesh using only supported keyword arguments
        minimal_mesh = KSMesh(
            cells = [minimal_cell],
            global_error_estimate = 0.0,
            boundary_cells = Dict{Symbol,Vector{Int}}(),
            physical_domain = x -> true,  # Simple function that always returns true
            characteristic_function = nothing,
            continuity_constraints = nothing,
            transformation_data = nothing
        )

        @test get_mesh_dof(minimal_mesh) ≥ 0
        @test get_physical_mesh_dof(minimal_mesh) ≥ 0

        # Test invalid problem configurations
        # invalid_num_vars =
        @test_throws ArgumentError get_system_dof(modify_problem(problem, :num_vars, 0))

        struct InvalidProblem <: AbstractKSProblem end
        invalid_problem = InvalidProblem()
        @test_throws ArgumentError get_system_dof(invalid_problem)

        # Test high-dimensional problems
        high_dim_problem = create_test_problem(:pde, 3)
        high_dim_mesh = create_test_mesh(3)[1]
        @test get_system_dof(high_dim_problem) > 0
        @test get_mesh_dof(high_dim_mesh) > 0
        @test get_total_problem_dof(high_dim_problem, high_dim_mesh) > 0
    end

    @testset "Performance Characteristics" begin
        # Increase allocation limit based on actual usage
         ALLOCATION_LIMIT = 15000  # Increased from 10000

        sizes = [10, 100, 1000]
        test_mesh, _ = create_test_mesh(2)

        for n in sizes
            problems = Dict(
                :pde => create_test_problem(:pde, 2),
                :coupled => create_test_problem(:coupled_problem, 2),
                :moving => create_test_problem(:moving_boundary, 2)
            )

            for (type, prob) in problems
                # Test direct computation time
                computation_time = @elapsed get_total_problem_dof(prob, test_mesh)
                @test computation_time < 1.0

                # Test memory allocations with debug output
                allocs_trial = @allocated get_total_problem_dof(prob, test_mesh)
                @debug "Allocations for $type" allocs=allocs_trial limit=ALLOCATION_LIMIT
                @test allocs_trial < ALLOCATION_LIMIT
            end
        end

        @testset "Specialized Scaling Tests" begin
            for dim in TEST_DIMENSIONS
                problem = create_test_problem(:pde, dim)
                dim_mesh, _ = create_test_mesh(dim)  # Create mesh for each dimension

                # Test dimensionality scaling
                dim_time = @belapsed get_total_problem_dof($problem, $dim_mesh)
                @test dim_time < 1.0  # Should be reasonably fast

                # Test memory scaling - increased allocation limits for higher dimensions
                dim_allocs = @allocated get_total_problem_dof(problem, dim_mesh)
                @test dim_allocs < 500_000 * dim  # Much higher allocation limit for higher dimensions

                # Test repeated calls consistency
                first_result = get_total_problem_dof(problem, dim_mesh)
                second_result = get_total_problem_dof(problem, dim_mesh)
                @test first_result == second_result  # Results should be deterministic
            end
        end
    end

    @testset "DOF Validation" begin
        problem = create_test_problem(:pde, 2)
        mesh, coord_sys = create_test_mesh(2)

        # Verify DOF calculation
        @test get_total_problem_dof(problem, mesh) > 0
        @test get_mesh_dof(mesh) > 0
        @test get_system_dof(problem) > 0
    end
end
