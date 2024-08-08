@testset "AMG Solvers" begin
    @testset "solve_amg" begin
        # Create a diagonally dominant sparse matrix
        n = 100
        A = sprand(n, n, 0.01)
        A = A + A' + n * I
        b = rand(n)
        x_expected = A \ b

        @testset "Basic functionality" begin
            solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            x = solve_amg(A, b, solver)
            @test x ≈ x_expected rtol = 1e-5
        end

        @testset "Jacobi smoother" begin
            solver = KSAMGSolver(1000, 1e-6, :jacobi)
            x = solve_amg(A, b, solver)
            @test x ≈ x_expected rtol = 1e-5
        end

        @testset "Verbose output" begin
            solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            # Redirect stdout to capture output
            original_stdout = stdout
            (read_pipe, write_pipe) = redirect_stdout()

            solve_amg(A, b, solver, verbose=true)

            # Restore stdout and close the pipe
            redirect_stdout(original_stdout)
            close(write_pipe)

            # Read the captured output
            output = String(read(read_pipe))

            @test occursin("AMG solver converged", output)
        end

        @testset "Non-convergence" begin
            # Create a matrix that won't converge easily
            A_nonconv = sprand(n, n, 0.1)
            solver = KSAMGSolver(10, 1e-12, :gauss_seidel)  # Very strict tolerance and few iterations

            # Redirect stderr to capture warnings
            original_stderr = stderr
            (read_pipe, write_pipe) = redirect_stderr()

            solve_amg(A_nonconv, b, solver, verbose=true)

            # Restore stderr and close the pipe
            redirect_stderr(original_stderr)
            close(write_pipe)

            # Read the captured output
            output = String(read(read_pipe))

            @test occursin("AMG solver did not converge within the specified tolerance", output)
        end

        @testset "Error handling" begin
            solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            @test_throws ArgumentError solve_amg(A[:, 1:n-1], b, solver)
            @test_throws ArgumentError solve_amg(A, b[1:n-1], solver)
            @test_throws ArgumentError solve_amg(A, b, KSAMGSolver(1000, 1e-6, :invalid_smoother))
        end
    end

    @testset "solve_amg_multiple_rhs" begin
        # Create a diagonally dominant sparse matrix
        n = 1000
        m = 5
        A = sprand(n, n, 0.01)
        A = A + A' + n * I
        B = rand(n, m)
        X_expected = A \ B

        @testset "Basic functionality" begin
            solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            X = solve_amg_multiple_rhs(A, B, solver)
            @test X ≈ X_expected rtol = 1e-5
        end

        @testset "Jacobi smoother" begin
            solver = KSAMGSolver(1000, 1e-6, :jacobi)
            X = solve_amg_multiple_rhs(A, B, solver)
            @test X ≈ X_expected rtol = 1e-5
        end

        @testset "Verbose output" begin
            solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            # Redirect stdout to capture output
            original_stdout = stdout
            (read_pipe, write_pipe) = redirect_stdout()

            solve_amg_multiple_rhs(A, B, solver, verbose=true)

            # Restore stdout and close the pipe
            redirect_stdout(original_stdout)
            close(write_pipe)

            # Read the captured output
            output = String(read(read_pipe))

            @test !occursin("AMG solver did not converge", output)
        end

        @testset "Error handling" begin
            solver = KSAMGSolver(1000, 1e-6, :gauss_seidel)
            @test_throws ArgumentError solve_amg_multiple_rhs(A[:, 1:n-1], B, solver)
            @test_throws ArgumentError solve_amg_multiple_rhs(A, B[1:n-1, :], solver)
            @test_throws ArgumentError solve_amg_multiple_rhs(A, B, KSAMGSolver(1000, 1e-6, :invalid_smoother))
        end
    end
end
