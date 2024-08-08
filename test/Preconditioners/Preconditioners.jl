
@testset "Preconditioners" begin
    # Test jacobi_preconditioner
    @testset "jacobi_preconditioner" begin
        A = Symmetric([4.0 1.0; 1.0 3.0])
        preconditioner = jacobi_preconditioner(A)
        @test preconditioner == Diagonal([1/4.0, 1/3.0])

        A_non_symmetric = [4.0 1.0; 2.0 3.0]
        @test_throws ArgumentError jacobi_preconditioner(A_non_symmetric)

        A_non_square = [4.0 1.0 2.0; 1.0 3.0 2.0]
        @test_throws ArgumentError jacobi_preconditioner(A_non_square)
    end

    # Test ilu_preconditioner
    @testset "ilu_preconditioner" begin
        A = sparse([4.0 1.0; 1.0 3.0])
        preconditioner = ilu_preconditioner(A)

        A_non_square = sparse([4.0 1.0 2.0; 1.0 3.0 2.0])
        @test_throws ArgumentError ilu_preconditioner(A_non_square)
    end

    # Test amg_preconditioner
    @testset "amg_preconditioner" begin
        A = sparse(Symmetric([4.0 1.0; 1.0 3.0]))
        preconditioner = amg_preconditioner(A)
        @test isa(preconditioner, AlgebraicMultigrid.MultiLevel)

        A_non_symmetric = sparse([4.0 1.0; 2.0 3.0])
        @test_throws ArgumentError amg_preconditioner(A_non_symmetric)

        A_non_square = sparse([4.0 1.0 2.0; 1.0 3.0 2.0])
        @test_throws ArgumentError amg_preconditioner(A_non_square)
    end
end
