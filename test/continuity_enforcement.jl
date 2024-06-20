using Test
using LinearAlgebra
using SparseArrays
using .KitchenSink.ContinuityEnforcement
# include("src/continuity_enforcement")

@testset "ContinuityEnforcement Tests" begin

    @testset "Default Continuity Order" begin
        num_elements = 3
        num_basis = 2
        A = sparse([1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0])
        b = [1.0, 2.0, 3.0, 4.0]

        A_mod, b_mod = ensure_higher_order_continuity!(A, b, num_elements, num_basis)

        expected_A =
            sparse([1.0 -1.0 0.0 0.0; -1.0 1.0 0.0 0.0; 0.0 0.0 1.0 -1.0; 0.0 0.0 -1.0 1.0])
        expected_b = [1.0 - 3.0, 2.0 - 4.0, 3.0, 4.0]

        @test A_mod == expected_A
        @test b_mod == expected_b
    end

    @testset "Zero Elements in Mesh" begin
        num_elements = 0
        num_basis = 2
        A = sparse(
            Float64[],
            Int[],
            Int[],
            (num_elements * num_basis, num_elements * num_basis),
        )
        b = Float64[]

        A_mod, b_mod = ensure_higher_order_continuity!(A, b, num_elements, num_basis)

        @test size(A_mod) == (num_elements * num_basis, num_elements * num_basis)
        @test length(b_mod) == num_elements * num_basis
    end

    @testset "Small Matrices and Vectors" begin
        num_elements = 2
        num_basis = 2
        A = sparse([1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0])
        b = [1.0, 2.0, 3.0, 4.0]

        A_mod, b_mod = ensure_higher_order_continuity!(A, b, num_elements, num_basis)

        expected_A =
            sparse([1.0 -1.0 0.0 0.0; -1.0 1.0 0.0 0.0; 0.0 0.0 1.0 -1.0; 0.0 0.0 -1.0 1.0])
        expected_b = [1.0 - 3.0, 2.0 - 4.0, 3.0, 4.0]

        @test A_mod == expected_A
        @test b_mod == expected_b
    end

    @testset "Different Continuity Order Values" begin
        num_elements = 3
        num_basis = 2
        A = sparse([1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 1.0 1.0])
        b = [1.0, 2.0, 3.0, 4.0]

        A_mod_2, b_mod_2 =
            ensure_higher_order_continuity!(copy(A), copy(b), num_elements, num_basis, 2)
        A_mod_3, b_mod_3 =
            ensure_higher_order_continuity!(copy(A), copy(b), num_elements, num_basis, 3)

        expected_A_2 = sparse(
            [1.0 -1.0 0.0 0.0; -1.0 1.0 -1.0 0.0; 0.0 -1.0 1.0 -1.0; 0.0 0.0 -1.0 1.0],
        )
        expected_b_2 = [1.0 - 3.0, 2.0 - 4.0, 3.0 - b[3], 4.0]

        expected_A_3 = sparse(
            [1.0 -1.0 0.0 0.0; -1.0 1.0 -1.0 0.0; 0.0 -1.0 1.0 -1.0; 0.0 0.0 -1.0 1.0],
        )
        expected_b_3 = [1.0 - 3.0, 2.0 - 4.0, 3.0 - b[3], 4.0]

        @test A_mod_2 == expected_A_2
        @test b_mod_2 == expected_b_2
        @test A_mod_3 == expected_A_3
        @test b_mod_3 == expected_b_3
    end
end
