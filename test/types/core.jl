using Test
using SparseArrays
using KitchenSink.Types

@testset "Core Types" begin

    @testset "HierarchicalBasisFunction" begin
        test_func(x) = x
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError HierarchicalBasisFunction(-1, test_func, 0.1)
                @test_throws ArgumentError HierarchicalBasisFunction(1, test_func, -0.1)
                @test_throws MethodError HierarchicalBasisFunction(1, nothing, 0.1)
                @test_throws MethodError HierarchicalBasisFunction(nothing, test_func, 0.1)
                @test_throws MethodError HierarchicalBasisFunction(1, test_func)
            end
            @testset "Correct Outputs" begin
                hbf = HierarchicalBasisFunction(1, test_func, 0.1)
                @test hbf.id == 1
                @test hbf.func(2) == 2
                @test hbf.decay_rate == 0.1
            end
        end
    end

    @testset "DecayRate" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError DecayRate(Float64[])
                @test_throws MethodError DecayRate(nothing)
                @test_throws MethodError DecayRate()
            end
            @testset "Correct Outputs" begin
                dr = DecayRate([0.1, 0.2])
                @test dr.rates == [0.1, 0.2]
            end
        end
    end

    @testset "TensorProductMask" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError TensorProductMask(
                    Vector{SparseMatrixCSC{Float64,Int}}(),
                )
                @test_throws MethodError TensorProductMask([nothing])
                @test_throws MethodError TensorProductMask()
            end
            @testset "Correct Outputs" begin
                tpm = TensorProductMask([
                    sparse(rand(Float64, 3, 3)),
                    sparse(rand(Float64, 3, 3)),
                ])
                @test length(tpm.masks) == 2
            end
        end
    end

    @testset "LocationMatrix" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws MethodError LocationMatrix(sparse(Float64[], Int[]))
                @test_throws MethodError LocationMatrix(nothing)
                @test_throws MethodError LocationMatrix()
            end
            @testset "Correct Outputs" begin
                lm = LocationMatrix(sparse(rand(Float64, 3, 3)))
                @test size(lm.matrix) == (3, 3)
            end
        end
    end
end
