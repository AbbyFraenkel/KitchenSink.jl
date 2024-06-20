using Test, SparseArrays
using KitchenSink.Types

@testset "Utility Functions" begin

    @testset "check_empty" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_empty(Float64[], "Array")
            @test_throws MethodError check_empty(sparse(Float64[], Int[]), "SparseMatrix")
            @test_throws MethodError check_empty(nothing, "Nothing")
        end
        @testset "Correct Outputs" begin
            @test check_empty([1.0], "Array") == false
            @test check_empty(sparse([1.0], [1], [1.0], 1, 1), "SparseMatrix") == false
        end
    end

    @testset "check_positive" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_positive(-1, "Negative Value")
            @test_throws MethodError check_positive(nothing, "Nothing")
        end
        @testset "Correct Outputs" begin
            @test check_positive(1, "Positive Value") == true
        end
    end

    @testset "check_non_negative" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_non_negative(-1, "Negative Value")
            @test_throws MethodError check_non_negative(nothing, "Nothing")
        end
        @testset "Correct Outputs" begin
            @test check_non_negative(0, "Non-negative Value") == true
        end
    end

    @testset "check_type" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_type(1, Float64, "Type Check")
            @test_throws ArgumentError check_type(nothing, Float64, "Type Check")
        end
        @testset "Correct Outputs" begin
            @test check_type(1.0, Float64, "Type Check") == true
        end
    end

    @testset "check_length" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_length([1, 2, 3], 2, "Length Check")
            @test_throws MethodError check_length(nothing, 2, "Length Check")
        end
        @testset "Correct Outputs" begin
            @test check_length([1, 2], 2, "Length Check") == true
        end
    end

    @testset "check_range" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_range(5, 1, 4, "Range Check")
            @test_throws MethodError check_range(nothing, 1, 4, "Range Check")
        end
        @testset "Correct Outputs" begin
            @test check_range(3, 1, 4, "Range Check") == true
        end
    end

    @testset "check_is_nothing" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_is_nothing(nothing, "Nothing Check")
        end
        @testset "Correct Outputs" begin
            @test check_is_nothing(1, "Nothing Check") == false
        end
    end

    @testset "check_fields" begin
        @testset "Error Handling" begin
            @test_throws ArgumentError check_fields((nothing, 1), "TestStruct")
            @test_throws ArgumentError check_fields(([], 1), "TestStruct")
        end
        @testset "Correct Outputs" begin
            @test check_fields(([1], 1), "TestStruct") == nothing
        end
    end

end
