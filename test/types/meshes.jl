using Test
using SparseArrays
using KitchenSink.Types

@testset "Mesh Types" begin

    @testset "Node" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws MethodError Node(-1, [0.0], nothing, Node[], true)
                @test_throws MethodError Node(1, nothing, nothing, Node[], true)
                @test_throws MethodError Node()
            end
            @testset "Correct Outputs" begin
                node = Node{Float64}(1, [0.0, 1.0], nothing, Node{Float64}[], true)
                # node = Node(1, [0.0, 1.0], nothing, Node[], true)
                @test node.id == 1
                @test node.coordinates == [0.0, 1.0]
            end
        end
    end

    @testset "BasisFunction" begin
        test_func(x) = x
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError BasisFunction(-1, test_func, true)
                @test_throws MethodError BasisFunction(1, nothing, true)
                @test_throws MethodError BasisFunction(1, test_func)
            end
            @testset "Correct Outputs" begin
                bf = BasisFunction(1, test_func, true)
                @test bf.id == 1
                @test bf.basis_function(2) == 2
            end
        end
    end

    @testset "Element" begin
        test_func(x) = x
        @testset "Construction" begin
            node = Node(1, [0.0, 1.0], nothing, nothing, true)
            bf = BasisFunction(1, test_func, true)
            tpm = TensorProductMask([
                sparse(rand(Float64, 3, 3)),
                sparse(rand(Float64, 3, 3)),
            ])
            lm = LocationMatrix(sparse(rand(Float64, 3, 3)))
            @testset "Error Handling" begin
                @test_throws MethodError Element(
                    Float64[],
                    [bf],
                    [1],
                    [1],
                    nothing,
                    0,
                    tpm,
                    lm,
                    0.1,
                    Dict(),
                )
                @test_throws MethodError Element(
                    [node],
                    Float64[],
                    [1],
                    [1],
                    nothing,
                    0,
                    tpm,
                    lm,
                    0.1,
                    Dict(),
                )
            end
            @testset "Correct Outputs" begin
                element = Element([node], [bf], [1], [1], nothing, 0, tpm, lm, 0.1, Dict())
                @test length(element.nodes) == 1
                @test length(element.basis_functions) == 1
            end
        end
    end



    @testset "Node Tests" begin
        @testset "Construction" begin
            @testset "Error Handling" begin
                @test_throws ArgumentError Node(
                    -1,
                    [0.0, 1.0],
                    true,
                    nothing,
                    Node{Float64}[],
                )
                @test_throws MethodError Node()
            end
            @testset "Correct Outputs" begin
                node1 = Node(1, [0.0, 1.0], true, nothing, Node{Float64}[])
                @test node1.id == 1
                @test node1.coordinates == [0.0, 1.0]
                @test node1.is_leaf == true
                @test node1.parent == nothing
                @test node1.children == Node{Float64}[]
            end
        end
    end

    @testset "Connectivity Tests" begin
        @testset "Error Handling" begin
            @test_throws MethodError Connectivity([[], [1, 2, 3]])
            @test_throws ArgumentError Connectivity(Vector{Vector{Int}}())
        end

        @testset "Correct Outputs" begin
            conn = Connectivity([[1, 2, 3], [4, 5, 6]])
            @test conn.matrix == [[1, 2, 3], [4, 5, 6]]
        end
    end


    @testset "Mesh Tests" begin
        @testset "Error Handling" begin
            elements = Element{Float64}[]
            connectivity = Connectivity([[1, 2, 3], [4, 5, 6]])
            levels = [1]
            is_leaf = [true]
            degrees = [1]

            @test_throws ArgumentError Mesh(
                elements,
                connectivity,
                levels,
                is_leaf,
                degrees,
                connectivity,
                true,
            )
            @test_throws MethodError Mesh()
        end

        @testset "Correct Outputs" begin
            nodes = [Node(1, [0.0, 1.0], true), Node(2, [1.0, 0.0], true)]
            basis_functions = [BasisFunction(1, x -> x, true)]
            elements = [
                Element(
                    nodes,
                    basis_functions,
                    [1],
                    [1],
                    nothing,
                    1,
                    TensorProductMask([sprand(3, 3, 0.5)]),
                    LocationMatrix(sprand(3, 3, 0.2)),
                    0.0,
                    Dict(),
                ),
            ]
            connectivity = Connectivity([[1, 2, 3], [4, 5, 6]])
            levels = [1]
            is_leaf = [true]
            degrees = [1]

            mesh =
                Mesh(elements, connectivity, levels, is_leaf, degrees, connectivity, true)
            @test mesh.elements == elements
            @test mesh.connectivity == connectivity
            @test mesh.levels == levels
            @test mesh.is_leaf == is_leaf
            @test mesh.degrees == degrees
            @test mesh.parallel == true
        end
    end
end
