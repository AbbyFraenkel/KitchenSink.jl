using Test
using LinearAlgebra, StaticArrays

using KitchenSink.IntergridOperators

@testset "IntergridOperators" begin
	# Helper functions to create mock objects for testing
	function create_mock_point(coords::Vector{Float64})
		return KSPoint{Float64, length(coords)}(SVector{length(coords), Float64}(coords), 1.0)
	end

	function create_mock_element(id::Int, points::Vector{KSPoint{Float64, N}}) where N
		basis_functions = [KSBasisFunction(i, x -> x[1]^(i - 1)) for i in 1:length(points)]
		return KSElement{Float64, N}(id, points, basis_functions, nothing, nothing, nothing, 1, 2, 0.0, points, [zeros(length(points), length(points)) for _ in 1:N])
	end

	function create_mock_mesh(elements::Vector{KSElement{Float64, N}}) where N
		return KSMesh{Float64, N}(elements, [trues(3, 3) for _ in 1:length(elements)], [Dict(i => i for i in 1:3) for _ in 1:length(elements)], [], 0.0)
	end

	@testset "get_active_indices" begin
		points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		element = create_mock_element(1, points)
		mesh = create_mock_mesh([element])

		active_indices = IntergridOperators.get_active_indices(mesh, element)
		@test length(active_indices) == 3
		@test all(1 .<= active_indices .<= 3)
	end

	@testset "prolongate!" begin
		coarse_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		fine_points = [create_mock_point([0.0, 0.0]), create_mock_point([0.5, 0.0]), create_mock_point([1.0, 0.0]),
			create_mock_point([0.0, 0.5]), create_mock_point([0.5, 0.5]), create_mock_point([1.0, 0.5]),
			create_mock_point([0.0, 1.0]), create_mock_point([0.5, 1.0]), create_mock_point([1.0, 1.0])]

		coarse_element = create_mock_element(1, coarse_points)
		fine_elements = [create_mock_element(i, fine_points[((i - 1) * 4 + 1):(i * 4)]) for i in 1:4]

		coarse_mesh = create_mock_mesh([coarse_element])
		fine_mesh = create_mock_mesh(fine_elements)

		coarse_solution = [1.0, 2.0, 3.0, 4.0]
		fine_solution = zeros(16)

		prolongate!(fine_solution, coarse_solution, coarse_mesh, fine_mesh)

		@test length(fine_solution) == 16
		@test all(fine_solution .>= 1.0)
		@test all(fine_solution .<= 4.0)

		@inferred prolongate!(fine_solution, coarse_solution, coarse_mesh, fine_mesh)
	end

	@testset "restrict!" begin
		coarse_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		fine_points = [create_mock_point([0.0, 0.0]), create_mock_point([0.5, 0.0]), create_mock_point([1.0, 0.0]),
			create_mock_point([0.0, 0.5]), create_mock_point([0.5, 0.5]), create_mock_point([1.0, 0.5]),
			create_mock_point([0.0, 1.0]), create_mock_point([0.5, 1.0]), create_mock_point([1.0, 1.0])]

		coarse_element = create_mock_element(1, coarse_points)
		fine_elements = [create_mock_element(i, fine_points[((i - 1) * 4 + 1):(i * 4)]) for i in 1:4]

		coarse_mesh = create_mock_mesh([coarse_element])
		fine_mesh = create_mock_mesh(fine_elements)

		fine_solution = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
		coarse_solution = zeros(4)

		restrict!(coarse_solution, fine_solution, fine_mesh, coarse_mesh)

		@test length(coarse_solution) == 4
		@test all(coarse_solution .>= 1.0)
		@test all(coarse_solution .<= 16.0)

		@inferred restrict!(coarse_solution, fine_solution, fine_mesh, coarse_mesh)
	end

	@testset "interpolate_between_meshes!" begin
		source_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		target_points = [create_mock_point([0.25, 0.25]), create_mock_point([0.75, 0.25]), create_mock_point([0.25, 0.75]), create_mock_point([0.75, 0.75])]

		source_element = create_mock_element(1, source_points)
		target_element = create_mock_element(1, target_points)

		source_mesh = create_mock_mesh([source_element])
		target_mesh = create_mock_mesh([target_element])

		source_solution = [1.0, 2.0, 3.0, 4.0]
		target_solution = zeros(4)

		interpolate_between_meshes!(target_solution, source_solution, source_mesh, target_mesh)

		@test length(target_solution) == 4
		@test all(target_solution .>= 1.0)
		@test all(target_solution .<= 4.0)

		@inferred interpolate_between_meshes!(target_solution, source_solution, source_mesh, target_mesh)
	end

	@testset "find_parent_element" begin
		coarse_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		fine_points = [create_mock_point([0.25, 0.25]), create_mock_point([0.75, 0.25]), create_mock_point([0.25, 0.75]), create_mock_point([0.75, 0.75])]

		coarse_element = create_mock_element(1, coarse_points)
		fine_element = create_mock_element(2, fine_points)

		coarse_mesh = create_mock_mesh([coarse_element])

		parent = find_parent_element(fine_element, coarse_mesh)
		@test parent == coarse_element

		outside_points = [create_mock_point([1.5, 1.5]), create_mock_point([2.0, 1.5]), create_mock_point([1.5, 2.0]), create_mock_point([2.0, 2.0])]
		outside_element = create_mock_element(3, outside_points)
		@test find_parent_element(outside_element, coarse_mesh) === nothing

		@inferred find_parent_element(fine_element, coarse_mesh)
	end

	@testset "find_child_elements" begin
		coarse_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		fine_points1 = [create_mock_point([0.0, 0.0]), create_mock_point([0.5, 0.0]), create_mock_point([0.0, 0.5]), create_mock_point([0.5, 0.5])]
		fine_points2 = [create_mock_point([0.5, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.5, 0.5]), create_mock_point([1.0, 0.5])]
		fine_points3 = [create_mock_point([0.0, 0.5]), create_mock_point([0.5, 0.5]), create_mock_point([0.0, 1.0]), create_mock_point([0.5, 1.0])]
		fine_points4 = [create_mock_point([0.5, 0.5]), create_mock_point([1.0, 0.5]), create_mock_point([0.5, 1.0]), create_mock_point([1.0, 1.0])]

		coarse_element = create_mock_element(1, coarse_points)
		fine_elements = [
			create_mock_element(2, fine_points1),
			create_mock_element(3, fine_points2),
			create_mock_element(4, fine_points3),
			create_mock_element(5, fine_points4)
		]

		fine_mesh = create_mock_mesh(fine_elements)

		children = find_child_elements(coarse_element, fine_mesh)
		@test length(children) == 4
		@test all(child in fine_elements for child in children)

		no_children_element = create_mock_element(6, [create_mock_point([2.0, 2.0]), create_mock_point([3.0, 2.0]), create_mock_point([2.0, 3.0]), create_mock_point([3.0, 3.0])])
		@test isempty(find_child_elements(no_children_element, fine_mesh))

		@inferred find_child_elements(coarse_element, fine_mesh)
	end

	@testset "find_containing_element" begin
		points1 = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		points2 = [create_mock_point([1.0, 0.0]), create_mock_point([2.0, 0.0]), create_mock_point([1.0, 1.0]), create_mock_point([2.0, 1.0])]
		points3 = [create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0]), create_mock_point([0.0, 2.0]), create_mock_point([1.0, 2.0])]

		elements = [
			create_mock_element(1, points1),
			create_mock_element(2, points2),
			create_mock_element(3, points3)
		]

		mesh = create_mock_mesh(elements)

		@test find_containing_element(create_mock_point([0.5, 0.5]), mesh) == elements[1]
		@test find_containing_element(create_mock_point([1.5, 0.5]), mesh) == elements[2]
		@test find_containing_element(create_mock_point([0.5, 1.5]), mesh) == elements[3]

		@test find_containing_element(create_mock_point([2.5, 2.5]), mesh) === nothing

		@inferred find_containing_element(create_mock_point([0.5, 0.5]), mesh)
	end

	@testset "interpolate_element" begin
		source_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		target_points = [create_mock_point([0.25, 0.25]), create_mock_point([0.75, 0.25]), create_mock_point([0.25, 0.75]), create_mock_point([0.75, 0.75])]

		source_element = create_mock_element(1, source_points)
		target_element = create_mock_element(2, target_points)

		source_solution = [1.0, 2.0, 3.0, 4.0]

		target_solution = interpolate_element(source_element, target_element, source_solution)

		@test length(target_solution) == 4
		@test all(target_solution .>= 1.0)
		@test all(target_solution .<= 4.0)

		@inferred interpolate_element(source_element, target_element, source_solution)
	end

	@testset "restrict_element" begin
		fine_points = [create_mock_point([0.0, 0.0]), create_mock_point([0.5, 0.0]), create_mock_point([0.0, 0.5]), create_mock_point([0.5, 0.5])]
		coarse_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]

		fine_element = create_mock_element(1, fine_points)
		coarse_element = create_mock_element(2, coarse_points)

		fine_solution = [1.0, 2.0, 3.0, 4.0]

		coarse_solution = restrict_element(fine_element, coarse_element, fine_solution)

		@test length(coarse_solution) == 4
		@test all(coarse_solution .>= 1.0)
		@test all(coarse_solution .<= 4.0)

		@inferred restrict_element(fine_element, coarse_element, fine_solution)
	end

	@testset "interpolate_point" begin
		element_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		element = create_mock_element(1, element_points)
		point = create_mock_point([0.5, 0.5])

		solution = [1.0, 2.0, 3.0, 4.0]

		interpolated_value = interpolate_point(point, element, solution)

		@test interpolated_value >= 1.0
		@test interpolated_value <= 4.0

		@inferred interpolate_point(point, element, solution)
	end

	@testset "compute_interpolation_weights" begin
		element_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		element = create_mock_element(1, element_points)
		point = create_mock_point([0.5, 0.5])

		weights = compute_interpolation_weights(point, element)

		@test length(weights) == 4
		@test all(weights .>= 0.0)
		@test all(weights .<= 1.0)
		@test isapprox(sum(weights), 1.0, atol = 1e-10)

		@inferred compute_interpolation_weights(point, element)
	end

	@testset "is_parent" begin
		coarse_points = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		fine_points = [create_mock_point([0.25, 0.25]), create_mock_point([0.75, 0.25]), create_mock_point([0.25, 0.75]), create_mock_point([0.75, 0.75])]

		coarse_element = create_mock_element(1, coarse_points)
		fine_element = create_mock_element(2, fine_points)

		@test is_parent(coarse_element, fine_element) == true

		non_parent_points = [create_mock_point([1.0, 1.0]), create_mock_point([2.0, 1.0]), create_mock_point([1.0, 2.0]), create_mock_point([2.0, 2.0])]
		non_parent_element = create_mock_element(3, non_parent_points)
		@test is_parent(non_parent_element, fine_element) == false

		@inferred is_parent(coarse_element, fine_element)
	end

	@testset "validate_mesh_compatibility" begin
		points1 = [create_mock_point([0.0, 0.0]), create_mock_point([1.0, 0.0]), create_mock_point([0.0, 1.0]), create_mock_point([1.0, 1.0])]
		points2 = [create_mock_point([0.5, 0.5]), create_mock_point([1.5, 0.5]), create_mock_point([0.5, 1.5]), create_mock_point([1.5, 1.5])]

		element1 = create_mock_element(1, points1)
		element2 = create_mock_element(2, points2)

		mesh1 = create_mock_mesh([element1])
		mesh2 = create_mock_mesh([element2])

		@test_nowarn validate_mesh_compatibility(mesh1, mesh2)

		points3 = [create_mock_point([0.0, 0.0, 0.0]), create_mock_point([1.0, 0.0, 0.0]), create_mock_point([0.0, 1.0, 0.0]), create_mock_point([1.0, 1.0, 0.0])]
		element3 = create_mock_element(3, points3)
		mesh3 = create_mock_mesh([element3])

		@test_throws ArgumentError validate_mesh_compatibility(mesh1, mesh3)

		@inferred validate_mesh_compatibility(mesh1, mesh2)
	end

	@testset "Performance tests" begin
		n = 1000
		coarse_points = [create_mock_point([Float64(i), Float64(j)]) for i in 0:9 for j in 0:9]
		fine_points = [create_mock_point([Float64(i) / 10, Float64(j) / 10]) for i in 0:99 for j in 0:99]

		coarse_elements = [create_mock_element(i, coarse_points[((i - 1) * 4 + 1):(i * 4)]) for i in 1:25]
		fine_elements = [create_mock_element(i, fine_points[((i - 1) * 4 + 1):(i * 4)]) for i in 1:2500]

		coarse_mesh = create_mock_mesh(coarse_elements)
		fine_mesh = create_mock_mesh(fine_elements)

		coarse_solution = rand(100)
		fine_solution = rand(10000)

		@time prolongate!(fine_solution, coarse_solution, coarse_mesh, fine_mesh)
		@time restrict!(coarse_solution, fine_solution, fine_mesh, coarse_mesh)
		@time interpolate_between_meshes!(fine_solution, coarse_solution, coarse_mesh, fine_mesh)
	end
end
