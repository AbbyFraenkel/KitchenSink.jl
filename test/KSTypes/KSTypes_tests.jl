using Test
using StaticArrays, SparseArrays, LinearAlgebra
using KitchenSink.KSTypes

include("../test_utils.jl")

function create_mock_coordinate_system(system_type::Symbol; inactive::Bool = false)
	if system_type == :Cartesian
		return KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))

	elseif system_type == :Polar
		if inactive
			return KSPolarCoordinates((0.0, 1.0), nothing)
		else
			return KSPolarCoordinates((0.0, 1.0), (0.0, π))
		end
	elseif system_type == :Cylindrical
		if inactive
			return KSCylindricalCoordinates((0.0, 1.0), nothing, (0.0, 1.0))
		else
			return KSCylindricalCoordinates((0.0, 1.0), (0.0, 2π), (0.0, 1.0))
		end
	elseif system_type == :Spherical
		if inactive
			return KSSphericalCoordinates((0.0, 1.0), (0.0, π), nothing)
		else
			return KSSphericalCoordinates((0.0, 1.0), (0.0, π), (0.0, 2π))
		end
	end
end

@testset "KSTypes" begin
	@testset "Abstract Types" begin
		@test isabstracttype(AbstractKSCell)
		@test isabstracttype(AbstractKSMesh)
		@test isabstracttype(AbstractKSProblem)
		@test isabstracttype(AbstractKSCoordinateSystem)
		@test isabstracttype(AbstractKSBoundaryCondition)
		@test isabstracttype(AbstractKSBasisFunction)
		@test isabstracttype(AbstractKSSolver)
		@test isabstracttype(AbstractKSOptimizationProblem)
		@test isabstracttype(AbstractKSLinearSolver)
		@test isabstracttype(AbstractStandardKSCell)
	end
	#
	# AffineTransform Tests
	@testset "Transforms" begin
		forward_map = x -> x^2
		jacobian = x -> 2x
		inverse_map = x -> sqrt(x)
		@testset "AffineTransform Tests" begin
			# Valid input
			matrix = [1.0 0.0; 0.0 1.0]
			translation = [1.0, 2.0]
			affine = AffineTransform(matrix, translation)
			@test affine.matrix == matrix
			@test affine.translation == translation

			# Invalid input: non-square matrix
			matrix = [1.0 0.0 0.0; 0.0 1.0 0.0]
			translation = [1.0, 2.0, 3.0]
			@test_throws DimensionMismatch AffineTransform(matrix, translation)

			# Invalid input: length mismatch between translation and matrix dimensions
			matrix = [1.0 0.0; 0.0 1.0]
			translation = [1.0, 2.0, 3.0]
			@test_throws DimensionMismatch AffineTransform(matrix, translation)
		end

		# NonlinearTransform Tests
		@testset "NonlinearTransform Tests" begin
			# Valid input
			nonlinear = NonlinearTransform(;
				forward_map = forward_map,
				inverse_map = inverse_map,
				jacobian = jacobian,
			)
			@test nonlinear.inverse_map === inverse_map

			# Invalid input: forward map is not callable
			@test_throws MethodError NonlinearTransform(123, jacobian)

			# Invalid input: jacobian is not callable
			@test_throws MethodError NonlinearTransform(forward_map, 456)

			# Invalid input: inverse map is not callable
			@test_throws MethodError NonlinearTransform(forward_map, jacobian, 789)
		end

		# CompositeTransform Tests
		@testset "CompositeTransform Tests" begin
			# Valid input
			T1 = AffineTransform([1.0 0.0; 0.0 1.0], [0.0, 1.0])
			T2 = NonlinearTransform(;
				forward_map = forward_map,
				inverse_map = inverse_map,
				jacobian = jacobian,
			)
			composite = CompositeTransform(; transforms = [T1, T2])
			@test length(composite.transforms) == 2
			@test composite.transforms[1] === T1
			@test composite.transforms[2] === T2

			# Invalid input: empty transform list
			@test_throws MethodError CompositeTransform([])
		end

		# MultiLevelTransform Tests
		@testset "MultiLevelTransform Tests" begin
			# Valid input
			T1 = AffineTransform([1.0 0.0; 0.0 1.0], [0.0, 1.0])
			T2 = NonlinearTransform(;
				forward_map = forward_map,
				inverse_map = inverse_map,
				jacobian = jacobian,
			)
			multi_level = MultiLevelTransform(;
				base_transform = T1,
				level_transforms = Vector{AbstractKSTransform}([T2]),
			)
			@test multi_level.base_transform === T1
			@test multi_level.level_transforms == [T2]

			# Invalid input: empty level transforms list
			@test_throws ArgumentError MultiLevelTransform(
				base_transform = T1,
				level_transforms = Vector{AbstractKSTransform}([]),
			)
		end

		# PrimitiveLevelSet Tests
		@testset "PrimitiveLevelSet Tests" begin
			# Valid input
			func = x -> x^2
			gradient = x -> 2x
			primitive = PrimitiveLevelSet(; func = func, gradient = gradient)
			@test primitive.func === func
			@test primitive.gradient === gradient

			# Valid input: no gradient provided
			primitive = PrimitiveLevelSet(; func = func)
			@test primitive.gradient === nothing

			# Invalid input: func is not callable
			@test_throws MethodError PrimitiveLevelSet(123)

			# Invalid input: gradient is not callable
			@test_throws MethodError PrimitiveLevelSet(func, 456)
		end

		@testset "CompositeLevelSet Tests" begin
			func = x -> x^2
			gradient = x -> 2x
			primitive1 = PrimitiveLevelSet(; func = func, gradient = gradient)
			primitive2 = PrimitiveLevelSet(; func = func, gradient = nothing)
			primitive3 = PrimitiveLevelSet(; func = func, gradient = gradient)

			op1 = (x, y) -> x + y
			op2 = (x, y) -> x * y

			# Valid case: two primitives, one operation
			cls1 = CompositeLevelSet(;
				operations = [op1],
				primitives = AbstractLevelSetFunction[primitive1, primitive2],
			)
			@test length(cls1.operations) == 1
			@test length(cls1.primitives) == 2

			# Valid case: three primitives, two operations
			cls2 = CompositeLevelSet(;
				operations = [op1, op2],
				primitives = AbstractLevelSetFunction[primitive1, primitive2, primitive3],
			)
			@test length(cls2.operations) == 2
			@test length(cls2.primitives) == 3

			# Valid case: single operation and a vector of primitives
			cls3 = CompositeLevelSet(;
				operations = [op1],
				primitives = AbstractLevelSetFunction[primitive1, primitive2],
			)
			@test length(cls3.operations) == 1
			@test length(cls3.primitives) == 2

			# Error case: empty operations
			@test_throws ArgumentError CompositeLevelSet(
				operations = [],
				primitives = AbstractLevelSetFunction[primitive1, primitive2],
			)

			# Error case: empty primitives
			@test_throws ArgumentError CompositeLevelSet(
				operations = [op1],
				primitives = AbstractLevelSetFunction[],
			)

			# Error case: operations and primitives mismatch
			@test_throws DimensionMismatch CompositeLevelSet(
				operations = [op1, op2],
				primitives = AbstractLevelSetFunction[primitive1, primitive2],
			)
			@test_throws MethodError CompositeLevelSet(
				[],
				primitives = AbstractLevelSetFunction[primitive1, primitive2],
			)
		end

		# Test for DomainMapping
		@testset "DomainMapping Tests" begin
			# Create dummy types for AbstractKSTransform and AbstractKSCoordinateSystem
			struct DummyTransform <: AbstractKSTransform end
			struct DummyCoordinateSystem <: AbstractKSCoordinateSystem end

			forward_transform = DummyTransform()
			inverse_transform = DummyTransform()
			physical_system = DummyCoordinateSystem()
			fictitious_system = DummyCoordinateSystem()

			# Valid DomainMapping creation
			domain_mapping = DomainMapping(;
				forward_transform = forward_transform,
				inverse_transform = inverse_transform,
				fictitious_system = fictitious_system,
				physical_system = physical_system,
			)

			@test domain_mapping.forward_transform === forward_transform
			@test domain_mapping.inverse_transform === inverse_transform
			@test domain_mapping.physical_system === physical_system
			@test domain_mapping.fictitious_system === fictitious_system

			# Invalid input: wrong types for forward_transform
			@test_throws TypeError DomainMapping(
				forward_transform = 123,
				inverse_transform = inverse_transform,
				physical_system = physical_system,
				fictitious_system = fictitious_system,
			)

			# Invalid input: wrong types for inverse_transform
			@test_throws TypeError DomainMapping(
				forward_transform = forward_transform,
				inverse_transform = 123,
				physical_system = physical_system,
				fictitious_system = fictitious_system,
			)

			# Invalid input: wrong types for physical_system
			@test_throws TypeError DomainMapping(
				forward_transform = forward_transform,
				inverse_transform = inverse_transform,
				physical_system = 123,
				fictitious_system = fictitious_system,
			)

			# Invalid input: wrong types for fictitious_system
			@test_throws TypeError DomainMapping(
				forward_transform = forward_transform,
				inverse_transform = inverse_transform,
				physical_system = physical_system,
				fictitious_system = 123,
			)
		end

		# TransformCache Tests
	end
	# StandardSpectralProperties Tests
	spectral_props1 = StandardSpectralProperties(;
		p = p1,
		continuity_order = 1,
		nodes_with_boundary = x_with_boundary1,
		nodes_interior = x1,
		weights_with_boundary = w_with_boundary1,
		weights_interior = w1,
		differentiation_matrix_with_boundary = D_with_boundary_1,
		differentiation_matrix_interior = D_1,
		quadrature_matrix = quad1,
	)

	spectral_props2 = StandardSpectralProperties(;
		p = p2,
		continuity_order = 1,
		nodes_with_boundary = x_with_boundary2,
		nodes_interior = x2,
		weights_with_boundary = w_with_boundary2,
		weights_interior = w2,
		differentiation_matrix_with_boundary = D_with_boundary_2,
		differentiation_matrix_interior = D_2,
		quadrature_matrix = quad2,
	)

	spectral_props3 = StandardSpectralProperties(;
		p = p3,
		continuity_order = 1,
		nodes_with_boundary = x_with_boundary3,
		nodes_interior = x3,
		weights_with_boundary = w_with_boundary3,
		weights_interior = w3,
		differentiation_matrix_with_boundary = D_with_boundary_3,
		differentiation_matrix_interior = D_3,
		quadrature_matrix = quad3,
	)

	# Test for StandardSpectralProperties
	@testset "StandardSpectralProperties Tests" begin
		@testset "Valid StandardSpectralProperties Creation" begin
			@test spectral_props1.p == 3
			@test spectral_props1.nodes_with_boundary == x_with_boundary1
			@test spectral_props1.nodes_interior == x1
			@test spectral_props1.weights_with_boundary == w_with_boundary1
			@test spectral_props1.weights_interior == w1
			@test spectral_props1.differentiation_matrix_with_boundary == D_with_boundary_1
			@test spectral_props1.differentiation_matrix_interior == D_1
			@test spectral_props1.quadrature_matrix == quad1

			@test spectral_props2.p == 8
			@test spectral_props2.nodes_with_boundary == x_with_boundary2
			@test spectral_props2.nodes_interior == x2
			@test spectral_props2.weights_with_boundary == w_with_boundary2
			@test spectral_props2.weights_interior == w2
			@test spectral_props2.differentiation_matrix_with_boundary == D_with_boundary_2
			@test spectral_props2.differentiation_matrix_interior == D_2
			@test spectral_props2.quadrature_matrix == quad2

			@test spectral_props3.p == 23
			@test spectral_props3.nodes_with_boundary == x_with_boundary3
			@test spectral_props3.nodes_interior == x3
			@test spectral_props3.weights_with_boundary == w_with_boundary3
			@test spectral_props3.weights_interior == w3
			@test spectral_props3.differentiation_matrix_with_boundary == D_with_boundary_3
			@test spectral_props3.differentiation_matrix_interior == D_3
			@test spectral_props3.quadrature_matrix == quad3
		end

		@testset "Invalid StandardSpectralProperties Creation" begin
			# Mismatched lengths: nodes_with_boundary is empty
			@test_throws DimensionMismatch StandardSpectralProperties(
				p = p2,
				continuity_order = 1,
				nodes_with_boundary = [-1, 1],
				nodes_interior = x2,
				weights_with_boundary = w_with_boundary2,
				weights_interior = w2,
				differentiation_matrix_with_boundary = D_with_boundary_2,
				differentiation_matrix_interior = D_2,
				quadrature_matrix = quad2,
			)

			# Mismatched lengths: weights_with_boundary has incorrect length
			@test_throws DimensionMismatch StandardSpectralProperties(
				p = 3,
				continuity_order = 1,
				nodes_with_boundary = [0.0, 1.0, 2.0],
				nodes_interior = [1.0],
				weights_with_boundary = [1.0, 1.0],  # Mismatched length
				weights_interior = [1.0],
				differentiation_matrix_with_boundary = [1.0 0.0; 0.0 1.0],
				differentiation_matrix_interior = [1.0;;],
				quadrature_matrix = quad1,
			)
		end
	end

	# @testset "TransformCache Tests" begin
	# 	# Create a valid cache
	# 	operator_cache = Dict(:op1 => x -> x^2, :op2 => x -> x + 1)
	# 	matrix_cache = Dict{Symbol, Matrix{Float64}}(:mat1 => [1.0 0.0; 0.0 1.0], :mat2 => [2.0 0.0; 0.0 2.0])

	# 	cache = TransformCache(operator_cache, matrix_cache)
	# 	@test cache.operator_cache == operator_cache
	# 	@test cache.matrix_cache == matrix_cache

	# 	# Invalid input: operator_cache not a Dict
	# 	@test_throws MethodError TransformCache(123, matrix_cache)

	# 	# Invalid input: matrix_cache not a Dict
	# 	@test_throws MethodError TransformCache(operator_cache, 123)

	# 	# Invalid input: operator_cache with invalid function
	# 	@test_throws MethodError TransformCache(Dict(:op1 => 123), matrix_cache)

	# 	# Invalid input: matrix_cache with non-matrix entry
	# 	@test_throws MethodError TransformCache(operator_cache, Dict{Symbol, Matrix{Float64}}(:mat1 => 123))
	# end
	# StandardKSCell Tests
	@testset "StandardKSCell Tests" begin
		level_1 = 1

		@testset "Valid StandardKSCell Creation" begin
			# Test 1D case with tuple input for p1
			@testset "Valid StandardKSCell Creation 1: 1D Case" begin
				standard_ks_cell_1 = StandardKSCell(;
					p = (p1,),
					level = level_1,
					continuity_order = (1,),
					nodes_with_boundary = (x_with_boundary1,),
					nodes_interior = (x1,),
					weights_with_boundary = (w_with_boundary1,),
					weights_interior = (w1,),
					differentiation_matrix_with_boundary = (D_with_boundary_1,),
					differentiation_matrix_interior = (D_1,),
					quadrature_matrix = (quad1,),
				)
				@test standard_ks_cell_1.level == 1
				@test standard_ks_cell_1.p == (p1,)
				@test standard_ks_cell_1.nodes_with_boundary == (x_with_boundary1,)
				@test standard_ks_cell_1.nodes_interior == (x1,)
				@test standard_ks_cell_1.weights_with_boundary == (w_with_boundary1,)
				@test standard_ks_cell_1.weights_interior == (w1,)
				@test standard_ks_cell_1.differentiation_matrix_with_boundary ==
					(D_with_boundary_1,)
				@test standard_ks_cell_1.differentiation_matrix_interior == (D_1,)
				@test standard_ks_cell_1.quadrature_matrix == (quad1,)
			end

			# Test 2D case
			@testset "Valid StandardKSCell Creation 2: 2D Case" begin
				standard_ks_cell_2D = StandardKSCell(;
					p = (p1, p2),
					level = 1,
					continuity_order = (1, 1),
					nodes_with_boundary = (x_with_boundary1, x_with_boundary2),
					nodes_interior = (x1, x2),
					weights_with_boundary = (w_with_boundary1, w_with_boundary2),
					weights_interior = (w1, w2),
					differentiation_matrix_with_boundary = (
						D_with_boundary_1,
						D_with_boundary_2,
					),
					differentiation_matrix_interior = (D_1, D_2),
					quadrature_matrix = (quad1, quad2),
				)
				@test standard_ks_cell_2D.level == 1
				@test standard_ks_cell_2D.p == (p1, p2)
				@test standard_ks_cell_2D.nodes_with_boundary ==
					(x_with_boundary1, x_with_boundary2)
				@test standard_ks_cell_2D.nodes_interior == (x1, x2)
				@test standard_ks_cell_2D.weights_with_boundary ==
					(w_with_boundary1, w_with_boundary2)
				@test standard_ks_cell_2D.weights_interior == (w1, w2)
				@test standard_ks_cell_2D.differentiation_matrix_with_boundary ==
					(D_with_boundary_1, D_with_boundary_2)
				@test standard_ks_cell_2D.differentiation_matrix_interior == (D_1, D_2)
				@test standard_ks_cell_2D.quadrature_matrix == (quad1, quad2)
			end

			@testset "Valid StandardKSCell Creation 3: 3D Case with p1, p2, and p3" begin
				standard_ks_cell_3D = StandardKSCell(;
					p = (p1, p2, p3),
					level = 1,
					continuity_order = (1, 1, 1),
					nodes_with_boundary = (
						x_with_boundary1,
						x_with_boundary2,
						x_with_boundary3,
					),
					nodes_interior = (x1, x2, x3),
					weights_with_boundary = (
						w_with_boundary1,
						w_with_boundary2,
						w_with_boundary3,
					),
					weights_interior = (w1, w2, w3),
					differentiation_matrix_with_boundary = (
						D_with_boundary_1,
						D_with_boundary_2,
						D_with_boundary_3,
					),
					differentiation_matrix_interior = (D_1, D_2, D_3),
					quadrature_matrix = (quad1, quad2, quad3),
				)
				@test standard_ks_cell_3D.level == 1
				@test standard_ks_cell_3D.p == (p1, p2, p3)
				@test standard_ks_cell_3D.nodes_with_boundary ==
					(x_with_boundary1, x_with_boundary2, x_with_boundary3)
				@test standard_ks_cell_3D.nodes_interior == (x1, x2, x3)
				@test standard_ks_cell_3D.weights_with_boundary ==
					(w_with_boundary1, w_with_boundary2, w_with_boundary3)
				@test standard_ks_cell_3D.weights_interior == (w1, w2, w3)
				@test standard_ks_cell_3D.differentiation_matrix_with_boundary ==
					(D_with_boundary_1, D_with_boundary_2, D_with_boundary_3)
				@test standard_ks_cell_3D.differentiation_matrix_interior == (D_1, D_2, D_3)
				@test standard_ks_cell_3D.quadrature_matrix == (quad1, quad2, quad3)
			end
		end
		@testset "StandardKSCell Error Handling" begin

			# Test Case 2: Empty `p`
			@testset "Empty p" begin
				@test_throws ArgumentError StandardKSCell(
					p = (),
					level = 1,
					continuity_order = (),
					nodes_with_boundary = (),
					nodes_interior = (),
					weights_with_boundary = (),
					weights_interior = (),
					differentiation_matrix_with_boundary = (),
					differentiation_matrix_interior = (),
					quadrature_matrix = (),
				)
			end

			# Test Case 3: Mismatched dimensions
			@testset "Mismatched Dimensions" begin
				@test_throws DimensionMismatch StandardKSCell(
					p = (p1,),
					level = level_1,
					continuity_order = (1,),
					nodes_with_boundary = (x_with_boundary1,),
					nodes_interior = (x1,),
					weights_with_boundary = (w_with_boundary1,),
					weights_interior = (w1,),
					differentiation_matrix_with_boundary = (D_with_boundary_1,),
					differentiation_matrix_interior = (D_1,),
					quadrature_matrix = (quad2,),
				)
			end
		end
	end

	# Test Suite for KSCell and KSMesh
	@testset "KSCell Tests" begin
		level_1 = 1

		# 1D Case
		@testset "1D Case" begin
			ks_cell_1D = KSCell(;
				id = 1,
				p = (p1,),
				level = level_1,
				continuity_order = (1,),
				standard_cell_key = ((p1,), level_1),
				neighbors = Dict(:dim1_neg => -1, :dim1_pos => 2),
				node_map = Dict((1,) => 1),
				tensor_product_mask = (fill(true, p1),),
				boundary_connectivity = Dict(:dim1_neg => -1, :dim1_pos => 2),
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			)

			@test ks_cell_1D.id == 1
			@test ks_cell_1D.p == (p1,)
			@test ks_cell_1D.level == level_1
			@test ks_cell_1D.standard_cell_key == ((p1,), level_1)
			@test ks_cell_1D.neighbors == Dict(:dim1_neg => -1, :dim1_pos => 2)
			@test ks_cell_1D.node_map == Dict((1,) => 1)
			@test ks_cell_1D.tensor_product_mask == (fill(true, p1),)
			@test ks_cell_1D.boundary_connectivity == Dict(:dim1_neg => -1, :dim1_pos => 2)
			@test ks_cell_1D.error_estimate == 0.1
			@test ks_cell_1D.legendre_decay_rate == 0.05
		end

		# 2D Case
		@testset "2D Case" begin
			ks_cell_2D = KSCell(;
				id = 1,
				p = (p1, p2),
				level = level_1,
				continuity_order = (1, 1),
				standard_cell_key = ((p1, p2), level_1),
				neighbors = Dict(
					:dim1_neg => -1,
					:dim1_pos => 2,
					:dim2_neg => -1,
					:dim2_pos => 2,
				),
				node_map = Dict((1, 1) => 1),
				tensor_product_mask = (fill(true, p1), fill(true, p2)),
				boundary_connectivity = Dict(
					:dim1_neg => -1,
					:dim1_pos => 2,
					:dim2_neg => -1,
					:dim2_pos => 2,
				),
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			)

			@test ks_cell_2D.id == 1
			@test ks_cell_2D.p == (p1, p2)
			@test ks_cell_2D.level == level_1
			@test ks_cell_2D.standard_cell_key == ((p1, p2), level_1)
			@test ks_cell_2D.neighbors ==
				Dict(:dim1_neg => -1, :dim1_pos => 2, :dim2_neg => -1, :dim2_pos => 2)
			@test ks_cell_2D.node_map == Dict((1, 1) => 1)
			@test ks_cell_2D.tensor_product_mask == (fill(true, p1), fill(true, p2))
			@test ks_cell_2D.boundary_connectivity ==
				Dict(:dim1_neg => -1, :dim1_pos => 2, :dim2_neg => -1, :dim2_pos => 2)
			@test ks_cell_2D.error_estimate == 0.1
			@test ks_cell_2D.legendre_decay_rate == 0.05
		end

		# 3D Case
		@testset "3D Case" begin
			ks_cell_3D = KSCell(;
				id = 1,
				p = (p1, p2, p3),
				level = level_1,
				continuity_order = (1, 1, 1),
				standard_cell_key = ((p1, p2, p3), level_1),
				neighbors = Dict(
					:dim1_neg => -1,
					:dim1_pos => 2,
					:dim2_neg => -1,
					:dim2_pos => 2,
					:dim3_neg => -1,
					:dim3_pos => 2,
				),
				node_map = Dict((1, 1, 1) => 1),
				tensor_product_mask = (fill(true, p1), fill(true, p2), fill(true, p3)),
				boundary_connectivity = Dict(
					:dim1_neg => -1,
					:dim1_pos => 2,
					:dim2_neg => -1,
					:dim2_pos => 2,
					:dim3_neg => -1,
					:dim3_pos => 2,
				),
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			)

			@test ks_cell_3D.id == 1
			@test ks_cell_3D.p == (p1, p2, p3)
			@test ks_cell_3D.level == level_1
			@test ks_cell_3D.standard_cell_key == ((p1, p2, p3), level_1)
			@test ks_cell_3D.neighbors == Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 2,
				:dim3_neg => -1,
				:dim3_pos => 2,
			)
			@test ks_cell_3D.node_map == Dict((1, 1, 1) => 1)
			@test ks_cell_3D.tensor_product_mask ==
				(fill(true, p1), fill(true, p2), fill(true, p3))
			@test ks_cell_3D.boundary_connectivity == Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 2,
				:dim3_neg => -1,
				:dim3_pos => 2,
			)
			@test ks_cell_3D.error_estimate == 0.1
			@test ks_cell_3D.legendre_decay_rate == 0.05
		end

		# 4D Case
		@testset "4D Case" begin
			ks_cell_4D = KSCell(;
				id = 1,
				p = (p1, p2, p3, p1),
				level = level_1,
				continuity_order = (1, 1, 1, 1),
				standard_cell_key = ((p1, p2, p3, p1), level_1),
				neighbors = Dict(
					:dim1_neg => -1,
					:dim1_pos => 2,
					:dim2_neg => -1,
					:dim2_pos => 2,
					:dim3_neg => -1,
					:dim3_pos => 2,
					:dim4_neg => -1,
					:dim4_pos => 2,
				),
				node_map = Dict((1, 1, 1, 1) => 1),
				tensor_product_mask = (
					fill(true, p1),
					fill(true, p2),
					fill(true, p3),
					fill(true, p1),
				),
				boundary_connectivity = Dict(
					:dim1_neg => -1,
					:dim1_pos => 2,
					:dim2_neg => -1,
					:dim2_pos => 2,
					:dim3_neg => -1,
					:dim3_pos => 2,
					:dim4_neg => -1,
					:dim4_pos => 2,
				),
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			)

			@test ks_cell_4D.id == 1
			@test ks_cell_4D.p == (p1, p2, p3, p1)
			@test ks_cell_4D.level == level_1
			@test ks_cell_4D.standard_cell_key == ((p1, p2, p3, p1), level_1)
			@test ks_cell_4D.neighbors == Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 2,
				:dim3_neg => -1,
				:dim3_pos => 2,
				:dim4_neg => -1,
				:dim4_pos => 2,
			)
			@test ks_cell_4D.node_map == Dict((1, 1, 1, 1) => 1)
			@test ks_cell_4D.tensor_product_mask ==
				(fill(true, p1), fill(true, p2), fill(true, p3), fill(true, p1))
			@test ks_cell_4D.boundary_connectivity == Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 2,
				:dim3_neg => -1,
				:dim3_pos => 2,
				:dim4_neg => -1,
				:dim4_pos => 2,
			)
			@test ks_cell_4D.error_estimate == 0.1
			@test ks_cell_4D.legendre_decay_rate == 0.05
		end

		# 7D Case
		@testset "7D Case" begin
			n4, n5, n6, n7 = 5, 6, 7, 8  # Example polynomial orders for 7 dimensions

			neighbors_7D = Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 2,
				:dim3_neg => -1,
				:dim3_pos => 2,
				:dim4_neg => -1,
				:dim4_pos => 2,
				:dim5_neg => -1,
				:dim5_pos => 2,
				:dim6_neg => -1,
				:dim6_pos => 2,
				:dim7_neg => -1,
				:dim7_pos => 2,
			)

			boundary_connectivity_7D = Dict(
				:dim1_neg => -1,
				:dim1_pos => 2,
				:dim2_neg => -1,
				:dim2_pos => 2,
				:dim3_neg => -1,
				:dim3_pos => 2,
				:dim4_neg => -1,
				:dim4_pos => 2,
				:dim5_neg => -1,
				:dim5_pos => 2,
				:dim6_neg => -1,
				:dim6_pos => 2,
				:dim7_neg => -1,
				:dim7_pos => 2,
			)

			ks_cell_7D = KSCell(;
				id = 1,
				p = (p1, p2, p3, n4, n5, n6, n7),
				level = level_1,
				continuity_order = (1, 1, 1, 1, 1, 1, 1),
				standard_cell_key = ((p1, p2, p3, n4, n5, n6, n7), level_1),
				neighbors = neighbors_7D,
				node_map = Dict((1, 1, 1, 1, 1, 1, 1) => 1),
				tensor_product_mask = (
					fill(true, p1),
					fill(true, p2),
					fill(true, p3),
					fill(true, n4),
					fill(true, n5),
					fill(true, n6),
					fill(true, n7),
				),
				boundary_connectivity = boundary_connectivity_7D,
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			)

			@test ks_cell_7D.id == 1
			@test ks_cell_7D.p == (p1, p2, p3, n4, n5, n6, n7)
			@test ks_cell_7D.level == level_1
			@test ks_cell_7D.standard_cell_key == ((p1, p2, p3, n4, n5, n6, n7), level_1)
			@test ks_cell_7D.neighbors == neighbors_7D
			@test ks_cell_7D.node_map == Dict((1, 1, 1, 1, 1, 1, 1) => 1)
			@test ks_cell_7D.tensor_product_mask == (
				fill(true, p1),
				fill(true, p2),
				fill(true, p3),
				fill(true, n4),
				fill(true, n5),
				fill(true, n6),
				fill(true, n7),
			)
			@test ks_cell_7D.boundary_connectivity == boundary_connectivity_7D
			@test ks_cell_7D.error_estimate == 0.1
			@test ks_cell_7D.legendre_decay_rate == 0.05
		end

		# No Standard Cell Key Given
		@testset "No Standard Cell Key Given" begin
			# 1D Case
			@testset "1D Case" begin
				ks_cell_1D = KSCell(;
					id = 1,
					p = (p1,),
					level = level_1,
					continuity_order = (1,),
					neighbors = Dict(:dim1_neg => -1, :dim1_pos => 2),
					node_map = Dict((1,) => 1),
					tensor_product_mask = (fill(true, p1),),
					boundary_connectivity = Dict(:dim1_neg => -1, :dim1_pos => 2),
					error_estimate = 0.1,
					legendre_decay_rate = 0.05,
				)

				@test ks_cell_1D.id == 1
				@test ks_cell_1D.p == (p1,)
				@test ks_cell_1D.level == level_1
				@test ks_cell_1D.standard_cell_key == ((p1,), level_1)  # Automatically set
				@test ks_cell_1D.neighbors == Dict(:dim1_neg => -1, :dim1_pos => 2)
				@test ks_cell_1D.node_map == Dict((1,) => 1)
				@test ks_cell_1D.tensor_product_mask == (fill(true, p1),)
				@test ks_cell_1D.boundary_connectivity ==
					Dict(:dim1_neg => -1, :dim1_pos => 2)
				@test ks_cell_1D.error_estimate == 0.1
				@test ks_cell_1D.legendre_decay_rate == 0.05
			end

			# 2D Case
			@testset "2D Case" begin
				ks_cell_2D = KSCell(;
					id = 1,
					p = (p1, p2),
					level = level_1,
					continuity_order = (1, 1),
					neighbors = Dict(
						:dim1_neg => -1,
						:dim1_pos => 2,
						:dim2_neg => -1,
						:dim2_pos => 2,
					),
					node_map = Dict((1, 1) => 1),
					tensor_product_mask = (fill(true, p1), fill(true, p2)),
					boundary_connectivity = Dict(
						:dim1_neg => -1,
						:dim1_pos => 2,
						:dim2_neg => -1,
						:dim2_pos => 2,
					),
					error_estimate = 0.1,
					legendre_decay_rate = 0.05,
				)

				@test ks_cell_2D.id == 1
				@test ks_cell_2D.p == (p1, p2)
				@test ks_cell_2D.level == level_1
				@test ks_cell_2D.standard_cell_key == ((p1, p2), level_1)  # Automatically set
				@test ks_cell_2D.neighbors ==
					Dict(:dim1_neg => -1, :dim1_pos => 2, :dim2_neg => -1, :dim2_pos => 2)
				@test ks_cell_2D.node_map == Dict((1, 1) => 1)
				@test ks_cell_2D.tensor_product_mask == (fill(true, p1), fill(true, p2))
				@test ks_cell_2D.boundary_connectivity ==
					Dict(:dim1_neg => -1, :dim1_pos => 2, :dim2_neg => -1, :dim2_pos => 2)
				@test ks_cell_2D.error_estimate == 0.1
				@test ks_cell_2D.legendre_decay_rate == 0.05
			end
		end
	end

	# Updated KSMesh constructor
	# KSMesh Tests
	@testset "KSMesh Tests" begin
		level_1 = 1

		# Create 1D KSCells
		ks_cells_1D = [
			KSCell(;
				id = i,
				p = (p1,),
				level = level_1,
				continuity_order = (1,),
				neighbors = Dict(:dim1_neg => i - 1, :dim1_pos => i + 1),
				node_map = Dict((1,) => i),
				tensor_product_mask = (fill(true, p1),),
				boundary_connectivity = Dict(
					:dim1_neg => i - 1, :dim1_pos => i + 1),
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			) for i in 1:3
		]

		# Create 2D KSCells
		ks_cells_2D = [
			KSCell(;
				id = i,
				p = (p1, p2),
				level = level_1,
				continuity_order = (1, 1),
				neighbors = Dict(
					:dim1_neg => i - 1,
					:dim1_pos => i + 1,
					:dim2_neg => -1,
					:dim2_pos => -1,
				),
				node_map = Dict((1, 1) => i),
				tensor_product_mask = (fill(true, p1), fill(true, p2)),
				boundary_connectivity = Dict(
					:dim1_neg => i - 1,
					:dim1_pos => i + 1,
					:dim2_neg => -1,
					:dim2_pos => -1,
				),
				error_estimate = 0.1,
				legendre_decay_rate = 0.05,
			) for i in 1:4
		]

		@testset "Basic Mesh Tests" begin
			# 1D KSMesh
			ks_mesh_1D = KSMesh(;
				cells = ks_cells_1D,
				global_error_estimate = 0.2,
				boundary_cells = Dict(:dim1_neg => [1], :dim1_pos => [3]),
				physical_domain = x -> true,
			)  # Assuming physical_domain is a function

			@test length(ks_mesh_1D.cells) == 3
			@test ks_mesh_1D.global_error_estimate == 0.2
			@test ks_mesh_1D.physical_cells == [1, 2, 3]
			@test ks_mesh_1D.boundary_cells == Dict(:dim1_neg => [1], :dim1_pos => [3])
			@test length(ks_mesh_1D.tensor_product_masks) == 3
			@test in(((p1,), level_1), ks_mesh_1D.standard_cell_keys)

			# Verify standard_spectral_properties mapping
			# @test haskey(ks_mesh_1D.standard_spectral_properties, ((p1,), level_1))
			# @test ks_mesh_1D.standard_spectral_properties[((p1,), level_1)] == compute_spectral_properties(((p1,), level_1))

			# 2D KSMesh
			ks_mesh_2D = KSMesh(;
				cells = ks_cells_2D,
				global_error_estimate = 0.2,
				boundary_cells = Dict(:dim1_neg => [1], :dim1_pos => [4]),
				physical_domain = x -> true,
			)  # Assumi/ng physical_domain is a function

			@test length(ks_mesh_2D.cells) == 4
			@test ks_mesh_2D.global_error_estimate == 0.2
			@test ks_mesh_2D.physical_cells == [1, 2, 3, 4]
			@test ks_mesh_2D.boundary_cells == Dict(:dim1_neg => [1], :dim1_pos => [4])
			@test length(ks_mesh_2D.tensor_product_masks) == 4
			@test in(((p1, p2), level_1), ks_mesh_2D.standard_cell_keys)

			# Verify standard_spectral_properties mapping
			# @test haskey(ks_mesh_2D.standard_spectral_properties, ((p1, p2), level_1))
			# @test ks_mesh_2D.standard_spectral_properties[((p1, p2), level_1)] == compute_spectral_properties(((p1, p2), level_1))
		end

		@testset "KSMesh Tests" begin
			@testset "KSMesh Error Handling" begin
				# Test 1: Empty cells (unchanged)
				@test_throws MethodError KSMesh(
					cells = [],
					global_error_estimate = 0.1,
					boundary_cells = Dict{Symbol, Vector{Int}}(),
					physical_domain = x -> true,
				)

				# Test 2: Empty physical cells (unchanged)
				ks_cell_1D = KSCell(;
					id = 1,
					p = (3,),
					level = 1,
					continuity_order = (2,),
					neighbors = Dict(:dim1_neg => -1, :dim1_pos => -1),
					node_map = Dict((1,) => 1),
					tensor_product_mask = (fill(true, 3),),
					boundary_connectivity = Dict(:dim1_neg => -1, :dim1_pos => -1),
					error_estimate = 0.1,
					legendre_decay_rate = 0.05,
				)

				@test_throws ArgumentError KSMesh(
					cells = [ks_cell_1D],
					global_error_estimate = 0.1,
					boundary_cells = Dict{Symbol, Vector{Int}}(),
					physical_domain = x -> false,
				)

				# Test 3: Duplicate cell IDs (unchanged)
				duplicate_cell = KSCell(;
					id = 1,  # Same ID as ks_cell_1D
					p = (3,),
					level = 1,
					continuity_order = (2,),
					neighbors = Dict(:dim1_neg => -1, :dim1_pos => -1),
					node_map = Dict((1,) => 2),
					tensor_product_mask = (fill(true, 3),),
					boundary_connectivity = Dict(:dim1_neg => -1, :dim1_pos => -1),
					error_estimate = 0.2,
					legendre_decay_rate = 0.04,
				)

				@test_throws ArgumentError KSMesh(
					cells = [ks_cell_1D, duplicate_cell],
					global_error_estimate = 0.2,
					boundary_cells = Dict(:dim1_neg => [1], :dim1_pos => [2]),
					physical_domain = x -> true,
				)

				# Test 4: Invalid boundary cell IDs (unchanged)
				@test_throws ArgumentError KSMesh(
					cells = [ks_cell_1D],
					global_error_estimate = 0.1,
					boundary_cells = Dict(:dim1_neg => [2]),  # Non-existent cell ID
					physical_domain = x -> true,
				)
			end

			@testset "KSMesh Functionality" begin
				# Create a more complex set of test cells
				ks_cells_2D = [
					KSCell(;
						id = i,
						p = (3, 4),
						level = 1,
						continuity_order = (2, 2),
						neighbors = Dict(
							:dim1_neg => i - 1,
							:dim1_pos => i + 1,
							:dim2_neg => -1,
							:dim2_pos => -1,
						),
						node_map = Dict((1, 1) => i),
						tensor_product_mask = (fill(true, 3), fill(true, 4)),
						boundary_connectivity = Dict(
							:dim1_neg => i - 1,
							:dim1_pos => i + 1,
							:dim2_neg => -1,
							:dim2_pos => -1,
						),
						error_estimate = 0.1,
						legendre_decay_rate = 0.05,
					) for i in 1:4
				]

				# Test KSMesh creation
				ks_mesh_2D = KSMesh(;
					cells = ks_cells_2D,
					global_error_estimate = 0.2,
					boundary_cells = Dict(:dim1_neg => [1], :dim1_pos => [4]),
					physical_domain = x -> true,
				)

				# Test basic properties
				@test length(ks_mesh_2D.cells) == 4
				@test ks_mesh_2D.global_error_estimate == 0.2
				@test ks_mesh_2D.physical_cells == [1, 2, 3, 4]
				@test ks_mesh_2D.boundary_cells == Dict(:dim1_neg => [1], :dim1_pos => [4])

				# Test tensor_product_masks
				@test length(ks_mesh_2D.tensor_product_masks) == 4
				@test all(
					mask -> mask == (fill(true, 3), fill(true, 4)),
					ks_mesh_2D.tensor_product_masks,
				)

				# Test standard_cell_keys
				@test ks_mesh_2D.standard_cell_keys == Set([((3, 4), 1)])

				# Test node_map
				@test length(ks_mesh_2D.node_map) == 4
				@test all(i -> haskey(ks_mesh_2D.node_map, (i, (1, 1))), 1:4)

				# Test get_unique_standard_cell_keys
				@test get_unique_standard_cell_keys(ks_mesh_2D) == Set([((3, 4), 1)])
			end

			# @testset "KSMesh Helper Functions" begin
			# 	# Define a more complex physical domain function
			# 	physical_domain(x) = all(0.25 .<= x .<= 0.75)

			# 	# Create test cells
			# 	ks_cells_2D = [KSCell(id = i,
			# 						  p = (3, 4),
			# 						  level = 1,
			# 						  continuity_order = (2, 2),
			# 						  neighbors = Dict(:dim1_neg => i - 1, :dim1_pos => i + 1, :dim2_neg => i - 2, :dim2_pos => i + 2),
			# 						  node_map = Dict((x, y) => i * 10 + x * 3 + y for x in 1:3, y in 1:4),
			# 						  tensor_product_mask = (fill(true, 3), fill(true, 4)),
			# 						  boundary_connectivity = Dict(:dim1_neg => i - 1, :dim1_pos => i + 1, :dim2_neg => i - 2, :dim2_pos => i + 2),
			# 						  error_estimate = 0.1,
			# 						  legendre_decay_rate = 0.05) for i in 1:16]

			# 	# Test detect_physical_cells function
			# 	@testset "detect_physical_cells" begin
			# 		physical_cells = detect_physical_cells(ks_cells_2D, physical_domain)
			# 		@test Set(physical_cells) == Set([6, 7, 10, 11])  # Cells that should be in the physical domain
			# 	end

			# 	# Test is_physical_cell function
			# 	@testset "is_physical_cell" begin
			# 		@test !is_physical_cell(ks_cells_2D[1], physical_domain)
			# 		@test is_physical_cell(ks_cells_2D[6], physical_domain)
			# 		@test is_physical_cell(ks_cells_2D[11], physical_domain)
			# 		@test !is_physical_cell(ks_cells_2D[16], physical_domain)
			# 	end
			# end
		end

		# 	@testset "Complex Domain Tests" begin
		# 		# # Define a circular domain
		# 		# function in_circle(x::Float64, y::Float64, cx::Float64, cy::Float64, r::Float64)::Bool
		# 		# 	# Calculate the distance from the point (x, y) to the circle center (cx, cy)
		# 		# 	distance_squared = (x - cx)^2 + (y - cy)^2
		# 		# 	# Check if the distance is less than or equal to the radius squared
		# 		# 	return distance_squared <= r^2
		# 		# end

		# 		# circle_center = (0.5, 0.5)
		# 		# circle_radius = 0.3

		# 		# # Create a 4x4 grid of 2D cells
		# 		# ks_cells_circle = [KSCell(id = i,
		# 		# 						  p = (p1, p2),
		# 		# 						  level = level_1,
		# 		# 						  continuity_order = (1, 1),
		# 		# 						  neighbors = Dict(:dim1_neg => i - 1, :dim1_pos => i + 1, :dim2_neg => i - 4, :dim2_pos => i + 4),
		# 		# 						  node_map = Dict((1, 1) => i),
		# 		# 						  tensor_product_mask = (fill(true, p1), fill(true, p2)),
		# 		# 						  boundary_connectivity = Dict(:dim1_neg => i - 1, :dim1_pos => i + 1, :dim2_neg => i - 4, :dim2_pos => i + 4),
		# 		# 						  error_estimate = 0.1,
		# 		# 						  legendre_decay_rate = 0.05) for i in 1:16]

		# 		# function physical_domain(node::Tuple{Float64, Float64})::Bool
		# 		# 	# Circle parameters
		# 		# 	circle_center = (0.0, 0.0)  # Example center of the circle
		# 		# 	circle_radius = 1.0         # Example radius of the circle

		# 		# 	# Extract the coordinates of the node
		# 		# 	x, y = node

		# 		# 	# Use the in_circle function to check if the node is within the circle
		# 		# 	return in_circle(x, y, circle_center[1], circle_center[2], circle_radius)
		# 		# end

		# 		# ks_mesh_circle = KSMesh(ks_cells_circle,
		# 		# 						0.2,
		# 		# 						Dict(:dim1_neg => [1, 5, 9, 13], :dim1_pos => [4, 8, 12, 16],
		# 		# 							 :dim2_neg => [1, 2, 3, 4], :dim2_pos => [13, 14, 15, 16]),
		# 		# 						physical_domain)

		# 		# @test length(ks_mesh_circle.cells) == 16
		# 		# @test ks_mesh_circle.global_error_estimate == 0.2
		# 		# @test !isempty(ks_mesh_circle.physical_cells)
		# 		# @test length(ks_mesh_circle.physical_cells) < 16  # Not all cells should be in the circle
		# 		# @test all(physical_domain(first(ks_mesh_circle.cells[i].node_map)[2]) for i in ks_mesh_circle.physical_cells)
		# 	end
	end

	@testset "KSBasisFunction Tests" begin
		# Basic case
		@testset "Basic Case" begin
			id = 1
			function_handle = x -> x^2
			contribution = 2.0
			is_removable = false

			basis_function = KSBasisFunction(
				id, function_handle, contribution, is_removable)

			@test basis_function.id == id
			@test basis_function.function_handle(2.0) == 4.0
			@test basis_function.contribution == contribution
			@test basis_function.is_removable == is_removable
		end

		# Edge case: Default values
		@testset "Default Values" begin
			id = 1
			basis_function = KSBasisFunction(id)

			@test basis_function.id == id
			@test basis_function.function_handle(2.0) == 2.0
			@test basis_function.contribution == 0
			@test basis_function.is_removable == false
		end

		# Exception handling: Invalid ID
		@testset "Invalid ID" begin
			id = -1
			@test_throws ArgumentError KSBasisFunction(id)
		end
	end

	@testset "KSTypes Problem Types Tests" begin
		@testset "KSProblem" begin
			equation = (x, t, u, ∇u, Δu) -> Δu + u
			bc = [
				KSDirichletBC(;
					boundary_value = x -> 0.0,
					boundary_region = x -> true,
					coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
				),
			]
			domain = ((0.0, 1.0),)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))
			T3 = Float64  # Ensure T3 is defined

			# Test basic construction
			problem = KSProblem(;
				equation = equation,
				boundary_conditions = bc,
				domain = domain,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1
			@test problem.continuity_order == 2
			@test problem.initial_conditions === nothing
			@test problem.tspan === nothing

			# Test with all optional parameters
			initial_conditions = [0.0]
			tspan = (0.0, 1.0)
			problem_full = KSProblem(;
				equation = equation,
				boundary_conditions = bc,
				domain = domain,
				coordinate_system = coord_sys,
				initial_conditions = initial_conditions,
				tspan = tspan,
				continuity_order = 3,
				num_vars = 2,
			)
			@test problem_full.num_vars == 2
			@test problem_full.continuity_order == 3
			@test problem_full.initial_conditions == initial_conditions
			@test problem_full.tspan == tspan

			# Test with function initial conditions
			init_func = x -> sin(π * x[1])
			problem_func_init = KSProblem(;
				equation = equation,
				boundary_conditions = bc,
				domain = domain,
				coordinate_system = coord_sys,
				initial_conditions = init_func,
			)
			@test isa(problem_func_init.initial_conditions, Function)

			# Test error handling
			@test_throws TypeError KSProblem(
				equation = equation,
				boundary_conditions = "not a BC",
				domain = domain,
				coordinate_system = coord_sys,
			)  # invalid boundary condition
			@test_throws ArgumentError KSProblem(
				equation = equation,
				boundary_conditions = bc,
				domain = domain,
				coordinate_system = coord_sys,
				continuity_order = -1,
			)  # negative continuity order
			@test_throws ArgumentError KSProblem(
				equation = equation,
				boundary_conditions = bc,
				domain = domain,
				coordinate_system = coord_sys,
				num_vars = 0,
			)  # invalid num_vars

			# Test with mixed number types
			domain_mixed = ((0.0, π),)
			tspan_mixed = (0.0, π)
			problem_mixed = KSProblem(;
				equation = equation,
				boundary_conditions = bc,
				domain = domain_mixed,
				coordinate_system = coord_sys,
				tspan = tspan_mixed,
			)
			@test eltype(problem_mixed.domain[1]) == Float64
			@test eltype(problem_mixed.tspan) == Float64

			domain_2d = ((0.0, 1.0), (0.0, 1.0))
			coord_sys_2d = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0)))
			bc_2d = KSDirichletBC(;
				boundary_value = x -> 0.0,
				boundary_region = x -> true,
				coordinate_system = coord_sys_2d,
			)
			problem_2d = KSProblem(;
				equation = equation,
				boundary_conditions = [bc_2d],
				domain = domain_2d,
				coordinate_system = coord_sys_2d,
			)
			@test length(problem_2d.domain) == 2
		end

		@testset "KSPDEProblem" begin
			pde = (x, t, u, ∇u, Δu) -> Δu + u
			init_func = x -> sin(π * x[1])
			bc = KSDirichletBC(;
				boundary_value = x -> 0.0,
				boundary_region = x -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			tspan = (0.0, 1.0)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSPDEProblem(;
				pde = pde,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = init_func,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1

			problem_multi = KSPDEProblem(;
				pde = pde,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = init_func,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 3,
			)
			@test problem_multi.num_vars == 3

			@test_throws ArgumentError KSPDEProblem(
				pde = pde,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = init_func,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
		end

		@testset "KSODEProblem" begin
			ode = (t, y) -> -y
			bc = KSDirichletBC(;
				boundary_value = t -> 0.0,
				boundary_region = t -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			initial_conditions = [1.0]
			tspan = (0.0, 1.0)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSODEProblem(;
				ode = ode,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1

			problem_multi = KSODEProblem(;
				ode = ode,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = [1.0, 2.0],
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem_multi.num_vars == 2

			@test_throws ArgumentError KSODEProblem(
				ode = ode,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
		end

		@testset "KSDAEProblem" begin
			dae = (t, y, dy) -> [dy[1] - y[2], y[1]^2 + y[2]^2 - 1]
			bc = KSDirichletBC(;
				boundary_value = t -> [0.0, 0.0],
				boundary_region = t -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			initial_conditions = [1.0, 0.0]
			tspan = (0.0, 1.0)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSDAEProblem(;
				dae = dae,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 2
			@test problem.num_algebraic_vars == 0

			problem_alg = KSDAEProblem(;
				dae = dae,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_algebraic_vars = 1,
			)
			@test problem_alg.num_vars == 2
			@test problem_alg.num_algebraic_vars == 1

			@test_throws ArgumentError KSDAEProblem(
				dae = dae,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_algebraic_vars = 2,
			)
			@test_throws MethodError KSDAEProblem(
				1,
				[bc],
				domain,
				initial_conditions,
				tspan,
				coord_sys,
			)  # dae not a function
		end

		@testset "KSBVDAEProblem" begin
			f = (t, y, dy) -> [dy[1] - y[2], y[1]]
			g = (t, y) -> [y[1]^2 + y[2]^2 - 1]
			bc = KSDirichletBC(;
				boundary_value = t -> [0.0, 0.0],
				boundary_region = t -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, π),)
			initial_conditions = [1.0, 0.0]
			algebraic_vars = [false, true]
			tspan = (0.0, π)
			coord_sys = KSCartesianCoordinates(((0.0, π),))

			problem = KSBVDAEProblem(;
				f = f,
				g = g,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				algebraic_vars = algebraic_vars,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 2

			@test_throws ArgumentError KSBVDAEProblem(
				f = f,
				g = g,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				algebraic_vars = algebraic_vars,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
			@test_throws TypeError KSBVDAEProblem(
				f = 1,
				g = g,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				algebraic_vars = algebraic_vars,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
		end

		@testset "KSIDEProblem" begin
			f = (t, y) -> cos(t) + 0.5 * y
			K = (t, s) -> exp(-(t - s)^2)
			bc = KSDirichletBC(;
				boundary_value = t -> 0.0,
				boundary_region = t -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			initial_conditions = [1.0]
			tspan = (0.0, 1.0)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSIDEProblem(;
				ide = f,
				kernel = K,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1

			problem_multi = KSIDEProblem(;
				ide = f,
				kernel = K,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = [1.0, 2.0],
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem_multi.num_vars == 2

			@test_throws ArgumentError KSIDEProblem(
				ide = f,
				kernel = K,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
		end

		@testset "KSPIDEProblem" begin
			pide = (t, u, du) -> du - u + 0.5 * (t^2 - u^2)
			kernel = (t, s) -> exp(-(t - s))
			bc = KSDirichletBC(;
				boundary_value = t -> sin(t),
				boundary_region = t -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			initial_conditions = x -> sin(x[1])
			tspan = (0.0, 1.0)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSPIDEProblem(;
				pide = pide,
				kernel = kernel,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1

			problem_multi = KSPIDEProblem(;
				pide = pide,
				kernel = kernel,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = [1.0, 2.0],
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 2,
			)
			@test problem_multi.num_vars == 2

			@test_throws ArgumentError KSPIDEProblem(
				pide = pide,
				kernel = kernel,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
		end

		@testset "KSMovingBoundaryPDEProblem" begin
			pde = (x, t, u, ux, ut) -> ut - 0.1 * ux^2
			bc = KSDirichletBC(;
				boundary_value = (x, t) -> x[1] == 0.0 ? 1.0 : 0.0,
				boundary_region = x -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			initial_conditions = [0.0]
			tspan = (0.0, 1.0)
			boundary_motion = (t, s, u, ux) -> -ux[1]
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSMovingBoundaryPDEProblem(;
				pde = pde,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				boundary_motion = boundary_motion,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1

			problem_multi = KSMovingBoundaryPDEProblem(;
				pde = pde,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = [0.0, 0.0],
				tspan = tspan,
				boundary_motion = boundary_motion,
				coordinate_system = coord_sys,
				num_vars = 2,
			)
			@test problem_multi.num_vars == 2

			@test_throws ArgumentError KSMovingBoundaryPDEProblem(
				pde = pde,
				boundary_conditions = [bc],
				domain = domain,
				initial_conditions = initial_conditions,
				tspan = tspan,
				boundary_motion = boundary_motion,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
		end

		@testset "KSBVPProblem" begin
			bvp = (du, u, p, t) -> (du - u^2 + 1)
			bc = KSDirichletBC(;
				boundary_value = (u0, u1, p) -> [u0 - 0, u1 - 1],
				boundary_region = x -> true,
				coordinate_system = KSCartesianCoordinates(((0.0, 1.0),)),
			)
			domain = ((0.0, 1.0),)
			initial_guess = x -> x[1]
			tspan = (0.0, 1.0)
			coord_sys = KSCartesianCoordinates(((0.0, 1.0),))

			problem = KSBVPProblem(;
				bvp = bvp,
				boundary_conditions = [bc],
				domain = domain,
				initial_guess = initial_guess,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			@test problem.num_vars == 1

			problem_multi = KSBVPProblem(;
				bvp = bvp,
				boundary_conditions = [bc],
				domain = domain,
				initial_guess = [0.0, 1.0],
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 2,
			)
			@test problem_multi.num_vars == 2

			@test_throws ArgumentError KSBVPProblem(
				bvp = bvp,
				boundary_conditions = [bc],
				domain = domain,
				initial_guess = initial_guess,
				tspan = tspan,
				coordinate_system = coord_sys,
				num_vars = 0,
			)
		end

		@testset "KSCoupledProblem" begin
			ode1 = (t, y) -> [y[2], -y[1]]
			ode2 = (t, y) -> -y
			domain = ((0.0, 2π),)
			initial_condition1 = [1.0, 0.0]
			initial_condition2 = [1.0]
			tspan = (0.0, 2π)
			coord_sys = KSCartesianCoordinates(((0.0, 2π),))

			problem1 = KSODEProblem(;
				ode = ode1,
				boundary_conditions = [
					KSDirichletBC(;
						boundary_value = t -> [0.0, 0.0],
						boundary_region = t -> true,
						coordinate_system = coord_sys,
					),
				],
				domain = domain,
				initial_conditions = initial_condition1,
				tspan = tspan,
				coordinate_system = coord_sys,
			)
			problem2 = KSODEProblem(;
				ode = ode2,
				boundary_conditions = [
					KSDirichletBC(;
						boundary_value = t -> 0.0,
						boundary_region = t -> true,
						coordinate_system = coord_sys,
					),
				],
				domain = domain,
				initial_conditions = initial_condition2,
				tspan = tspan,
				coordinate_system = coord_sys,
			)

			coupling_terms = Matrix{Union{Nothing, Function}}(nothing, 2, 2)
			coupling_terms[1, 2] =
				coupling_terms[2, 1] =
					(mesh) -> spdiagm(0 => 0.1 *
										   ones(length(mesh.node_map)))

			coupled_problem = KSCoupledProblem(;
				problems = [problem1, problem2],
				coupling_terms = coupling_terms,
			)
			@test coupled_problem.num_vars == 3

			@test_throws ArgumentError KSCoupledProblem(
				problems = [problem1],
				coupling_terms = coupling_terms,
			)  # Inconsistent dimensions
			@test_throws TypeError KSCoupledProblem(
				problems = [problem1, problem2],
				coupling_terms = ones(2, 2),
			)  # Invalid coupling_terms type
		end

		@testset "KSDiscretizedProblem" begin
			time_nodes = [0.0, 0.5, 1.0]
			spatial_nodes = ([0.0, 0.5, 1.0],)
			system_matrix = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
			initial_conditions = [0.0, 0.0, 0.0]
			problem_functions = (x -> x^2,)

			problem = KSDiscretizedProblem(;
				time_nodes = time_nodes,
				spatial_nodes = spatial_nodes,
				system_matrix = system_matrix,
				initial_conditions = initial_conditions,
				problem_functions = problem_functions,
				num_vars = 1,
			)
			@test problem.num_vars == 1

			problem_multi = KSDiscretizedProblem(;
				time_nodes = time_nodes,
				spatial_nodes = spatial_nodes,
				system_matrix = system_matrix,
				initial_conditions = initial_conditions,
				problem_functions = problem_functions,
				num_vars = 3,
			)
			@test problem_multi.num_vars == 3

			@test_throws ArgumentError KSDiscretizedProblem(
				time_nodes = time_nodes,
				spatial_nodes = spatial_nodes,
				system_matrix = system_matrix,
				initial_conditions = initial_conditions,
				problem_functions = problem_functions,
				num_vars = 0,
			)
			@test_throws ArgumentError KSDiscretizedProblem(
				time_nodes = Float64[],
				spatial_nodes = spatial_nodes,
				system_matrix = system_matrix,
				initial_conditions = initial_conditions,
				problem_functions = problem_functions,
			)  # Empty time_nodes
		end

		@testset "KSOptimalControlProblem" begin
			state_equations = [x -> [0.0, 1.0], x -> [-x[2], x[1]]]
			cost_functions = [(x, u) -> 0.5 * (x[1]^2 + x[2]^2 + u[1]^2)]
			terminal_cost = x -> 0.0
			initial_state = [1.0, 0.0]
			t_span = (0.0, 1.0)
			control_bounds = [(-1.0, 1.0)]
			dt = 0.01

			problem = KSOptimalControlProblem(;
				state_equations = state_equations,
				cost_functions = cost_functions,
				terminal_cost = terminal_cost,
				initial_state = initial_state,
				t_span = t_span,
				control_bounds = control_bounds,
				dt = dt,
			)

			@test problem.num_vars == 2
			@test problem.num_controls == 1
			@test problem.num_time_steps == 101
		end
	end
	@testset "Coordinate Systems" begin
		@testset "KSCartesianCoordinates" begin
			# 1D case
			coords_1d = KSCartesianCoordinates(((0.0, 1.0),))
			@test coords_1d.ranges == ((0.0, 1.0),)
			@test coords_1d.active == (true,)
			@test coords_1d.domains == ((0.0, 1.0),)

			# 2D case
			coords_2d = KSCartesianCoordinates(((0.0, 1.0), (-1.0, 1.0)))
			@test coords_2d.ranges == ((0.0, 1.0), (-1.0, 1.0))
			@test coords_2d.active == (true, true)
			@test coords_2d.domains == ((0.0, 1.0), (-1.0, 1.0))

			# 3D case
			coords_3d = KSCartesianCoordinates(((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)))
			@test coords_3d.ranges == ((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
			@test coords_3d.active == (true, true, true)
			@test coords_3d.domains == ((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0))

			# Invalid range cases
			@test_throws ArgumentError KSCartesianCoordinates(((1.0, 0.0),))  # Invalid range
		end

		@testset "KSPolarCoordinates" begin
			coords_full = KSPolarCoordinates((0.0, 2.0), (0.0, 2π))
			@test coords_full.r == (0.0, 2.0)
			@test isapprox(coords_full.theta[1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.theta[2], 2π, atol = 1e-10)
			@test coords_full.active == (true, true)
			@test coords_full.domains[1] == (0.0, 2.0)
			@test isapprox(coords_full.domains[2][1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.domains[2][2], 2π, atol = 1e-10)

			@testset "Inactive domains" begin
				coords_partial = KSPolarCoordinates((0.0, 2.0), nothing)
				@test coords_partial.r == (0.0, 2.0)
				@test coords_partial.theta === nothing
				@test coords_partial.active == (true, false)
				@test coords_partial.domains == ((0.0, 2.0), nothing)
			end
			@test_throws ArgumentError KSPolarCoordinates((-1.0, 1.0), (0.0, 2π))  # Invalid r range
			@test_throws ArgumentError KSPolarCoordinates((0.0, 1.0), (0.0, 3π))  # Invalid theta range
		end

		@testset "KSCylindricalCoordinates" begin
			coords_full = KSCylindricalCoordinates((0.0, 2.0), (0.0, 2π), (-1.0, 1.0))
			@test coords_full.r == (0.0, 2.0)
			@test isapprox(coords_full.theta[1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.theta[2], 2π, atol = 1e-10)
			@test coords_full.z == (-1.0, 1.0)
			@test coords_full.active == (true, true, true)
			@test coords_full.domains[1] == (0.0, 2.0)
			@test isapprox(coords_full.domains[2][1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.domains[2][2], 2π, atol = 1e-10)
			@test coords_full.domains[3] == (-1.0, 1.0)

			@testset "Inactive domains" begin
				# Test with inactive theta
				coords_no_theta = KSCylindricalCoordinates((0.0, 2.0), nothing, (-1.0, 1.0))
				@test coords_no_theta.r == (0.0, 2.0)
				@test coords_no_theta.theta === nothing
				@test coords_no_theta.z == (-1.0, 1.0)
				@test coords_no_theta.active == (true, false, true)
				@test coords_no_theta.domains == ((0.0, 2.0), nothing, (-1.0, 1.0))

				coords_no_z = KSCylindricalCoordinates((0.0, 2.0), (0.0, 2π), nothing)
				@test coords_no_z.r == (0.0, 2.0)
				@test isapprox(coords_no_z.theta[1], 0.0, atol = 1e-10)
				@test isapprox(coords_no_z.theta[2], 2π, atol = 1e-10)
				@test coords_no_z.z === nothing
				@test coords_no_z.active == (true, true, false)
				@test coords_no_z.domains[1] == (0.0, 2.0)
				@test isapprox(coords_no_z.domains[2][1], 0.0, atol = 1e-10)
				@test isapprox(coords_no_z.domains[2][2], 2π, atol = 1e-10)
				@test coords_no_z.domains[3] === nothing
			end
			@testset "Error handling" begin
				@test_throws ArgumentError KSCylindricalCoordinates(
					(-1.0, 1.0),
					(0.0, 2π),
					(-1.0, 1.0),
				)  # Invalid r range
				@test_throws ArgumentError KSCylindricalCoordinates(
					(0.0, 1.0),
					(0.0, 3π),
					(-1.0, 1.0),
				)  # Invalid theta range
			end
		end
		@testset "KSSphericalCoordinates" begin
			coords_full = KSSphericalCoordinates((0.0, 2.0), (0.0, π), (0.0, 2π))
			@test coords_full.r == (0.0, 2.0)
			@test isapprox(coords_full.theta[1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.theta[2], π, atol = 1e-10)
			@test isapprox(coords_full.phi[1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.phi[2], 2π, atol = 1e-10)
			@test coords_full.active == (true, true, true)
			@test coords_full.domains[1] == (0.0, 2.0)
			@test isapprox(coords_full.domains[2][1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.domains[2][2], π, atol = 1e-10)
			@test isapprox(coords_full.domains[3][1], 0.0, atol = 1e-10)
			@test isapprox(coords_full.domains[3][2], 2π, atol = 1e-10)

			@testset "Inactive domain" begin
				# Test with inactive theta
				coords_no_theta = KSSphericalCoordinates((0.0, 2.0), nothing, (0.0, 2π))
				@test coords_no_theta.r == (0.0, 2.0)
				@test coords_no_theta.theta === nothing
				@test isapprox(coords_no_theta.phi[1], 0.0, atol = 1e-10)
				@test isapprox(coords_no_theta.phi[2], 2π, atol = 1e-10)
				@test coords_no_theta.active == (true, false, true)
				@test coords_no_theta.domains == ((0.0, 2.0), nothing, coords_no_theta.phi)

				coords_no_phi = KSSphericalCoordinates((0.0, 2.0), (0.0, π), nothing)
				@test coords_no_phi.r == (0.0, 2.0)
				@test isapprox(coords_no_phi.theta[1], 0.0, atol = 1e-10)
				@test isapprox(coords_no_phi.theta[2], π, atol = 1e-10)
				@test coords_no_phi.phi === nothing
				@test coords_no_phi.active == (true, true, false)
				@test coords_no_phi.domains == ((0.0, 2.0), coords_no_phi.theta, nothing)
			end
			@testset "Error handling" begin
				@test_throws ArgumentError KSSphericalCoordinates(
					(-1.0, 1.0),
					(0.0, π),
					(0.0, 2π),
				)  # Invalid r range
				@test_throws ArgumentError KSSphericalCoordinates(
					(0.0, 1.0),
					(0.0, 2π),
					(0.0, 2π),
				)  # Invalid theta range
				@test_throws ArgumentError KSSphericalCoordinates(
					(0.0, 1.0),
					(0.0, π),
					(0.0, 3π),
				)  # Invalid phi range
			end
		end
	end
	@testset "Boundary Conditions with Coordinate Systems" begin
		for system_type in [:Cartesian, :Polar, :Cylindrical, :Spherical]
			@testset "$system_type Coordinate System" begin

				# Test Dirichlet Boundary Condition
				@testset "KSDirichletBC" begin
					boundary_value = (x, t) -> sin(π * x)
					boundary_region = x -> x == 0 || x == 1
					coord_system = create_mock_coordinate_system(system_type)

					bc = KSDirichletBC(;
						boundary_value = boundary_value,
						boundary_region = boundary_region,
						coordinate_system = coord_system,
					)
					@test bc.boundary_value == boundary_value
					@test bc.boundary_region == boundary_region
					@test bc.coordinate_system == coord_system

					# Test with AbstractArray as input
					boundary_value_array = [1.0, 2.0, 3.0]
					bc_array = KSDirichletBC(;
						boundary_value = boundary_value_array,
						coordinate_system = coord_system,
					)
					@test bc_array.boundary_value == boundary_value_array
				end

				# Test Dirichlet Boundary Condition with inactive domains
				@testset "KSDirichletBC with inactive domains" begin
					boundary_value = (x, t) -> sin(π * x)
					coord_system_inactive = create_mock_coordinate_system(
						system_type; inactive = true)

					bc_inactive = KSDirichletBC(;
						boundary_value = boundary_value,
						coordinate_system = coord_system_inactive,
					)
					@test bc_inactive.boundary_value == boundary_value
					@test bc_inactive.coordinate_system == coord_system_inactive

					# Test with AbstractArray as input
					boundary_value_array = [1.0, 2.0, 3.0]
					bc_array_inactive = KSDirichletBC(;
						boundary_value = boundary_value_array,
						coordinate_system = coord_system_inactive,
					)
					@test bc_array_inactive.boundary_value == boundary_value_array
				end

				# Test Neumann Boundary Condition
				@testset "KSNeumannBC" begin
					flux_value = (x, t) -> cos(π * x)
					boundary_region = x -> x == 0 || x == 1
					coord_system = create_mock_coordinate_system(system_type)

					bc = KSNeumannBC(;
						flux_value = flux_value,
						boundary_region = boundary_region,
						coordinate_system = coord_system,
					)
					@test bc.flux_value == flux_value
					@test bc.boundary_region == boundary_region
					@test bc.coordinate_system == coord_system

					# Test with AbstractArray as input
					flux_value_array = [0.5, 1.0, 1.5]
					bc_array = KSNeumannBC(;
						flux_value = flux_value_array,
						coordinate_system = coord_system,
					)
					@test bc_array.flux_value == flux_value_array
				end

				# Test Neumann Boundary Condition with inactive domains
				@testset "KSNeumannBC with inactive domains" begin
					flux_value = (x, t) -> cos(π * x)
					coord_system_inactive = create_mock_coordinate_system(
						system_type; inactive = true)

					bc_inactive = KSNeumannBC(;
						flux_value = flux_value,
						coordinate_system = coord_system_inactive,
					)
					@test bc_inactive.flux_value == flux_value
					@test bc_inactive.coordinate_system == coord_system_inactive

					# Test with AbstractArray as input
					flux_value_array = [0.5, 1.0, 1.5]
					bc_array_inactive = KSNeumannBC(;
						flux_value = flux_value_array,
						coordinate_system = coord_system_inactive,
					)
					@test bc_array_inactive.flux_value == flux_value_array
				end

				# Test Robin Boundary Condition
				@testset "KSRobinBC" begin
					neumann_coefficient = (x, t) -> 1.0
					dirichlet_coefficient = (x, t) -> 2.0
					boundary_value = (x, t) -> sin(π * x)
					boundary_region = x -> x == 0 || x == 1
					coord_system = create_mock_coordinate_system(system_type)

					bc = KSRobinBC(;
						neumann_coefficient = neumann_coefficient,
						dirichlet_coefficient = dirichlet_coefficient,
						boundary_value = boundary_value,
						boundary_region = boundary_region,
						coordinate_system = coord_system,
					)
					@test bc.neumann_coefficient == neumann_coefficient
					@test bc.dirichlet_coefficient == dirichlet_coefficient
					@test bc.boundary_value == boundary_value
					@test bc.boundary_region == boundary_region
					@test bc.coordinate_system == coord_system

					# Test with AbstractArray as input
					neumann_coefficient_array = [1.0, 2.0]
					dirichlet_coefficient_array = [3.0, 4.0]
					boundary_value_array = [0.5, 1.0]
					bc_array = KSRobinBC(;
						neumann_coefficient = neumann_coefficient_array,
						dirichlet_coefficient = dirichlet_coefficient_array,
						boundary_value = boundary_value_array,
						coordinate_system = coord_system,
					)
					@test bc_array.neumann_coefficient == neumann_coefficient_array
					@test bc_array.dirichlet_coefficient == dirichlet_coefficient_array
					@test bc_array.boundary_value == boundary_value_array
				end

				# Test Robin Boundary Condition with inactive domains
				@testset "KSRobinBC with inactive domains" begin
					neumann_coefficient = (x, t) -> 1.0
					dirichlet_coefficient = (x, t) -> 2.0
					boundary_value = (x, t) -> sin(π * x)
					coord_system_inactive = create_mock_coordinate_system(
						system_type; inactive = true)

					bc_inactive = KSRobinBC(;
						neumann_coefficient = neumann_coefficient,
						dirichlet_coefficient = dirichlet_coefficient,
						boundary_value = boundary_value,
						coordinate_system = coord_system_inactive,
					)
					@test bc_inactive.neumann_coefficient == neumann_coefficient
					@test bc_inactive.dirichlet_coefficient == dirichlet_coefficient
					@test bc_inactive.boundary_value == boundary_value
					@test bc_inactive.coordinate_system == coord_system_inactive

					# Test with AbstractArray as input
					neumann_coefficient_array = [1.0, 2.0]
					dirichlet_coefficient_array = [3.0, 4.0]
					boundary_value_array = [0.5, 1.0]
					bc_array_inactive = KSRobinBC(;
						neumann_coefficient = neumann_coefficient_array,
						dirichlet_coefficient = dirichlet_coefficient_array,
						boundary_value = boundary_value_array,
						coordinate_system = coord_system_inactive,
					)
					@test bc_array_inactive.neumann_coefficient == neumann_coefficient_array
					@test bc_array_inactive.dirichlet_coefficient ==
						dirichlet_coefficient_array
					@test bc_array_inactive.boundary_value == boundary_value_array
				end
			end
		end
	end

	@testset "KSSolverOptions" begin
		@testset "Valid case" begin
			options = KSSolverOptions(
				100,
				1e-6,
				true,
				5,
				0.1,
				10,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test options.max_iterations == 100
			@test options.tolerance == 1e-6
			@test options.adaptive == true
			@test options.max_levels == 5
			@test options.smoothness_threshold == 0.1
			@test options.initial_elements == 10
			@test options.initial_degree == 3
			@test options.use_domain_decomposition == false
			@test options.num_subdomains == 1
			@test options.use_strang_splitting == false
			@test options.dt == 0.01
			@test options.num_steps == 100
			@test options.legendre_threshold == 0.1
		end

		@testset "Invalid cases" begin
			@test_throws ArgumentError KSSolverOptions(
				0,
				1e-6,
				true,
				5,
				0.1,
				10,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test_throws ArgumentError KSSolverOptions(
				100,
				0.0,
				true,
				5,
				0.1,
				10,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test_throws ArgumentError KSSolverOptions(
				100,
				1e-6,
				true,
				0,
				0.1,
				10,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test_throws ArgumentError KSSolverOptions(
				100,
				1e-6,
				true,
				5,
				0.0,
				10,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test_throws ArgumentError KSSolverOptions(
				100,
				1e-6,
				true,
				5,
				0.1,
				0,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test_throws ArgumentError KSSolverOptions(
				100,
				1e-6,
				true,
				5,
				0.1,
				10,
				0,
				false,
				1,
				false,
				0.01,
				100,
				0.1,
			)
			@test_throws ArgumentError KSSolverOptions(
				100,
				1e-6,
				true,
				5,
				0.1,
				10,
				3,
				false,
				1,
				false,
				0.01,
				100,
				0.0,
			)
		end
	end

	@testset "Linear Solvers" begin
		@testset "KSDirectSolver" begin
			solver = KSDirectSolver(:lu)
			@test solver.method == :lu
		end

		@testset "KSIterativeSolver" begin
			solver = KSIterativeSolver(:cg, 1000, 1e-6)
			@test solver.method == :cg
			@test solver.maxiter == 1000
			@test solver.tolerance == 1e-6
			@test solver.preconditioner === nothing

			precond = x -> x
			solver_with_precond = KSIterativeSolver(:gmres, 500, 1e-8, precond)
			@test solver_with_precond.method == :gmres
			@test solver_with_precond.maxiter == 500
			@test solver_with_precond.tolerance == 1e-8
			@test solver_with_precond.preconditioner == precond

			@test_throws ArgumentError KSIterativeSolver(:cg, 0, 1e-6)
			@test_throws ArgumentError KSIterativeSolver(:cg, 1000, 0.0)
		end

		@testset "KSAMGSolver" begin
			solver = KSAMGSolver(1000, 1e-6, :jacobi)
			@test solver.maxiter == 1000
			@test solver.tolerance == 1e-6
			@test solver.preconditioner == :jacobi

			@test_throws ArgumentError KSAMGSolver(0, 1e-6, :jacobi)
			@test_throws ArgumentError KSAMGSolver(1000, 0.0, :jacobi)
		end
	end
	@testset "Helper Functions" begin
		@testset "tuple_if_active" begin
			@test tuple_if_active((1, 2), nothing, (3, 4)) == ((1, 2), (3, 4))
			@test tuple_if_active(nothing, (5, 6), nothing) == ((5, 6),)
			@test tuple_if_active(nothing, nothing, nothing) == ()
			@test tuple_if_active((1, 2), (3, 4), (5, 6)) == ((1, 2), (3, 4), (5, 6))
		end
	end
	@testset "Cache System" begin
		@testset "Cache Stats" begin
			stats = CacheStats()
			@test stats.hits == 0
			@test stats.misses == 0
			@test stats.evictions == 0
			@test stats.total_time_saved == 0.0
			@test stats.custom isa Dict
		end

		@testset "Cache Metadata" begin
			metadata = CacheMetadata(1, time(), time(), 100)
			@test metadata.access_count == 1
			@test metadata.size == 100
			@test metadata.custom isa Dict{Symbol, Any}
		end

		@testset "TransformCache" begin
			cache = TransformCache{Float64}(; max_size = 100)
			@test cache.max_size == 100
			@test cache.cleanup_threshold ≈ 0.8
			@test isempty(cache.cells)
			@test isempty(cache.operator_cache)
			@test isempty(cache.matrix_cache)
		end

		@testset "Cache Eviction Strategies" begin
			@testset "LRU Eviction" begin
				lru = LRUEviction()
				@test isa(lru, CacheEvictionStrategy)
			end

			@testset "FIFO Eviction" begin
				fifo = FIFOEviction()
				@test isempty(fifo.insertion_order)
			end

			@testset "Custom Eviction" begin
				custom_func = x -> true
				custom = CustomEviction(custom_func)
				@test custom.evict == custom_func
			end
		end

		@testset "Cache Manager" begin
			manager = CacheManager{Int}(100; strategy = :lru)
			@test manager.capacity == 100
			@test isempty(manager.items)
			@test isempty(manager.metadata)
			@test isa(manager.strategy, LRUEviction)
		end
	end
end

# end
