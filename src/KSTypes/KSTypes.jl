module KSTypes
using StaticArrays, SparseArrays, LinearAlgebra

# 1. Exports
export AbstractKSCell, AbstractStandardKSCell, AbstractKSMesh, AbstractKSProblem
export AbstractKSCoordinateSystem, AbstractKSBoundaryCondition, AbstractKSBasisFunction
export AbstractKSSolver, AbstractKSOptimizationProblem, AbstractKSLinearSolver
export AbstractKSTransform, AbstractDomainMapping, AbstractLevelSetFunction, LRUEviction
export StandardSpectralProperties, AffineTransform, NonlinearTransform, CompositeTransform
export MultiLevelTransform, PrimitiveLevelSet, CompositeLevelSet, DomainMapping, KSCell
export StandardKSCell, TransformCache, KSMesh, KSBasisFunction, KSProblem, KSODEProblem
export KSBVDAEProblem, KSBVPProblem, KSCoupledProblem, KSIDEProblem, KSPDEProblem
export KSDAEProblem, KSMovingBoundaryPDEProblem, KSPIDEProblem, KSDiscretizedProblem
export KSOptimalControlProblem, validate_range, KSCartesianCoordinates, KSPolarCoordinates
export KSCylindricalCoordinates, KSSphericalCoordinates, KSDirichletBC, KSNeumannBC
export KSRobinBC, KSSolverOptions, KSDirectSolver, KSIterativeSolver, KSAMGSolver
export tuple_if_active, detect_physical_cells, is_physical_cell, get_cell_corners
export get_unique_standard_cell_keys, CompositeLevelSet, CoordinateRanges, FIFOEviction
export CacheEvictionStrategy, CacheManager, CacheMetadata, CacheStats, CustomEviction

# 2. Abstract types
abstract type AbstractKSCell{T, N} end
abstract type AbstractStandardKSCell{T, N} end
abstract type AbstractKSMesh{T, N} end
abstract type AbstractKSProblem end
abstract type AbstractKSCoordinateSystem end
abstract type AbstractKSBoundaryCondition end
abstract type AbstractKSBasisFunction end
abstract type AbstractKSSolver end
abstract type AbstractKSOptimizationProblem end
abstract type AbstractKSLinearSolver end
abstract type AbstractKSTransform end
abstract type AbstractDomainMapping end
abstract type AbstractLevelSetFunction end
abstract type CacheEvictionStrategy end

function validate_range(
	range::Tuple{T, T}, name::String, min_val::T = -Inf, max_val::T = Inf
) where {T <: Number}
	lower, upper = range
	if lower >= upper
		throw(
			ArgumentError(
				"Invalid $name range: lower bound ($lower) must be less than upper bound ($upper)"
			),
		)
	end
	if lower < min_val
		throw(
			ArgumentError("Invalid $name range: lower bound ($lower) must be >= $min_val")
		)
	end
	if upper > max_val
		throw(
			ArgumentError("Invalid $name range: upper bound ($upper) must be <= $max_val")
		)
	end
	return (lower, upper)
end

# 4. Cache-related types (since other types depend on these)
mutable struct CacheStats
	hits::Int
	misses::Int
	evictions::Int
	total_time_saved::Float64
	last_cleanup::Float64
	custom::Dict{Any, Any} # Add field for custom stats

	CacheStats() = new(0, 0, 0, 0.0, time(), Dict())
end

mutable struct CacheMetadata
	access_count::Int
	last_accessed::Float64
	created::Float64
	size::Int
	custom::Dict{Symbol, Any}

	function CacheMetadata(
		access_count::Int, last_accessed::Float64, created::Float64, size::Int
	)
		return new(access_count, last_accessed, created, size, Dict{Symbol, Any}())
	end
end

struct LRUEviction <: CacheEvictionStrategy end
struct FIFOEviction <: CacheEvictionStrategy
	insertion_order::Vector{Any}

	FIFOEviction() = new(Vector{Any}())
end
struct CustomEviction <: CacheEvictionStrategy
	evict::Function
end

# 5. Core geometric/mathematical types
struct StandardSpectralProperties{T <: Number}
	p::Int
	continuity_order::Int
	nodes_with_boundary::Vector{T}
	nodes_interior::Vector{T}
	weights_with_boundary::Vector{T}
	weights_interior::Vector{T}
	differentiation_matrix_with_boundary::Matrix{T}
	differentiation_matrix_interior::Matrix{T}
	quadrature_matrix::Matrix{T}
	higher_order_diff_matrices_with_boundary::Vector{Matrix{T}}
	higher_order_diff_matrices_interior::Vector{Matrix{T}}

	# Inner constructor
	function StandardSpectralProperties{T}(
		p::Int,
		continuity_order::Int,
		nodes_with_boundary::Vector{T},
		nodes_interior::Vector{T},
		weights_with_boundary::Vector{T},
		weights_interior::Vector{T},
		differentiation_matrix_with_boundary::Matrix{T},
		differentiation_matrix_interior::Matrix{T},
		quadrature_matrix::Matrix{T},
		higher_order_diff_matrices_with_boundary::Vector{Matrix{T}} = Matrix{T}[],
		higher_order_diff_matrices_interior::Vector{Matrix{T}} = Matrix{T}[],
	) where {T <: Number}
		# Validation
		p >= 1 || throw(ArgumentError("The polynomial degree `p` must be at least 1"))
		continuity_order >= 0 ||
			throw(ArgumentError("Continuity order must be non-negative"))
		length(nodes_with_boundary) == p + 2 || throw(
			DimensionMismatch(
				"Length of nodes_with_boundary ($(length(nodes_with_boundary))) must be p + 2 ($(p + 2))"
			),
		)
		length(nodes_interior) == p || throw(
			DimensionMismatch(
				"Length of nodes_interior ($(length(nodes_interior))) must be p ($p)"
			),
		)
		length(weights_with_boundary) == p + 2 || throw(
			DimensionMismatch(
				"Length of weights_with_boundary ($(length(weights_with_boundary))) must be p + 2 ($(p + 2))"
			),
		)
		length(weights_interior) == p || throw(
			DimensionMismatch(
				"Length of weights_interior ($(length(weights_interior))) must be p ($p)"
			),
		)
		size(differentiation_matrix_with_boundary) == (p + 2, p + 2) || throw(
			DimensionMismatch(
				"Size of differentiation_matrix_with_boundary ($(size(differentiation_matrix_with_boundary))) must be (p + 2, p + 2)"
			),
		)
		size(differentiation_matrix_interior) == (p, p) || throw(
			DimensionMismatch(
				"Size of differentiation_matrix_interior ($(size(differentiation_matrix_interior))) must be (p, p)"
			),
		)
		size(quadrature_matrix) == (p, p) || throw(
			DimensionMismatch(
				"Size of quadrature_matrix ($(size(quadrature_matrix))) must be (p, p)"
			),
		)

		return new{T}(
			p,
			continuity_order,
			nodes_with_boundary,
			nodes_interior,
			weights_with_boundary,
			weights_interior,
			differentiation_matrix_with_boundary,
			differentiation_matrix_interior,
			quadrature_matrix,
			higher_order_diff_matrices_with_boundary,
			higher_order_diff_matrices_interior,
		)
	end
end

# Outer constructor
function StandardSpectralProperties(;
	p::Int,
	continuity_order::Int,
	nodes_with_boundary::AbstractVector,
	nodes_interior::AbstractVector,
	weights_with_boundary::AbstractVector,
	weights_interior::AbstractVector,
	differentiation_matrix_with_boundary::AbstractMatrix,
	differentiation_matrix_interior::AbstractMatrix,
	quadrature_matrix::AbstractMatrix,
	higher_order_diff_matrices_with_boundary::Vector{<:AbstractMatrix} = Matrix{Float64}[],
	higher_order_diff_matrices_interior::Vector{<:AbstractMatrix} = Matrix{Float64}[],
)
	T = promote_type(
		eltype(nodes_with_boundary),
		eltype(weights_with_boundary),
		eltype(differentiation_matrix_with_boundary),
	)

	return StandardSpectralProperties{T}(
		p,
		continuity_order,
		convert(Vector{T}, nodes_with_boundary),
		convert(Vector{T}, nodes_interior),
		convert(Vector{T}, weights_with_boundary),
		convert(Vector{T}, weights_interior),
		convert(Matrix{T}, differentiation_matrix_with_boundary),
		convert(Matrix{T}, differentiation_matrix_interior),
		convert(Matrix{T}, quadrature_matrix),
		convert(Vector{Matrix{T}}, higher_order_diff_matrices_with_boundary),
		convert(Vector{Matrix{T}}, higher_order_diff_matrices_interior),
	)
end

struct AffineTransform{T <: Number} <: AbstractKSTransform
	matrix::Matrix{T}
	translation::Vector{T}

	# Inner constructor with validation
	function AffineTransform(
		matrix::Union{Matrix{Bool}, Matrix{T}}, translation::Vector{T}
	) where {T <: Number}
		size(matrix, 1) == size(matrix, 2) ||
			throw(DimensionMismatch("Affine matrix must be square"))
		length(translation) == size(matrix, 1) || throw(
			DimensionMismatch(
				"Translation vector must have length equal to matrix dimensions"
			),
		)
		matrix_T = convert(Matrix{T}, matrix)
		return new{T}(matrix_T, translation)
	end
end

# Outer constructor for flexible input types
function AffineTransform(; matrix::AbstractMatrix, translation::AbstractVector)
	T = promote_type(eltype(matrix), eltype(translation))
	matrix_T = convert(Matrix{T}, matrix)
	translation_T = convert(Vector{T}, translation)
	return AffineTransform(matrix_T, translation_T)
end

struct NonlinearTransform <: AbstractKSTransform
	forward_map::Function
	inverse_map::Union{Nothing, Function}
	jacobian::Union{Nothing, Function}

	function NonlinearTransform(;
		forward_map::Function,
		inverse_map::Union{Nothing, Function} = nothing,
		jacobian::Union{Nothing, Function} = nothing)
		# Validation
		return new(forward_map, inverse_map, jacobian)
	end
end

# Outer constructor for flexible input

# CompositeTransform
struct CompositeTransform <: AbstractKSTransform
	transforms::Vector{AbstractKSTransform}

	# Inner constructor with validation
	function CompositeTransform(; transforms::Vector{<:AbstractKSTransform})
		isempty(transforms) &&
			throw(ArgumentError("CompositeTransform requires at least one transform"))
		return new(collect(AbstractKSTransform, transforms))
	end
end

# Constructor for direct transform input
function CompositeTransform(transforms::Vector{<:AbstractKSTransform})
	return CompositeTransform(; transforms = transforms)
end

# Constructor for varargs transform input
function CompositeTransform(transforms::AbstractKSTransform...)
	return CompositeTransform(collect(AbstractKSTransform, transforms))
end

# MultiLevelTransform
struct MultiLevelTransform <: AbstractKSTransform
	base_transform::AbstractKSTransform
	level_transforms::Vector{AbstractKSTransform}

	# Inner constructor with validation
	function MultiLevelTransform(;
		base_transform::AbstractKSTransform,
		level_transforms::Vector{AbstractKSTransform},
	)
		isempty(level_transforms) &&
			throw(ArgumentError("At least one level transform must be provided"))
		return new(base_transform, level_transforms)
	end
end

function MultiLevelTransform(
	base_transform::AbstractKSTransform,
	level_transforms::AbstractVector{AbstractKSTransform},
)
	return MultiLevelTransform(base_transform, Vector(level_transforms))
end
# Outer constructor for flexible input

# PrimitiveLevelSet
struct PrimitiveLevelSet <: AbstractLevelSetFunction
	func::Function
	gradient::Union{Nothing, Function}

	# Inner constructor with validation
	function PrimitiveLevelSet(;
		func::Function,
		gradient::Union{Nothing, Function} = nothing,
	)
		typeof(func) <: Function || throw(ArgumentError("func must be a function"))
		gradient !== nothing &&
			!(typeof(gradient) <: Function) &&
			throw(ArgumentError("gradient must be a function or nothing"))
		return new(func, gradient)
	end
end

# CompositeLevelSet
struct CompositeLevelSet <: AbstractLevelSetFunction
	operations::Vector{Function}
	primitives::Vector{AbstractLevelSetFunction}

	# Inner constructor with validation and type promotion
	function CompositeLevelSet(;
		operations::Vector{T},
		primitives::Vector{AbstractLevelSetFunction},
	) where {T}
		isempty(operations) && throw(ArgumentError("Operations cannot be empty"))
		isempty(primitives) && throw(ArgumentError("Primitives cannot be empty"))
		length(operations) == length(primitives) - 1 || throw(
			DimensionMismatch(
				"Number of operations must be one less than number of primitives"
			),
		)
		# new(promote_type(T, Function)[operations...], primitives)
		return new(operations, convert(Vector{AbstractLevelSetFunction}, primitives))
	end
end

# Outer constructor for a single operation and a vector of primitives with type promotion
function CompositeLevelSet(
	operation::T,
	primitives::Vector{AbstractLevelSetFunction},
) where {T}
	isempty(primitives) && throw(ArgumentError("Primitives cannot be empty"))
	length(primitives) > 1 || throw(ArgumentError("At least two primitives are required"))
	return CompositeLevelSet(promote_type(T, Function)[operation], primitives)
end

struct DomainMapping{T <: Number} <: AbstractDomainMapping
	forward_transform::AbstractKSTransform
	inverse_transform::AbstractKSTransform
	fictitious_system::AbstractKSCoordinateSystem
	physical_system::AbstractKSCoordinateSystem

	# Inner constructor with validation
	function DomainMapping{T}(;
		forward_transform::AbstractKSTransform,
		inverse_transform::AbstractKSTransform,
		fictitious_system::AbstractKSCoordinateSystem,
		physical_system::AbstractKSCoordinateSystem) where {T <: Number}
		# Basic type validation
		forward_transform isa AbstractKSTransform ||
			throw(ArgumentError("Invalid forward_transform type"))
		inverse_transform isa AbstractKSTransform ||
			throw(ArgumentError("Invalid inverse_transform type"))
		physical_system isa AbstractKSCoordinateSystem ||
			throw(ArgumentError("Invalid physical_system type"))
		fictitious_system isa AbstractKSCoordinateSystem ||
			throw(ArgumentError("Invalid fictitious_system type"))
		return new(forward_transform, inverse_transform, fictitious_system, physical_system)
	end
end

# Outer constructor for flexible input types
function DomainMapping(; kwargs...)
	return DomainMapping{Float64}(; kwargs...)
end

# 6. Coordinate system types
struct KSCartesianCoordinates{T <: Number, N} <: AbstractKSCoordinateSystem
	ranges::NTuple{N, Tuple{T, T}}
	active::NTuple{N, Bool}
	domains::NTuple{N, Tuple{T, T}}

	function KSCartesianCoordinates{T, N}(
		ranges::NTuple{
			N, Tuple{T, T}},
	) where {T <: Number, N}
		validated_ranges = ntuple(i -> validate_range(ranges[i], "dimension $i"), N)
		active = ntuple(_ -> true, Val(N))
		return new{T, N}(validated_ranges, active, validated_ranges)
	end
end

# Outer constructor for type promotion and 1D case
function KSCartesianCoordinates(
	ranges::Union{Tuple{Tuple{T1, T2}},
		NTuple{N, Tuple{T1, T2}}}) where {N, T1 <: Number, T2 <: Number}
	T = promote_type(T1, T2)
	promoted_ranges = map(r -> (convert(T, r[1]), convert(T, r[2])), ranges)
	return KSCartesianCoordinates{T, length(ranges)}(promoted_ranges)
end

# Polar Coordinates
struct KSPolarCoordinates{T <: Number} <: AbstractKSCoordinateSystem
	r::Tuple{T, T}
	theta::Union{Tuple{T, T}, Nothing}
	active::NTuple{2, Bool}
	domains::NTuple{2, Union{Tuple{T, T}, Nothing}}

	function KSPolarCoordinates{T}(
		r::Tuple{T, T},
		theta::Union{Tuple{T, T}, Nothing} = nothing,
	) where {T <: Number}
		validated_r = validate_range(r, "r", zero(T))
		validated_theta =
			theta !== nothing ?
			validate_range(theta, "theta", zero(T), 2T(π)) : nothing
		active = (true, theta !== nothing)
		return new{T}(validated_r, validated_theta, active, (validated_r, validated_theta))
	end
end

# Outer constructor for type promotion, handling inactive domains
function KSPolarCoordinates(
	r::Tuple{T1, T2},
	theta::Union{Tuple{T3, T4}, Nothing} = nothing,
) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number}
	T = promote_type(
		T1,
		T2,
		theta !== nothing ? (T3 <: Number ? T3 : T1) : T1,
		theta !== nothing ? (T4 <: Number ? T4 : T2) : T2,
	)
	promoted_r = (convert(T, r[1]), convert(T, r[2]))
	promoted_theta =
		theta !== nothing ? (convert(T, theta[1]), convert(T, theta[2])) :
		nothing
	return KSPolarCoordinates{T}(promoted_r, promoted_theta)
end

# Cylindrical Coordinates
struct KSCylindricalCoordinates{T <: Number} <: AbstractKSCoordinateSystem
	r::Tuple{T, T}
	theta::Union{Tuple{T, T}, Nothing}
	z::Union{Tuple{T, T}, Nothing}
	active::NTuple{3, Bool}
	domains::NTuple{3, Union{Tuple{T, T}, Nothing}}

	function KSCylindricalCoordinates{T}(
		r::Tuple{T, T},
		theta::Union{Tuple{T, T}, Nothing},
		z::Union{Tuple{T, T}, Nothing},
	) where {T <: Number}
		validated_r = validate_range(r, "r", zero(T))
		validated_theta =
			theta !== nothing ?
			validate_range(theta, "theta", zero(T), 2T(π)) : nothing
		validated_z = z !== nothing ? validate_range(z, "z") : nothing
		active = (true, theta !== nothing, z !== nothing)
		return new{T}(
			validated_r,
			validated_theta,
			validated_z,
			active,
			(validated_r, validated_theta, validated_z),
		)
	end
end

# Outer constructor for type promotion, handling inactive domains
function KSCylindricalCoordinates(
	r::Tuple{T1, T2},
	theta::Union{Tuple{T3, T4}, Nothing},
	z::Union{Tuple{T5, T6}, Nothing},
) where {
	T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number, T5 <: Number, T6 <: Number}
	T = promote_type(
		T1,
		T2,
		theta !== nothing ? (T3 <: Number ? T3 : T1) : T1,
		theta !== nothing ? (T4 <: Number ? T4 : T2) : T2,
		z !== nothing ? (T5 <: Number ? T5 : T1) : T1,
		z !== nothing ? (T6 <: Number ? T6 : T2) : T2,
	)
	promoted_r = (convert(T, r[1]), convert(T, r[2]))
	promoted_theta =
		theta !== nothing ? (convert(T, theta[1]), convert(T, theta[2])) :
		nothing
	promoted_z = z !== nothing ? (convert(T, z[1]), convert(T, z[2])) : nothing
	return KSCylindricalCoordinates{T}(promoted_r, promoted_theta, promoted_z)
end

# Spherical Coordinates
struct KSSphericalCoordinates{T <: Number} <: AbstractKSCoordinateSystem
	r::Tuple{T, T}
	theta::Union{Tuple{T, T}, Nothing}
	phi::Union{Tuple{T, T}, Nothing}
	active::NTuple{3, Bool}
	domains::NTuple{3, Union{Tuple{T, T}, Nothing}}

	function KSSphericalCoordinates{T}(
		r::Tuple{T, T},
		theta::Union{Tuple{T, T}, Nothing},
		phi::Union{Tuple{T, T}, Nothing},
	) where {T <: Number}
		validated_r = validate_range(r, "r", zero(T))
		validated_theta =
			theta !== nothing ?
			validate_range(theta, "theta", zero(T), T(π)) : nothing
		validated_phi =
			phi !== nothing ? validate_range(phi, "phi", zero(T), 2T(π)) :
			nothing
		active = (true, theta !== nothing, phi !== nothing)
		return new{T}(
			validated_r,
			validated_theta,
			validated_phi,
			active,
			(validated_r, validated_theta, validated_phi),
		)
	end
end

# Outer constructor for type promotion, handling inactive domains
function KSSphericalCoordinates(
	r::Tuple{T1, T2},
	theta::Union{Tuple{T3, T4}, Nothing},
	phi::Union{Tuple{T5, T6}, Nothing},
) where {
	T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number, T5 <: Number, T6 <: Number}
	T = promote_type(
		T1,
		T2,
		theta !== nothing ? (T3 <: Number ? T3 : T1) : T1,
		theta !== nothing ? (T4 <: Number ? T4 : T2) : T2,
		phi !== nothing ? (T5 <: Number ? T5 : T1) : T1,
		phi !== nothing ? (T6 <: Number ? T6 : T2) : T2,
	)
	promoted_r = (convert(T, r[1]), convert(T, r[2]))
	promoted_theta =
		theta !== nothing ? (convert(T, theta[1]), convert(T, theta[2])) :
		nothing
	promoted_phi = phi !== nothing ? (convert(T, phi[1]), convert(T, phi[2])) : nothing
	return KSSphericalCoordinates{T}(promoted_r, promoted_theta, promoted_phi)
end

# 7. Cell and mesh types
struct StandardKSCell{T <: Number, N}
	p::NTuple{N, Int}
	level::Int
	continuity_order::NTuple{N, Int}
	nodes_with_boundary::NTuple{N, Vector{T}}
	nodes_interior::NTuple{N, Vector{T}}
	weights_with_boundary::NTuple{N, Vector{T}}
	weights_interior::NTuple{N, Vector{T}}
	differentiation_matrix_with_boundary::NTuple{N, Matrix{T}}
	differentiation_matrix_interior::NTuple{N, Matrix{T}}
	quadrature_matrix::NTuple{N, Matrix{T}}
	higher_order_diff_matrices_with_boundary::NTuple{N, Vector{Matrix{T}}}
	higher_order_diff_matrices_interior::NTuple{N, Vector{Matrix{T}}}

	# Inner constructor (unchanged)
	function StandardKSCell{T, N}(
		p::NTuple{N, Int},
		level::Int,
		continuity_order::NTuple{N, Int},
		nodes_with_boundary::NTuple{N, AbstractVector},
		nodes_interior::NTuple{N, AbstractVector},
		weights_with_boundary::NTuple{N, AbstractVector},
		weights_interior::NTuple{N, AbstractVector},
		differentiation_matrix_with_boundary::NTuple{N, AbstractMatrix},
		differentiation_matrix_interior::NTuple{N, AbstractMatrix},
		quadrature_matrix::NTuple{N, AbstractMatrix},
		higher_order_diff_matrices_with_boundary::NTuple{
			N,
			Union{Nothing, AbstractVector{<:AbstractMatrix}},
		} = ntuple(_ -> nothing, N),
		higher_order_diff_matrices_interior::NTuple{
			N,
			Union{Nothing, AbstractVector{<:AbstractMatrix}},
		} = ntuple(_ -> nothing, N)) where {T <: Number, N}
		# Validation
		N > 0 || throw(ArgumentError("Number of dimensions must be positive"))
		all(x -> x >= 1, p) ||
			throw(ArgumentError("All polynomial degrees must be at least 1"))
		level >= 0 || throw(ArgumentError("Level must be non-negative"))
		all(x -> x >= 0, continuity_order) ||
			throw(ArgumentError("Continuity orders must be non-negative"))

		for i in 1:N
			length(nodes_with_boundary[i]) == p[i] + 2 || throw(
				DimensionMismatch(
					"Length of nodes_with_boundary[$i] ($(length(nodes_with_boundary[i]))) must be p[$i] + 2 ($(p[i] + 2))"
				),
			)
			length(nodes_interior[i]) == p[i] || throw(
				DimensionMismatch(
					"Length of nodes_interior[$i] ($(length(nodes_interior[i]))) must be p[$i] ($(p[i]))"
				),
			)
			length(weights_with_boundary[i]) == p[i] + 2 || throw(
				DimensionMismatch(
					"Length of weights_with_boundary[$i] ($(length(weights_with_boundary[i]))) must be p[$i] + 2 ($(p[i] + 2))"
				),
			)
			length(weights_interior[i]) == p[i] || throw(
				DimensionMismatch(
					"Length of weights_interior[$i] ($(length(weights_interior[i]))) must be p[$i] ($(p[i]))"
				),
			)
			size(differentiation_matrix_with_boundary[i]) == (p[i] + 2, p[i] + 2) || throw(
				DimensionMismatch(
					"Size of differentiation_matrix_with_boundary[$i] ($(size(differentiation_matrix_with_boundary[i]))) must be (p[$i] + 2, p[$i] + 2)"
				),
			)
			size(differentiation_matrix_interior[i]) == (p[i], p[i]) || throw(
				DimensionMismatch(
					"Size of differentiation_matrix_interior[$i] ($(size(differentiation_matrix_interior[i]))) must be (p[$i], p[$i])"
				),
			)
			size(quadrature_matrix[i]) == (p[i], p[i]) || throw(
				DimensionMismatch(
					"Size of quadrature_matrix[$i] ($(size(quadrature_matrix[i]))) must be (p[$i], p[$i])"
				),
			)
		end

		# Convert higher-order matrices, using empty vector if nothing is provided
		convert_higher_order =
			(x) -> isnothing(x) ? Vector{Matrix{T}}() :
				   convert(Vector{Matrix{T}}, x)

		return new{T, N}(
			p,
			level,
			continuity_order,
			Tuple(convert(Vector{T}, x) for x in nodes_with_boundary),
			Tuple(convert(Vector{T}, x) for x in nodes_interior),
			Tuple(convert(Vector{T}, x) for x in weights_with_boundary),
			Tuple(convert(Vector{T}, x) for x in weights_interior),
			Tuple(convert(Matrix{T}, x) for x in differentiation_matrix_with_boundary),
			Tuple(convert(Matrix{T}, x) for x in differentiation_matrix_interior),
			Tuple(convert(Matrix{T}, x) for x in quadrature_matrix),
			Tuple(
				convert_higher_order(x) for x in higher_order_diff_matrices_with_boundary
			),
			Tuple(convert_higher_order(x) for x in higher_order_diff_matrices_interior),
		)
	end
end

# Outer constructor

# Updated outer constructor
function StandardKSCell(;
	p::Union{Int, NTuple{N, Int} where N},
	level::Int,
	continuity_order::Union{Int, NTuple{N, Int} where N},
	nodes_with_boundary::Union{AbstractVector, NTuple{N, AbstractVector} where N},
	nodes_interior::Union{AbstractVector, NTuple{N, AbstractVector} where N},
	weights_with_boundary::Union{AbstractVector, NTuple{N, AbstractVector} where N},
	weights_interior::Union{AbstractVector, NTuple{N, AbstractVector} where N},
	differentiation_matrix_with_boundary::Union{
		AbstractMatrix,
		NTuple{N, AbstractMatrix} where N,
	},
	differentiation_matrix_interior::Union{
		AbstractMatrix, NTuple{N, AbstractMatrix} where N},
	quadrature_matrix::Union{AbstractMatrix, NTuple{N, AbstractMatrix} where N},
	higher_order_diff_matrices_with_boundary::Union{
		Nothing,
		AbstractVector{<:AbstractMatrix},
		NTuple{N, Union{Nothing, AbstractVector{<:AbstractMatrix}}} where N,
	} = nothing,
	higher_order_diff_matrices_interior::Union{
		Nothing,
		AbstractVector{<:AbstractMatrix},
		NTuple{N, Union{Nothing, AbstractVector{<:AbstractMatrix}}} where N,
	} = nothing)
	# Determine N from the input
	N = if p isa Int
		1
	else
		length(p)
	end

	# Convert single values to tuples if necessary
	p_tuple = p isa Int ? (p,) : Tuple(p)
	continuity_order_tuple =
		continuity_order isa Int ? (continuity_order,) :
		Tuple(continuity_order)
	nodes_with_boundary_tuple = if nodes_with_boundary isa AbstractVector
		(nodes_with_boundary,)
	else
		Tuple(nodes_with_boundary)
	end
	nodes_interior_tuple =
		nodes_interior isa AbstractVector ? (nodes_interior,) :
		Tuple(nodes_interior)
	weights_with_boundary_tuple = if weights_with_boundary isa AbstractVector
		(weights_with_boundary,)
	else
		Tuple(weights_with_boundary)
	end
	weights_interior_tuple =
		weights_interior isa AbstractVector ? (weights_interior,) :
		Tuple(weights_interior)
	differentiation_matrix_with_boundary_tuple =
		if differentiation_matrix_with_boundary isa
			AbstractMatrix
			(differentiation_matrix_with_boundary,)
		else
			Tuple(differentiation_matrix_with_boundary)
		end
	differentiation_matrix_interior_tuple =
		if differentiation_matrix_interior isa
			AbstractMatrix
			(differentiation_matrix_interior,)
		else
			Tuple(differentiation_matrix_interior)
		end
	quadrature_matrix_tuple = if quadrature_matrix isa AbstractMatrix
		(quadrature_matrix,)
	else
		Tuple(quadrature_matrix)
	end

	higher_order_diff_matrices_with_boundary_tuple =
		if higher_order_diff_matrices_with_boundary ===
			nothing
			ntuple(_ -> nothing, N)
		elseif higher_order_diff_matrices_with_boundary isa AbstractVector
			(higher_order_diff_matrices_with_boundary,)
		else
			Tuple(higher_order_diff_matrices_with_boundary)
		end

	higher_order_diff_matrices_interior_tuple =
		if higher_order_diff_matrices_interior ===
			nothing
			ntuple(_ -> nothing, N)
		elseif higher_order_diff_matrices_interior isa AbstractVector
			(higher_order_diff_matrices_interior,)
		else
			Tuple(higher_order_diff_matrices_interior)
		end

	# Infer the numeric type T from the input arguments
	T = promote_type(
		map(eltype, nodes_with_boundary_tuple)...,
		map(eltype, nodes_interior_tuple)...,
		map(eltype, weights_with_boundary_tuple)...,
		map(eltype, weights_interior_tuple)...,
		map(eltype, differentiation_matrix_with_boundary_tuple)...,
		map(eltype, differentiation_matrix_interior_tuple)...,
		map(eltype, quadrature_matrix_tuple)...,
	)

	# Call the inner constructor with the inferred types and converted inputs
	return StandardKSCell{T, N}(
		p_tuple,
		level,
		continuity_order_tuple,
		nodes_with_boundary_tuple,
		nodes_interior_tuple,
		weights_with_boundary_tuple,
		weights_interior_tuple,
		differentiation_matrix_with_boundary_tuple,
		differentiation_matrix_interior_tuple,
		quadrature_matrix_tuple,
		higher_order_diff_matrices_with_boundary_tuple,
		higher_order_diff_matrices_interior_tuple,
	)
end

mutable struct KSCell{T <: Number, N} <: AbstractKSCell{T, N}
	id::Int
	p::NTuple{N, Int}  # Polynomial orders across N dimensions
	level::Int
	continuity_order::NTuple{N, Int}
	standard_cell_key::Tuple{NTuple{N, Int}, Int}  # Mapping to (p, level)
	neighbors::Dict{Symbol, Int}
	node_map::Dict{NTuple{N, Int}, Int}
	tensor_product_mask::NTuple{N, AbstractArray{Bool}}
	boundary_connectivity::Dict{Symbol, Int}
	error_estimate::T
	legendre_decay_rate::T
	is_leaf::Bool
	is_fictitious::Bool
	refinement_options::Union{Nothing, Dict{Symbol, Any}}
	parent_id::Union{Nothing, Int}
	child_ids::Union{Nothing, Vector{Int}}
	b_spline_coefficients::Union{Nothing, NTuple{N, AbstractVector{T}}}
	b_spline_knots::Union{Nothing, NTuple{N, AbstractVector{T}}}

	# Inner constructor with validation
	function KSCell{T, N}(
		id::Int,
		p::NTuple{N, Int},
		level::Int,
		continuity_order::NTuple{N, Int},
		standard_cell_key::Tuple{NTuple{N, Int}, Int},
		neighbors::Dict{Symbol, Int},
		node_map::Dict{NTuple{N, Int}, Int},
		tensor_product_mask::NTuple{N, AbstractArray{Bool}},
		boundary_connectivity::Dict{Symbol, Int},
		error_estimate::T,
		legendre_decay_rate::T,
		is_leaf::Bool,
		is_fictitious::Bool,
		refinement_options::Union{Nothing, Dict{Symbol, Any}},
		parent_id::Union{Nothing, Int},
		child_ids::Union{Nothing, Vector{Int}},
		b_spline_coefficients::Union{Nothing, NTuple{N, AbstractVector{T}}} = nothing,
		b_spline_knots::Union{Nothing, NTuple{N, AbstractVector{T}}} = nothing,
	) where {T <: Number, N}

		# Validation checks

		id > 0 || throw(ArgumentError("Cell ID must be positive"))
		level >= 0 || throw(ArgumentError("Level must be non-negative"))
		length(p) == N ||
			throw(ArgumentError("Length of polynomial degree tuple must match N"))
		length(continuity_order) == N ||
			throw(ArgumentError("Length of continuity order tuple must match N"))
		length(p) == N ||
			throw(ArgumentError("Polynomial degree tuple must have length $N"))

		return new(
			id,
			p,
			level,
			continuity_order,
			standard_cell_key,
			neighbors,
			node_map,
			tensor_product_mask,
			boundary_connectivity,
			error_estimate,
			legendre_decay_rate,
			is_leaf,
			is_fictitious,
			refinement_options,
			parent_id,
			child_ids,
			b_spline_coefficients,
			b_spline_knots,
		)
	end
end

# Outer constructor for flexible initialization
function KSCell(;
	id::Int,
	p::NTuple{N, Int},
	level::Int,
	continuity_order::NTuple{N, Int},
	standard_cell_key::Union{Nothing, Tuple{NTuple{N, Int}, Int}} = nothing,
	neighbors::Dict{Symbol, Int},
	node_map::Dict{NTuple{N, Int}, Int},
	tensor_product_mask::NTuple{N, AbstractArray{Bool}},
	boundary_connectivity::Dict{Symbol, Int},
	error_estimate::Real = 0.0,
	legendre_decay_rate::Real = 0.0,
	is_leaf::Bool = true,
	is_fictitious::Bool = false,
	refinement_options::Union{Nothing, Dict{Symbol, Any}} = nothing,
	parent_id::Union{Nothing, Int} = nothing,
	child_ids::Union{Nothing, Vector{Int}} = nothing,
	b_spline_coefficients::Union{Nothing, NTuple{N, AbstractVector}} = nothing,
	b_spline_knots::Union{Nothing, NTuple{N, AbstractVector}} = nothing,
) where {N}

	# Promote types of error_estimate and legendre_decay_rate
	promoted_type = promote_type(typeof(error_estimate), typeof(legendre_decay_rate))

	# Automatically generate the standard_cell_key if not provided
	if standard_cell_key === nothing
		standard_cell_key = (p, level)
	end

	# Call the inner constructor with appropriate type promotion
	return KSCell{promoted_type, N}(
		id,
		p,
		level,
		continuity_order,
		standard_cell_key,
		neighbors,
		node_map,
		tensor_product_mask,
		boundary_connectivity,
		promoted_type(error_estimate),
		promoted_type(legendre_decay_rate),
		is_leaf,
		is_fictitious,
		refinement_options,
		parent_id,
		child_ids,
		b_spline_coefficients,
		b_spline_knots,
	)
end

mutable struct KSMesh{T <: Number, N}
	cells::Vector{KSCell{T, N}}
	tensor_product_masks::Vector{NTuple{N, Vector{Bool}}}
	node_map::Dict{Tuple{Int, NTuple{N, Int}}, Int}
	standard_cell_keys::Set{Tuple{NTuple{N, Int}, Int}}
	global_error_estimate::T
	physical_cells::Vector{Int}
	boundary_cells::Dict{Symbol, Vector{Int}}
	physical_domain::Function
	characteristic_function::Union{Nothing, Function}
	continuity_constraints::Union{Nothing, Dict{Symbol, Matrix{T}}}
	transformation_data::Union{Nothing, AbstractKSTransform}

	# Inner constructor (unchanged)
	function KSMesh{T, N}(
		cells::Vector{KSCell{T, N}},
		global_error_estimate::T,
		boundary_cells::Dict{Symbol, Vector{Int}},
		physical_domain::Function;
		characteristic_function::Union{Nothing, Function} = nothing,
		continuity_constraints::Union{Nothing, Dict{Symbol, Matrix{T}}} = nothing,
		transformation_data::Union{Nothing, AbstractKSTransform} = nothing,
	) where {T <: Number, N}
		# Validation
		isempty(cells) && throw(ArgumentError("Cells cannot be empty"))

		tensor_product_masks = [cell.tensor_product_mask for cell in cells]
		node_map = Dict{Tuple{Int, NTuple{N, Int}}, Int}()
		for (i, cell) in enumerate(cells)
			for (local_index, global_index) in cell.node_map
				node_map[(i, local_index)] = global_index
			end
		end
		standard_cell_keys = Set([(cell.p, cell.level) for cell in cells])
		physical_cells = detect_physical_cells(cells, physical_domain)

		isempty(physical_cells) && throw(ArgumentError("No physical cells detected"))

		cell_ids = [cell.id for cell in cells]
		length(unique(cell_ids)) == length(cell_ids) ||
			throw(ArgumentError("Duplicate cell IDs detected"))

		for (_, boundary_cell_ids) in boundary_cells
			all(id -> id in cell_ids, boundary_cell_ids) ||
				throw(ArgumentError("Invalid cell ID in boundary cells"))
		end

		return new{T, N}(
			cells,
			tensor_product_masks,
			node_map,
			standard_cell_keys,
			global_error_estimate,
			physical_cells,
			boundary_cells,
			physical_domain,
			characteristic_function,
			continuity_constraints,
			transformation_data,
		)
	end
end

# Updated outer constructor with keyword arguments
function KSMesh(;
	cells::Vector{KSCell{T, N}},
	global_error_estimate::Real,
	boundary_cells::Dict{Symbol, Vector{Int}},
	physical_domain::Function,
	characteristic_function::Union{Nothing, Function} = nothing,
	continuity_constraints::Union{Nothing, Dict{Symbol, Matrix{T}}} = nothing,
	transformation_data::Union{Nothing, AbstractKSTransform} = nothing,
) where {T <: Number, N}
	return KSMesh{T, N}(
		cells,
		T(global_error_estimate),
		boundary_cells,
		physical_domain;
		characteristic_function = characteristic_function,
		continuity_constraints = continuity_constraints,
		transformation_data = transformation_data,
	)
end

function get_unique_standard_cell_keys(mesh::KSMesh)
	return mesh.standard_cell_keys
end

function detect_physical_cells(
	cells::Vector{KSCell{T, N}},
	physical_domain::Function,
)::Vector{Int} where {T <: Number, N}
	physical_cells = Int[]
	for (i, cell) in enumerate(cells)
		if is_physical_cell(cell, physical_domain)
			push!(physical_cells, i)
		end
	end
	return physical_cells
end

function is_physical_cell(
	cell::KSCell{T, N},
	physical_domain::Function,
)::Bool where {T <: Number, N}
	corners = get_cell_corners(cell)
	result = any(physical_domain.(corners))
	return result
end

function get_cell_corners(cell::KSCell{T, N}) where {T <: Number, N}
	# Assuming the cell is defined on a unit hypercube [0, 1]^N
	corners = Vector{NTuple{N, T}}()
	for corner_indices in Iterators.product(ntuple(i -> [0, 1], N)...)
		push!(corners, ntuple(i -> T(corner_indices[i]), N))
	end
	return corners
end
mutable struct KSBasisFunction <: AbstractKSBasisFunction
	id::Int
	function_handle::Function
	contribution::Number
	is_removable::Bool

	function KSBasisFunction(
		id::Int,
		function_handle::Function = x -> x,
		contribution = 0,
		is_removable::Bool = false,
	)
		if id <= 0
			throw(ArgumentError("Basis function ID must be positive"))
		end
		return new(id, function_handle, contribution, is_removable)
	end
end

# 8. Problem types
struct KSProblem{T <: Number, N} <: AbstractKSProblem
	equation::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{Nothing, AbstractVector{T}, Function}
	tspan::Union{Nothing, Tuple{T, T}}
	continuity_order::Int
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSProblem(;
		equation::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T1, T2}},
		coordinate_system::AbstractKSCoordinateSystem,
		initial_conditions::Union{Nothing, AbstractVector, Function} = nothing,
		tspan::Union{Nothing, Tuple{<:Number, <:Number}} = nothing,
		continuity_order::Int = 2,
		num_vars::Int = 1,
	) where {N, T1 <: Number, T2 <: Number}
		T = promote_type(T1, T2)
		if initial_conditions isa AbstractVector
			T = promote_type(T, eltype(initial_conditions))
		end
		if tspan !== nothing
			T = promote_type(T, typeof(tspan[1]), typeof(tspan[2]))
		end

		domain_T = NTuple{N, Tuple{T, T}}(map(d -> (T(d[1]), T(d[2])), domain))

		if !isa(equation, Function)
			throw(ArgumentError("equation must be a Function, but got $(typeof(equation))"))
		end
		if !all(isa.(boundary_conditions, AbstractKSBoundaryCondition))
			throw(
				ArgumentError(
					"boundary_conditions must be an AbstractVector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if continuity_order < 0
			throw(
				ArgumentError(
					"Continuity order must be non-negative, but got $continuity_order"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end

		initial_conditions_T = if isa(initial_conditions, AbstractVector)
			convert(AbstractVector{T}, initial_conditions)
		else
			initial_conditions
		end

		tspan_T = if tspan !== nothing
			(T(tspan[1]), T(tspan[2]))
		else
			nothing
		end

		return new{T, N}(
			equation,
			boundary_conditions,
			domain_T,
			initial_conditions_T,
			tspan_T,
			continuity_order,
			coordinate_system,
			num_vars,
		)
	end
end

struct KSPDEProblem{T <: Number, N} <: AbstractKSProblem
	pde::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSPDEProblem(;
		pde::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_conditions::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = 1,
	) where {T <: Number, N}
		if !isa(pde, Function)
			throw(ArgumentError("PDE must be a Function, but got $(typeof(pde))"))
		end
		if !isa(boundary_conditions, AbstractVector{<:AbstractKSBoundaryCondition})
			throw(
				ArgumentError(
					"boundary_conditions must be a Vector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(
				ArgumentError(
					"domain must be an NTuple of Tuples, but got $(typeof(domain))"
				),
			)
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		return new{T, N}(
			pde,
			boundary_conditions,
			domain,
			initial_conditions,
			tspan,
			coordinate_system,
			num_vars,
		)
	end
end

struct KSODEProblem{T <: Number, N} <: AbstractKSProblem
	ode::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSODEProblem(;
		ode::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_conditions::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = length(initial_conditions),
	) where {T <: Number, N}
		if !isa(ode, Function)
			throw(ArgumentError("ODE must be a Function, but got $(typeof(ode))"))
		end
		if !isa(boundary_conditions, AbstractVector{<:AbstractKSBoundaryCondition})
			throw(
				ArgumentError(
					"boundary_conditions must be a Vector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(
				ArgumentError(
					"domain must be an NTuple of Tuples, but got $(typeof(domain))"
				),
			)
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(
				ArgumentError(
					"initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"
				),
			)
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		return new{T, N}(
			ode,
			boundary_conditions,
			domain,
			initial_conditions,
			tspan,
			coordinate_system,
			num_vars,
		)
	end
end

struct KSDAEProblem{T <: Number, N} <: AbstractKSProblem
	dae::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int
	num_algebraic_vars::Int

	function KSDAEProblem(;
		dae::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_conditions::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = length(initial_conditions),
		num_algebraic_vars::Int = 0,
	) where {T <: Number, N}
		if !isa(dae, Function)
			throw(ArgumentError("DAE must be a Function, but got $(typeof(dae))"))
		end
		if !isa(boundary_conditions, AbstractVector{<:AbstractKSBoundaryCondition})
			throw(
				ArgumentError(
					"boundary_conditions must be a Vector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(
				ArgumentError(
					"domain must be an NTuple of Tuples, but got $(typeof(domain))"
				),
			)
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(
				ArgumentError(
					"initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"
				),
			)
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		if num_algebraic_vars < 0 || num_algebraic_vars >= num_vars
			throw(
				ArgumentError(
					"num_algebraic_vars must be non-negative and less than num_vars, but got $num_algebraic_vars"
				),
			)
		end
		return new{T, N}(
			dae,
			boundary_conditions,
			domain,
			initial_conditions,
			tspan,
			coordinate_system,
			num_vars,
			num_algebraic_vars,
		)
	end
end

struct KSBVDAEProblem{T <: Number, N} <: AbstractKSProblem
	f::Function
	g::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	algebraic_vars::AbstractVector{Bool}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSBVDAEProblem(;
		f::Function,
		g::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T1, T2}},
		initial_conditions::AbstractVector{T3},
		algebraic_vars::AbstractVector{Bool},
		tspan::Tuple{T4, T5},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = length(initial_conditions),
	) where {N, T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number, T5 <: Number}
		T = promote_type(T1, T2, T3, T4, T5)
		domain_T = NTuple{N, Tuple{T, T}}(map(d -> (T(d[1]), T(d[2])), domain))
		initial_conditions_T = convert(AbstractVector{T}, initial_conditions)
		tspan_T = (T(tspan[1]), T(tspan[2]))

		if !isa(f, Function) || !isa(g, Function)
			throw(ArgumentError("f and g must be Functions"))
		end
		if !isa(algebraic_vars, AbstractVector{Bool})
			throw(
				ArgumentError(
					"algebraic_vars must be a vector of Bool, but got $(typeof(algebraic_vars))"
				),
			)
		end
		if length(initial_conditions) != length(algebraic_vars)
			throw(
				ArgumentError(
					"Length of initial_conditions must match length of algebraic_vars"
				),
			)
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end

		return new{T, N}(
			f,
			g,
			boundary_conditions,
			domain_T,
			initial_conditions_T,
			algebraic_vars,
			tspan_T,
			coordinate_system,
			num_vars,
		)
	end
end

struct KSIDEProblem{T <: Number, N} <: AbstractKSProblem
	ide::Function
	kernel::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSIDEProblem(;
		ide::Function,
		kernel::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_conditions::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = length(initial_conditions),
	) where {T <: Number, N}
		if !isa(ide, Function)
			throw(ArgumentError("IDE must be a Function, but got $(typeof(ide))"))
		end
		if !isa(kernel, Function)
			throw(ArgumentError("Kernel must be a Function, but got $(typeof(kernel))"))
		end
		if !isa(boundary_conditions, AbstractVector{<:AbstractKSBoundaryCondition})
			throw(
				ArgumentError(
					"boundary_conditions must be a Vector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(
				ArgumentError(
					"domain must be an NTuple of Tuples, but got $(typeof(domain))"
				),
			)
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(
				ArgumentError(
					"initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"
				),
			)
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		return new{T, N}(
			ide,
			kernel,
			boundary_conditions,
			domain,
			initial_conditions,
			tspan,
			coordinate_system,
			num_vars,
		)
	end
end

struct KSPIDEProblem{T <: Number, N} <: AbstractKSProblem
	pide::Function
	kernel::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSPIDEProblem(;
		pide::Function,
		kernel::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_conditions::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = if isa(initial_conditions, AbstractVector)
			length(initial_conditions)
		else
			1
		end,
	) where {T <: Number, N}
		if !isa(pide, Function)
			throw(ArgumentError("PIDE must be a Function, but got $(typeof(pide))"))
		end
		if !isa(kernel, Function)
			throw(ArgumentError("kernel must be a Function, but got $(typeof(kernel))"))
		end
		if !isa(boundary_conditions, AbstractVector{<:AbstractKSBoundaryCondition})
			throw(
				ArgumentError(
					"boundary_conditions must be a Vector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(
				ArgumentError(
					"domain must be an NTuple of Tuples, but got $(typeof(domain))"
				),
			)
		end
		if !(
			isa(initial_conditions, AbstractVector{T}) || isa(initial_conditions, Function)
		)
			throw(
				ArgumentError(
					"initial_conditions must be an AbstractVector or Function, but got $(typeof(initial_conditions))"
				),
			)
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		return new{T, N}(
			pide,
			kernel,
			boundary_conditions,
			domain,
			initial_conditions,
			tspan,
			coordinate_system,
			num_vars,
		)
	end
end

struct KSMovingBoundaryPDEProblem{T <: Number, N} <: AbstractKSProblem
	pde::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	boundary_motion::Function
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSMovingBoundaryPDEProblem(;
		pde::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_conditions::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		boundary_motion::Function,
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = length(initial_conditions),
	) where {T <: Number, N}
		if !isa(pde, Function)
			throw(ArgumentError("PDE must be a Function, but got $(typeof(pde))"))
		end
		if !isa(boundary_conditions, AbstractVector{<:AbstractKSBoundaryCondition})
			throw(
				ArgumentError(
					"boundary_conditions must be a Vector of AbstractKSBoundaryCondition, but got $(typeof(boundary_conditions))"
				),
			)
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(
				ArgumentError(
					"domain must be an NTuple of Tuples, but got $(typeof(domain))"
				),
			)
		end
		if !(
			isa(initial_conditions, AbstractVector{T}) || isa(initial_conditions, Function)
		)
			throw(
				ArgumentError(
					"initial_conditions must be an AbstractVector or Function, but got $(typeof(initial_conditions))"
				),
			)
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(boundary_motion, Function)
			throw(
				ArgumentError(
					"boundary_motion must be a Function, but got $(typeof(boundary_motion))"
				),
			)
		end
		if !isa(coordinate_system, AbstractKSCoordinateSystem)
			throw(
				ArgumentError(
					"coordinate_system must be an AbstractKSCoordinateSystem, but got $(typeof(coordinate_system))"
				),
			)
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		return new{T, N}(
			pde,
			boundary_conditions,
			domain,
			initial_conditions,
			tspan,
			boundary_motion,
			coordinate_system,
			num_vars,
		)
	end
end

mutable struct KSBVPProblem{T <: Number, N} <: AbstractKSProblem
	bvp::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_guess::Union{AbstractVector{T}, Function}
	tspan::Union{Nothing, Tuple{T, T}}  # Added optional tspan
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	# Constructor with tspan support
	function KSBVPProblem(;
		bvp::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_guess::Union{AbstractVector{T}, Function},
		tspan::Union{Nothing, Tuple{T, T}} = nothing,
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = 1
	) where {T <: Number, N}
		 # Validate num_vars
        if num_vars < 1
            throw(ArgumentError("num_vars must be at least 1, got $num_vars"))
        end

        # Validate other inputs
        if !isa(bvp, Function)
            throw(ArgumentError("bvp must be a Function"))
        end
        if isempty(boundary_conditions)
            throw(ArgumentError("boundary_conditions cannot be empty"))
        end
        if !all(bc -> bc isa AbstractKSBoundaryCondition, boundary_conditions)
            throw(ArgumentError("all boundary conditions must be subtypes of AbstractKSBoundaryCondition"))
        end

        # Validate tspan if provided
        if tspan !== nothing
            if !(tspan isa Tuple{T,T}) || tspan[1] >= tspan[2]
                throw(ArgumentError("tspan must be a tuple (t0,tf) with t0 < tf"))
            end
        end

        return new{T, N}(
            bvp, boundary_conditions, domain, initial_guess,
            tspan, coordinate_system, num_vars
        )
	end
end

struct KSCoupledProblem <: AbstractKSProblem
	problems::AbstractVector{<:AbstractKSProblem}
	coupling_terms::Matrix{Union{Nothing, Function}}
	num_vars::Int

	function KSCoupledProblem(;
		problems::AbstractVector{<:AbstractKSProblem},
		coupling_terms::Matrix{Union{Nothing, Function}},
	)
		if !isa(problems, AbstractVector{<:AbstractKSProblem})
			throw(
				ArgumentError(
					"problems must be an AbstractVector of AbstractKSProblem, but got $(typeof(problems))"
				),
			)
		end
		if size(coupling_terms, 1) != size(coupling_terms, 2) ||
			size(coupling_terms, 1) != length(problems)
			throw(ArgumentError("Inconsistent coupling terms dimensions"))
		end
		num_vars = sum(p.num_vars for p in problems)
		return new(problems, coupling_terms, num_vars)
	end
end

struct KSDiscretizedProblem{T <: Number, N, M, MT <: AbstractMatrix} <: AbstractKSProblem
	time_nodes::AbstractVector{T}
	spatial_nodes::NTuple{N, AbstractVector{T}}
	system_matrix::MT
	initial_conditions::Union{AbstractVector{T}, Function}
	problem_functions::NTuple{M, Function}
	num_vars::Int

	function KSDiscretizedProblem(;
		time_nodes::AbstractVector{T},
		spatial_nodes::NTuple{N, AbstractVector{T}},
		system_matrix::MT,
		initial_conditions::Union{AbstractVector{T}, Function},
		problem_functions::NTuple{M, Function},
		num_vars::Int = 1,
	) where {T <: Number, N, M, MT <: AbstractMatrix}
		if length(time_nodes) == 0
			throw(ArgumentError("Time nodes cannot be empty"))
		end
		if length(spatial_nodes) != N
			throw(ArgumentError("Number of spatial node sets must match N"))
		end
		if !all(isa.(problem_functions, Function))
			throw(ArgumentError("All problem_functions must be callable"))
		end
		if num_vars < 1
			throw(ArgumentError("num_vars must be at least 1, but got $num_vars"))
		end
		return new{T, N, M, MT}(
			time_nodes,
			spatial_nodes,
			system_matrix,
			initial_conditions,
			problem_functions,
			num_vars,
		)
	end
end

struct KSOptimalControlProblem{T <: Number} <: AbstractKSProblem
	state_equations::Vector{<:Function}
	cost_functions::Vector{<:Function}
	terminal_cost::Function
	initial_state::Vector{T}
	t_span::Tuple{T, T}
	control_bounds::Vector{Tuple{T, T}}
	num_vars::Int
	num_controls::Int
	dt::T
	num_time_steps::Int

	function KSOptimalControlProblem(;
		state_equations::Vector{<:Function},
		cost_functions::Vector{<:Function},
		terminal_cost::Function,
		initial_state::Vector{T},
		t_span::Tuple{T, T},
		control_bounds::Vector{Tuple{T, T}},
		dt::T,
	) where {T <: Number}
		num_vars = length(initial_state)
		num_controls = length(control_bounds)
		num_time_steps = round(Int, (t_span[2] - t_span[1]) / dt) + 1

		if length(state_equations) != num_vars
			throw(
				ArgumentError(
					"Number of state equations must match the number of state variables"
				),
			)
		end
		if length(cost_functions) != num_controls
			throw(
				ArgumentError(
					"Number of cost functions must match the number of control variables"
				),
			)
		end

		return new{T}(
			state_equations,
			cost_functions,
			terminal_cost,
			initial_state,
			t_span,
			control_bounds,
			num_vars,
			num_controls,
			dt,
			num_time_steps,
		)
	end
end

# 9. Boundary condition types
struct KSDirichletBC <: AbstractKSBoundaryCondition
	boundary_value::Union{Function, AbstractArray}
	boundary_region::Function
	coordinate_system::AbstractKSCoordinateSystem

	function KSDirichletBC(;
		boundary_value::Union{Function, AbstractArray} = x -> 0.0,
		boundary_region::Function = x -> true,
		coordinate_system::AbstractKSCoordinateSystem = KSCartesianCoordinates(((
			0.0,
			1.0,
		),)),
	)
		return new(boundary_value, boundary_region, coordinate_system)
	end
end

struct KSNeumannBC <: AbstractKSBoundaryCondition
	flux_value::Union{Function, AbstractArray}
	boundary_region::Function
	coordinate_system::AbstractKSCoordinateSystem

	function KSNeumannBC(;
		flux_value::Union{Function, AbstractArray} = x -> 0.0,
		boundary_region::Function = x -> true,
		coordinate_system::AbstractKSCoordinateSystem = KSCartesianCoordinates(((
			0.0, 1.0),)),
	)
		return new(flux_value, boundary_region, coordinate_system)
	end
end

struct KSRobinBC <: AbstractKSBoundaryCondition
	neumann_coefficient::Union{Function, AbstractArray}
	dirichlet_coefficient::Union{Function, AbstractArray}
	boundary_value::Union{Function, AbstractArray}
	boundary_region::Function
	coordinate_system::AbstractKSCoordinateSystem

	function KSRobinBC(;
		neumann_coefficient::Union{Function, AbstractArray} = x -> 1.0,
		dirichlet_coefficient::Union{Function, AbstractArray} = x -> 1.0,
		boundary_value::Union{Function, AbstractArray} = x -> 0.0,
		boundary_region::Function = x -> true,
		coordinate_system::AbstractKSCoordinateSystem = KSCartesianCoordinates(((
			0.0,
			1.0,
		),)),
	)
		return new(
			neumann_coefficient,
			dirichlet_coefficient,
			boundary_value,
			boundary_region,
			coordinate_system,
		)
	end
end

# 10. Solver types
struct KSSolverOptions{T <: Number}
	max_iterations::Int
	tolerance::T
	adaptive::Bool
	max_levels::Int
	smoothness_threshold::T
	initial_elements::Int
	initial_degree::Int
	use_domain_decomposition::Bool
	num_subdomains::Int
	use_strang_splitting::Bool
	dt::T
	num_steps::Int
	legendre_threshold::T

	function KSSolverOptions(
		max_iterations::Int,
		tolerance::T,
		adaptive::Bool,
		max_levels::Int,
		smoothness_threshold::T,
		initial_elements::Int,
		initial_degree::Int,
		use_domain_decomposition::Bool,
		num_subdomains::Int,
		use_strang_splitting::Bool,
		dt::T,
		num_steps::Int,
		legendre_threshold::T,
	) where {T <: Number}
		if max_iterations <= 0
			throw(ArgumentError("Maximum iterations must be positive"))
		end
		if tolerance <= 0
			throw(ArgumentError("Tolerance must be positive"))
		end
		if max_levels <= 0
			throw(ArgumentError("Maximum levels must be positive"))
		end
		if smoothness_threshold <= 0
			throw(ArgumentError("Smoothness threshold must be positive"))
		end
		if initial_elements <= 0
			throw(ArgumentError("Initial elements must be positive"))
		end
		if initial_degree <= 0
			throw(ArgumentError("Initial degree must be positive"))
		end
		if legendre_threshold <= 0
			throw(ArgumentError("Legendre threshold must be positive"))
		end
		return new{T}(
			max_iterations,
			tolerance,
			adaptive,
			max_levels,
			smoothness_threshold,
			initial_elements,
			initial_degree,
			use_domain_decomposition,
			num_subdomains,
			use_strang_splitting,
			dt,
			num_steps,
			legendre_threshold,
		)
	end
end

struct KSDirectSolver <: AbstractKSLinearSolver
	method::Symbol

	function KSDirectSolver(method::Symbol)
		return new(method)
	end
end

struct KSIterativeSolver{T <: Number} <: AbstractKSLinearSolver
	method::Symbol
	maxiter::Int
	tolerance::T
	preconditioner::Union{Nothing, Function}

	function KSIterativeSolver(
		method::Symbol,
		maxiter::Int,
		tolerance::T,
		preconditioner::Union{Nothing, Function} = nothing,
	) where {T <: Number}
		if maxiter <= 0
			throw(ArgumentError("Maximum iterations must be positive"))
		end
		if tolerance <= 0
			throw(ArgumentError("Tolerance must be positive"))
		end
		if !isnothing(preconditioner) && !isa(preconditioner, Function)
			throw(
				ArgumentError(
					"Preconditioner must be a Function or Nothing, but got $(typeof(preconditioner))"
				),
			)
		end
		return new{T}(method, maxiter, tolerance, preconditioner)
	end
end

"""
	TransformCache{T<:Number}

A thread-safe cache for storing and managing transform-related data.

# Fields
- `cells::Dict{Any,Any}`: Storage for transformed cells
- `operator_cache::Dict{Symbol,Function}`: Cache for transform operators
- `matrix_cache::Dict{Symbol,StandardSpectralProperties{T}}`: Cache for spectral matrices
- `_lock::ReentrantLock`: Main lock for thread safety
- `cell_locks::Dict{Any,ReentrantLock}`: Per-cell locks for granular thread safety
- `stats::CacheStats`: Statistics tracking hits, misses, etc.
- `max_size::Int`: Maximum number of items to store
- `cleanup_threshold::Float64`: Threshold ratio triggering cleanup

# Type Parameters
- `T`: Numeric type for matrix calculations (e.g., Float64)
"""
mutable struct TransformCache{T <: Number}
	cells::Dict{Any, Any}
	operator_cache::Dict{Symbol, Function}
	matrix_cache::Dict{Symbol, StandardSpectralProperties{T}}
	_lock::ReentrantLock
	cell_locks::Dict{Any, ReentrantLock}
	stats::CacheStats
	max_size::Int
	cleanup_threshold::Float64

	function TransformCache{T}(;
		max_size::Int = 1000, cleanup_threshold::Float64 = 0.8
	) where {T <: Number}
		# Validate arguments
		max_size > 0 || throw(ArgumentError("max_size must be positive, got $max_size"))
		0 < cleanup_threshold < 1 || throw(
			ArgumentError(
				"cleanup_threshold must be between 0 and 1, got $cleanup_threshold"
			),
		)

		# Initialize with type-stable dictionaries
		cells = Dict{Any, Any}()
		operator_cache = Dict{Symbol, Function}()
		matrix_cache = Dict{Symbol, StandardSpectralProperties{T}}()
		_lock = ReentrantLock()
		cell_locks = Dict{Any, ReentrantLock}()

		# Initialize stats with type-stable custom dict
		stats = CacheStats()
		stats.custom = Dict{Any, Int}()

		# Pre-allocate expected capacity
		sizehint!(cells, max_size)
		sizehint!(cell_locks, max_size)

		return new{T}(
			cells, operator_cache, matrix_cache,
			_lock, cell_locks,
			stats, max_size, cleanup_threshold,
		)
	end
end

# Convenience constructor for Float64
TransformCache(; kwargs...) = TransformCache{Float64}(; kwargs...)

"""
	CacheManager{T,S<:CacheEvictionStrategy}

Enhanced cache manager with flexible storage and eviction.

# Type Parameters
- T: Type of cached items
- S: Type of eviction strategy

# Fields
- capacity::Int: Maximum cache capacity
- items::AbstractDict: Storage for cached items
- metadata::Dict: Item metadata
- strategy::S: Eviction strategy
- on_evict::Union{Function,Nothing}: Optional callback on eviction
- on_insert::Union{Function,Nothing}: Optional callback on insertion
"""
mutable struct CacheManager{T, S <: CacheEvictionStrategy}
	capacity::Int
	items::AbstractDict{Any, T}
	metadata::Dict{Any, CacheMetadata}
	strategy::S
	on_evict::Union{Function, Nothing}
	on_insert::Union{Function, Nothing}
	lock::ReentrantLock

	function CacheManager{T}(
		capacity::Int;
		strategy::Symbol = :lru,
		storage_type::Type{D} = Dict,
		on_evict::Union{Function, Nothing} = nothing,
		on_insert::Union{Function, Nothing} = nothing,
	) where {T, D <: AbstractDict}
		capacity > 0 || throw(ArgumentError("Cache capacity must be positive"))
		eviction_strategy = if strategy == :lru
			LRUEviction()
		elseif strategy == :fifo
			FIFOEviction()
		else
			throw(ArgumentError("Unknown eviction strategy: $strategy"))
		end

		return new{T, typeof(eviction_strategy)}(
			capacity,
			storage_type{Any, T}(),
			Dict{Any, CacheMetadata}(),
			eviction_strategy,
			on_evict,
			on_insert,
			ReentrantLock(),
		)
	end
end

struct KSAMGSolver{T <: Number}
	maxiter::Int
	tolerance::T
	preconditioner::Any

	function KSAMGSolver(maxiter::Int, tolerance::T, preconditioner) where {T <: Number}
		if maxiter <= 0
			throw(ArgumentError("maxiter must be greater than 0"))
		end
		if tolerance <= 0.0
			throw(ArgumentError("tolerance must be greater than 0"))
		end
		return new{T}(maxiter, tolerance, preconditioner)
	end
end

function tuple_if_active(dims...)
	return Tuple(filter(!isnothing, dims))
end

# Handle coordinate ranges
struct CoordinateRanges{T <: Number, N}
	ranges::NTuple{N, Tuple{T, T}}
	active::NTuple{N, Bool}

	function CoordinateRanges(
		ranges::NTuple{N, Tuple{T, T}}, active::NTuple{N, Bool}
	) where {N, T <: Number}
		length(ranges) == length(active) || throw(DimensionMismatch(
			"ranges and active dimensions must match"
		))
		return new{T, N}(ranges, active)
	end
end

# Add iterate methods for KSMesh
function Base.iterate(mesh::KSMesh)
	isempty(mesh.cells) && return nothing
	return (first(mesh.cells), 1)
end

function Base.iterate(mesh::KSMesh, state)
	state >= length(mesh.cells) && return nothing
	return (mesh.cells[state + 1], state + 1)
end

Base.length(mesh::KSMesh) = length(mesh.cells)
Base.eltype(::Type{KSMesh{T, N}}) where {T, N} = KSCell{T, N}

# Add coordinate system compatibility utilities
function compare_domain_ranges(r1::Tuple{T,T}, r2::Tuple{T,T}) where T
    tol = sqrt(eps(T))
    return abs(r1[1] - r2[1]) < tol && abs(r1[2] - r2[2]) < tol
end

function ranges_overlap(r1::Tuple{T,T}, r2::Tuple{T,T}) where T
    tol = 100*sqrt(eps(T))  # Larger tolerance for different systems
    r1_min, r1_max = r1
    r2_min, r2_max = r2
    return !((r1_max + tol < r2_min) || (r2_max + tol < r1_min))
end

end
