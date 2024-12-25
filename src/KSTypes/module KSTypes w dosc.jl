module KSTypes
using StaticArrays, SparseArrays, LinearAlgebra, Base.Threads
import Base: shift!  # Add explicit import

# Exported types
export AbstractKSCell, AbstractStandardKSCell, AbstractKSMesh, AbstractKSProblem
export AbstractKSSolver, AbstractKSOptimizationProblem, AbstractKSLinearSolver
export AbstractKSTransform, AbstractDomainMapping, AbstractLevelSetFunction
export StandardSpectralProperties, AffineTransform, NonlinearTransform, CompositeTransform
export MultiLevelTransform, PrimitiveLevelSet, CompositeLevelSet, DomainMapping, KSCell
export StandardKSCell, TransformCache, KSMesh, KSBasisFunction, KSProblem, KSODEProblem
export KSBVDAEProblem, KSBVPProblem, KSCoupledProblem, KSIDEProblem, KSPDEProblem
export KSDAEProblem, KSMovingBoundaryPDEProblem, KSPIDEProblem, KSDiscretizedProblem
export KSOptimalControlProblem, validate_range, KSCartesianCoordinates, KSPolarCoordinates
export KSCylindricalCoordinates, KSSphericalCoordinates, KSDirichletBC, KSNeumannBC
export KSRobinBC, KSSolverOptions, KSDirectSolver, KSIterativeSolver, KSAMGSolver
export tuple_if_active, detect_physical_cells, is_physical_cell, get_cell_corners
export get_unique_standard_cell_keys, CompositeLevelSet, CoordinateRanges

# Abstract types
@doc raw"""
	AbstractKSCell{T<:Number, N}

An abstract type representing an element in `N`-dimensional space.

# Mathematical Description
An `AbstractKSCell` represents a cell or element in an `N`-dimensional space. It is typically
used in the context of finite element methods or other numerical methods that involve
discretizing a domain into smaller elements.

# Example
For a 2-dimensional space, an `AbstractKSCell` could represent a quadrilateral or triangular
element. For a 3-dimensional space, it could represent a hexahedral or tetrahedral element.
"""
abstract type AbstractKSCell{T, N} end

@doc raw"""
	AbstractStandardKSCell{T<:Number, N}

An abstract type representing a standard element in `N`-dimensional space.

# Mathematical Description
An `AbstractStandardKSCell` represents a standard or reference cell in an `N`-dimensional
space. Standard cells are often used as reference elements in numerical methods, with
mappings defined to transform them to physical elements.

# Example
For a 2-dimensional space, an `AbstractStandardKSCell` could represent a unit square or unit
triangle. For a 3-dimensional space, it could represent a unit cube or unit tetrahedron.
"""
abstract type AbstractStandardKSCell{T, N} end

@doc raw"""
	AbstractKSMesh{T<:Number, N}

An abstract type representing a mesh in `N`-dimensional space.

# Mathematical Description
An `AbstractKSMesh` represents a collection of cells or elements that discretize an
`N`-dimensional domain. Meshes are used in numerical methods to approximate solutions to
partial differential equations and other problems.

# Example
For a 2-dimensional space, an `AbstractKSMesh` could represent a grid of quadrilateral or
triangular elements. For a 3-dimensional space, it could represent a grid of hexahedral or
tetrahedral elements.
"""
abstract type AbstractKSMesh{T, N} end

@doc raw"""
	AbstractKSProblem

An abstract type representing a general problem in the KitchenSink framework.

# Mathematical Description
An `AbstractKSProblem` represents a general problem that can be defined and solved within the
KitchenSink framework. This could include differential equations, optimization problems, and
other types of mathematical problems.

# Example
An `AbstractKSProblem` could represent a partial differential equation problem, an
optimization problem, or a system of algebraic equations.
"""
abstract type AbstractKSProblem end

@doc raw"""
	AbstractKSCoordinateSystem

An abstract type representing a coordinate system.

# Mathematical Description
An `AbstractKSCoordinateSystem` represents a coordinate system used to define the positions
of points in space. This could include Cartesian, polar, cylindrical, spherical, and other
coordinate systems.

# Example
A Cartesian coordinate system in 2D is defined by the coordinates (x, y). A spherical
coordinate system in 3D is defined by the coordinates (r, theta, phi).
"""
abstract type AbstractKSCoordinateSystem end

@doc raw"""
	AbstractKSBoundaryCondition

An abstract type representing a boundary condition.

# Mathematical Description
An `AbstractKSBoundaryCondition` represents a condition that must be satisfied on the
boundary of a domain. Boundary conditions are used in the formulation of differential
equations and other mathematical problems.

# Example
For a partial differential equation, a boundary condition could specify the value of the
solution on the boundary (Dirichlet condition) or the value of the derivative of the solution
on the boundary (Neumann condition).
"""
abstract type AbstractKSBoundaryCondition end

@doc raw"""
	AbstractKSBasisFunction

An abstract type representing a basis function.

# Mathematical Description
An `AbstractKSBasisFunction` represents a basis function used to approximate solutions to
mathematical problems. Basis functions are used in numerical methods such as finite element
methods and spectral methods.

# Example
In a finite element method, basis functions could be piecewise linear functions defined on
each element of the mesh. In a spectral method, basis functions could be global polynomials
or trigonometric functions.
"""
abstract type AbstractKSBasisFunction end

@doc raw"""
	AbstractKSSolver

An abstract type representing a solver.

# Mathematical Description
An `AbstractKSSolver` represents a solver used to find solutions to mathematical problems.
Solvers can be used to solve systems of equations, optimization problems, and other types of
problems.

# Example
A solver for a system of linear equations could use methods such as Gaussian elimination or
iterative methods like conjugate gradient. A solver for an optimization problem could use
methods such as gradient descent or Newton's method.
"""
abstract type AbstractKSSolver end

@doc raw"""
	AbstractKSOptimizationProblem

An abstract type representing an optimization problem.

# Mathematical Description
An `AbstractKSOptimizationProblem` represents an optimization problem that can be defined and
solved within the KitchenSink framework. Optimization problems involve finding the maximum or
minimum of a function subject to constraints.

# Example
An optimization problem could involve finding the minimum of a quadratic function subject to
linear constraints. Another example could be a nonlinear optimization problem with equality
and inequality constraints.
"""
abstract type AbstractKSOptimizationProblem end

@doc raw"""
	AbstractKSLinearSolver

An abstract type representing a linear solver.

# Mathematical Description
An `AbstractKSLinearSolver` represents a solver specifically designed to solve linear systems
of equations. Linear solvers are used in many numerical methods and optimization algorithms.

# Example
A linear solver could use direct methods such as LU decomposition or Cholesky decomposition,
or iterative methods such as conjugate gradient or GMRES.
"""
abstract type AbstractKSLinearSolver end

@doc raw"""
	AbstractKSTransform

An abstract type representing a transformation data.

# Mathematical Description
An `AbstractKSTransform` represents a transformation applied to data or functions.
Transformations can include operations such as Fourier transforms, wavelet transforms, and
other types of mathematical transformations.

# Example
A Fourier transform can be used to transform a function from the time domain to the frequency
domain. A wavelet transform can be used to analyze the local frequency content of a signal.
"""
abstract type AbstractKSTransform end

@doc raw"""
	AbstractDomainMapping

Abstract type for mappings between physical and fictitious domains.

# Mathematical Description
An `AbstractDomainMapping` represents a mapping between physical and fictitious domains. This
is often used in numerical methods to simplify the problem or to apply certain boundary
conditions.

# Example
A mapping could transform a complex physical domain into a simpler reference domain where the
problem is easier to solve. For example, a curved boundary in the physical domain could be
mapped to a straight boundary in the reference domain.
"""
abstract type AbstractDomainMapping end

@doc raw"""
	AbstractLevelSetFunction

Abstract type for level set functions.

# Mathematical Description
An `AbstractLevelSetFunction` represents a level set function used to describe interfaces and
shapes. Level set functions are used in computational geometry, image processing, and
numerical simulations.

# Example
A level set function can represent the interface between two phases in a multiphase flow. The
zero level set of the function defines the interface, and the function can evolve over time
to represent changes in the interface.
"""
abstract type AbstractLevelSetFunction end

# Helper functions

"""
	to_abstract_vector(x::AbstractVector) -> AbstractVector

Converts input `x` to an `AbstractVector`. If `x` is a `StaticArray`, it is converted to a
regular vector.

# Arguments
- `x`: The input that could be an `AbstractVector` or `StaticArray`.

# Returns
- An `AbstractVector`.

# Example
```julia
to_abstract_vector(AbstractVector(1.0, 2.0, 3.0))  # returns [1.0, 2.0, 3.0]
```
"""
to_abstract_vector(x::AbstractVector) = x
to_abstract_vector(x::StaticArray) = collect(x)
to_abstract_vector(x) = [x...]

# AffineTransform
@doc raw"""
	struct AffineTransform{T} <: AbstractKSTransform

Represents an affine transformation in `N`-dimensional space.

# Fields
- `matrix::AbstractMatrix{T}`: The transformation matrix.
- `translation::AbstractVector{T}`: The translation vector.

# Mathematical Description
An affine transformation in \(N\)-dimensional space can be represented as: \[ \mathbf{y} =
\mathbf{A} \mathbf{x} + \mathbf{b} \] where \(\mathbf{A}\) is the transformation matrix,
\(\mathbf{x}\) is the input vector, \(\mathbf{b}\) is the translation vector, and
\(\mathbf{y}\) is the transformed vector.

# Constructor
	AffineTransform(matrix::AbstractMatrix{T}, translation::AbstractVector{T}) where {T <: Number}

Initializes the affine transformation with the matrix and translation vector.

# Example
```julia
A = AffineTransform([1.0 0.0; 0.0 1.0], [0.0, 1.0])
```
Index Explanation matrix: The transformation matrix (\mathbf{A}).

translation: The translation vector (\mathbf{b}).
"""
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
# NonlinearTransform
@doc raw"""
	struct NonlinearTransform <: AbstractKSTransform

Represents a nonlinear transformation in `N`-dimensional space.

# Fields
- `forward_map::Function`: A function mapping from fictitious to physical domain.
- `jacobian::Function`: A function computing the Jacobian of the transformation.
- `inverse_map::Union{Nothing, Function}`: A function for the inverse transformation
  (optional).

# Mathematical Description
A nonlinear transformation \( T \) maps a point \( x \) in the fictitious domain to a point
\( y \) in the physical domain: \[ y = T(x) \] The Jacobian \( J \) of the transformation is
a matrix of partial derivatives: \[ J_{ij} = \frac{\partial T_i}{\partial x_j} \] If the
inverse transformation \( T^{-1} \) exists, it maps a point \( y \) in the physical domain
back to a point \( x \) in the fictitious domain: \[ x = T^{-1}(y) \]

# Constructor
	NonlinearTransform(; forward_map::Function, jacobian::Function, inverse_map::Union{Nothing, Function}=nothing)

Initializes the nonlinear transformation.

# Example
```julia
nonlinear = NonlinearTransform(forward_map = x -> x^2, jacobian = x -> 2x)
```
Index Explanation forward_map: The function representing the forward transformation.

jacobian: The function computing the Jacobian of the transformation.

inverse_map: The function representing the inverse transformation, if it exists.
"""
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

# CompositeTrans]form
@doc raw"""
	struct CompositeTransform{T, N} <: AbstractKSTransform

Represents a composite transformation consisting of multiple sub-transformations.

# Fields
- `transforms::Vector{AbstractKSTransform}`: A vector of transformations.

# Mathematical Description
A composite transformation \( T \) is defined as a sequence of sub-transformations \( \{T_i\}
\). The composite transformation is applied by sequentially applying each sub-transformation
in the order they appear in the vector. Mathematically, if \( \mathbf{x} \) is the input
vector, the composite transformation is given by: \[ T(\mathbf{x}) = T_n(T_{n-1}(\ldots
T_1(\mathbf{x}) \ldots)) \] where \( T_i \) is the \( i \)-th sub-transformation.

# Example
```julia
T1 = AffineTransform(AbstractMatrix{2, 2}(1.0, 0.0, 0.0, 1.0), AbstractVector{2}(0.0, 1.0))
T2 = NonlinearTransform(x -> x^2, x -> 2x)
composite = CompositeTransform([T1, T2])
```
"""
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
@doc raw"""
	struct MultiLevelTransform{T, N}

Represents a transformation that applies different transformations across levels.

# Fields
- `base_transform::AbstractKSTransform`: The base transformation.
- `level_transforms::Vector{AbstractKSTransform}`: The transformations applied at different
  levels.

# Mathematical Description
A `MultiLevelTransform` applies a series of transformations to an input. The base
transformation \( T_0 \) is applied first, followed by a sequence of level-specific
transformations \( T_i \) for \( i = 1, 2, \ldots, N \).

Given an input \( x \), the transformation can be described as: \[ y = T_N(T_{N-1}(\ldots
T_1(T_0(x)) \ldots)) \] where \( T_0 \) is the base transformation and \( T_i \) are the
level-specific transformations.

# Constructor
	MultiLevelTransform(base_transform::AbstractKSTransform, level_transforms::Vector{AbstractKSTransform})

Initializes the multi-level transformation with a base transformation and a vector of
level-specific transformations.

# Example
```julia
T1 = AffineTransform(AbstractMatrix{2, 2}(1.0, 0.0, 0.0, 1.0), AbstractVector{2}(0.0, 1.0))
T2 = NonlinearTransform(x -> x^2, x -> 2x)
multi_level = MultiLevelTransform(T1, [T2])
```
In this example, T1 is an affine transformation and T2 is a nonlinear transformation. The
MultiLevelTransform applies T1 first, followed by T2. """
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
@doc raw"""
	struct PrimitiveLevelSet{T} <: AbstractLevelSetFunction

Represents a primitive level set function.

# Fields
- `func::Function`: The level set function.
- `gradient::Union{Nothing, Function}`: The gradient of the level set function (optional).

# Mathematical Description
A level set function \( \phi(x) \) is a scalar function whose zero level set defines a
surface or curve. The gradient of the level set function \( \nabla \phi(x) \) is a vector
function that points in the direction of the steepest ascent of \( \phi \).

# Example
For a level set function \( \phi(x) = x^2 \), the gradient is \( \nabla \phi(x) = 2x \).

# Constructor
	PrimitiveLevelSet(; func::Function, gradient::Union{Nothing, Function}=nothing)

Initializes the primitive level set function.

# Index Explanation
- `func`: The function representing the level set.
- `gradient`: The gradient of the level set function, which can be a function or `nothing`.

# Example Usage
```julia
level_set = PrimitiveLevelSet(func = x -> x^2, gradient = x -> 2x)
```
"""
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
@doc raw"""
	struct CompositeLevelSet <: AbstractLevelSetFunction

Represents a composite level set function that combines multiple level set functions using
operations.

# Fields
- `operations::Vector{Function}`: A vector of operations combining the level set functions.
- `primitives::Vector{AbstractLevelSetFunction}`: A vector of primitive level set functions.

# Mathematical Description
A composite level set function \( \phi \) is defined by combining multiple primitive level
set functions \( \phi_i \) using a set of operations \( \mathcal{O}_j \). Mathematically,
this can be represented as: \[ \phi = \mathcal{O}_{n-1}(\mathcal{O}_{n-2}(\ldots
\mathcal{O}_1(\phi_1, \phi_2), \phi_3), \ldots, \phi_n) \] where \( \phi_i \) are the
primitive level set functions and \( \mathcal{O}_j \) are the operations.

In this example, the composite level set function is defined as: [ \phi(x) = \phi_1(x) +
\phi_2(x) = x^2 + \sin(x) ]
# Constructor
- `CompositeLevelSet(operations::Vector{Function},
  primitives::Vector{AbstractLevelSetFunction})`: Initializes the composite level set.

# Example
```julia
primitive1 = PrimitiveLevelSet(x -> x^2)
primitive2 = PrimitiveLevelSet(x -> sin(x))
operations = [(x, y) -> x + y]
composite = CompositeLevelSet(operations, [primitive1, primitive2])
```
"""
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

@doc raw"""
	struct DomainMapping <: AbstractDomainMapping

Concrete type representing a mapping between physical and fictitious domains.

# Fields
- `forward_transform::AbstractKSTransform`: Transform from physical to fictitious domain.
- `inverse_transform::AbstractKSTransform`: Transform from fictitious to physical domain.
- `physical_system::AbstractKSCoordinateSystem`: Coordinate system of the physical domain.
- `fictitious_system::AbstractKSCoordinateSystem`: Coordinate system of the fictitious
  domain.

# Mathematical Description
A `DomainMapping` represents a bijective mapping between a physical domain and a fictitious
domain. The forward transform \( F \) maps points from the physical domain to the fictitious
domain: \[ \mathbf{x}_{\text{fict}} = F(\mathbf{x}_{\text{phys}}) \] The inverse transform \(
F^{-1} \) maps points from the fictitious domain back to the physical domain: \[
\mathbf{x}_{\text{phys}} = F^{-1}(\mathbf{x}_{\text{fict}}) \]

# Constructor
	DomainMapping(; forward_transform::AbstractKSTransform,
				   inverse_transform::AbstractKSTransform,
				   fictitious_system::AbstractKSCoordinateSystem,
				   physical_system::AbstractKSCoordinateSystem)

Constructs a `DomainMapping` with the given parameters.

# Index Explanation
- `forward_transform`: The transform from the physical domain to the fictitious domain.
- `inverse_transform`: The transform from the fictitious domain to the physical domain.
- `physical_system`: The coordinate system of the physical domain.
- `fictitious_system`: The coordinate system of the fictitious domain.
"""

struct DomainMapping <: AbstractDomainMapping
	forward_transform::AbstractKSTransform
	inverse_transform::AbstractKSTransform
	fictitious_system::AbstractKSCoordinateSystem
	physical_system::AbstractKSCoordinateSystem

	# Inner constructor with validation
	function DomainMapping(;
		forward_transform::AbstractKSTransform,
		inverse_transform::AbstractKSTransform,
		fictitious_system::AbstractKSCoordinateSystem,
		physical_system::AbstractKSCoordinateSystem,
	)
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

# In KSTypes.jl
# Add a positional constructor
function DomainMapping(
	forward_transform::AbstractKSTransform,
	inverse_transform::AbstractKSTransform,
	fictitious_system::AbstractKSCoordinateSystem,
	physical_system::AbstractKSCoordinateSystem)
	return DomainMapping(;
		forward_transform = forward_transform,
		inverse_transform = inverse_transform,
		fictitious_system = fictitious_system,
		physical_system = physical_system,
	)
end
@doc raw"""
	struct StandardSpectralProperties{T <: Number}

Represents spectral properties such as nodes, weights, differentiation matrices, and
higher-order differentiation matrices for a given polynomial order and continuity.

# Fields
- `p::Int`: Polynomial order.
- `continuity_order::Int`: Order of continuity.
- `nodes_with_boundary::AbstractVector{T}`: Nodes including boundary points.
- `nodes_interior::AbstractVector{T}`: Interior nodes excluding boundary points.
- `weights_with_boundary::AbstractVector{T}`: Weights including boundary points.
- `weights_interior::AbstractVector{T}`: Weights for interior nodes.
- `differentiation_matrix_with_boundary::AbstractMatrix{T}`: First-order differentiation
  matrix including boundary nodes.
- `differentiation_matrix_interior::AbstractMatrix{T}`: First-order differentiation matrix
  for interior nodes.
- `quadrature_matrix::AbstractMatrix{T}`: Quadrature matrix for numerical integration.
- `higher_order_diff_matrices_with_boundary::Vector{AbstractMatrix{T}}`: Higher-order
  differentiation matrices including boundary nodes.
- `higher_order_diff_matrices_interior::Vector{AbstractMatrix{T}}`: Higher-order
  differentiation matrices excluding boundary nodes.

# Mathematical Description
Given a set of interpolation nodes, \( x_i \), and weights, \( w_i \), the differentiation
matrix, \( D \), is defined such that:

\[ (D f)_i = \sum_{j=1}^{n} D_{ij} f_j \]

where \( f_j \) are the values of a function at the nodes \( x_j \).

The quadrature matrix \( Q \) is used for numerical integration and is defined such that:

\[ \int_a^b f(x) \, dx \approx \sum_{i=1}^{n} Q_{ij} f_j \]

Higher-order differentiation matrices \( D^{(k)} \) for \( k \geq 2 \) are defined similarly
to the first-order differentiation matrix.

# Example Usage
```julia
spectral_props = StandardSpectralProperties(
	p = 1,
	continuity_order = 0,
	nodes_with_boundary = [-1.0, 0.0, 1.0],
	nodes_interior = [0.0],
	weights_with_boundary = [2.0, 2.0, 2.0],
	weights_interior = [2.0],
	differentiation_matrix_with_boundary = Diagonal([2.0, -2.0, 2.0]),
	differentiation_matrix_interior = Diagonal([2.0]),
	quadrature_matrix = Diagonal([2.0]),
	higher_order_diff_matrices_with_boundary = [Diagonal([2.0, -2.0, 2.0])],
	higher_order_diff_matrices_interior = [Diagonal([2.0])]
)

spectral_props_2 = StandardSpectralProperties(
	p = 4,
	continuity_order = 1,
	nodes_with_boundary = [-1.0, -0.5, 0.5, 1.0],
	nodes_interior = [-0.5, 0.5],
	weights_with_boundary = [1.0, 2.0, 2.0, 1.0],
	weights_interior = [2.0, 2.0],
	differentiation_matrix_with_boundary = Diagonal([1.0, 2.0, -2.0, 1.0]),
	differentiation_matrix_interior = Diagonal([2.0, 2.0]),
	quadrature_matrix = Diagonal([1.0, 1.0, 1.0, 1.0]),
	higher_order_diff_matrices_with_boundary = [Diagonal([1.0, 2.0, -2.0, 1.0])],
	higher_order_diff_matrices_interior = [Diagonal([2.0, 2.0])]
)

	````

# See also
- @ref StandardKSCell
"""

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

"""
	TransformCache{T <: Number}

A cache to store transformed operators and spectral properties to avoid redundant
computations.

# Fields
- `operator_cache::Dict{Symbol, Function}`: Caches transformed operators.
- `matrix_cache::Dict{Symbol, StandardSpectralProperties{T}}`: Caches transformed spectral
  properties.
"""
mutable struct TransformCache{T <: Number}
	operator_cache::Dict{Symbol, Function}
	matrix_cache::Dict{Symbol, StandardSpectralProperties{T}}
	lock::ReentrantLock

	function TransformCache{T}() where {T <: Number}
		return new{T}(
			Dict{Symbol, Function}(),
			Dict{Symbol, StandardSpectralProperties{T}}(),
			ReentrantLock(),
		)
	end
end
@doc raw"""
	struct StandardKSCell{T <: Number, N} <: AbstractStandardKSCell{T, N}

Represents precomputed, scaled data for a cell in the Finite Cell Method (FCM). This
structure is intended to be reused across cells with the same polynomial order combination
and level in multiple dimensions.

# Fields
- `p::NTuple{N, Int}`: Polynomial orders in each dimension.
- `level::Int`: The refinement level of the cell.
- `continuity_order::NTuple{N, Int}`: The order of continuity.
- `nodes_with_boundary::NTuple{N, AbstractVector{T}}`: Nodes with boundary points.
- `nodes_interior::NTuple{N, AbstractVector{T}}`: Interior nodes excluding boundary points.
- `weights_with_boundary::NTuple{N, AbstractVector{T}}`: Weights with boundary points.
- `weights_interior::NTuple{N, AbstractVector{T}}`: Weights for interior nodes.
- `differentiation_matrix_with_boundary::NTuple{N, AbstractMatrix{T}}`: First-order
  differentiation matrix with boundary nodes.
- `differentiation_matrix_interior::NTuple{N, AbstractMatrix{T}}`: First-order
  differentiation matrix for interior nodes.
- `quadrature_matrix::NTuple{N, AbstractMatrix{T}}`: Quadrature matrix for integration.
- `higher_order_diff_matrices_with_boundary::NTuple{N, Vector{AbstractMatrix{T}}}`:
  Higher-order differentiation matrices with boundary nodes.
- `higher_order_diff_matrices_interior::NTuple{N, Vector{AbstractMatrix{T}}}`: Higher-order
  differentiation matrices without boundary nodes.

# Mathematical Description
The polynomial degree \( p_k \) in each dimension determines the approximation order within
the cell. The nodes and weights are used for numerical integration and differentiation. The
differentiation matrices \( D_k \) and quadrature matrices \( Q_k \) are used to compute
derivatives and integrals, respectively. Higher-order differentiation matrices \( D_k^{(m)}
\) are used for computing higher-order derivatives.

# Example Usage
```julia
# Define StandardSpectralProperties for a 1D case
spectral_props = StandardSpectralProperties(
	p = 3,
	nodes_with_boundary = [-1.0, 0.0, 1.0],
	nodes_interior = [0.0],
	weights_with_boundary = [2.0, 2.0, 2.0],
	weights_interior = [2.0],
	differentiation_matrix_with_boundary = Diagonal([2.0, -2.0, 2.0]),
	differentiation_matrix_interior = Diagonal([2.0]),
	quadrature_matrix = Diagonal([2.0]))

# Create a StandardKSCell for a 1D case
cell = StandardKSCell(
	p = (3,),
	level = 1,
	continuity_order = 0,
	nodes_with_boundary = ([-1.0, 0.0, 1.0],),
	nodes_interior = ([0.0],),
	weights_with_boundary = ([2.0, 2.0, 2.0],),
	weights_interior = ([2.0],),
	differentiation_matrix_with_boundary = (Diagonal([2.0, -2.0, 2.0]),),
	differentiation_matrix_interior = (Diagonal([2.0]),),
	quadrature_matrix = (Diagonal([2.0]),),
	higher_order_diff_matrices_with_boundary = (([Diagonal([2.0, -2.0, 2.0])],),),
	higher_order_diff_matrices_interior = (([Diagonal([2.0])],),))

# Define StandardSpectralProperties for a 2D case
spectral_props_1 = StandardSpectralProperties(
	p = 3,
	nodes_with_boundary = [-1.0, 0.0, 1.0],
	nodes_interior = [0.0],
	weights_with_boundary = [2.0, 2.0, 2.0],
	weights_interior = [2.0],
	differentiation_matrix_with_boundary = Diagonal([2.0, -2.0, 2.0]),
	differentiation_matrix_interior = Diagonal([2.0]),
	quadrature_matrix = Diagonal([2.0]))

spectral_props_2 = StandardSpectralProperties(
	p = 4,
	nodes_with_boundary = [-1.0, -0.5, 0.5, 1.0],
	nodes_interior = [-0.5, 0.5],
	weights_with_boundary = [1.0, 2.0, 2.0, 1.0],
	weights_interior = [2.0, 2.0],
	differentiation_matrix_with_boundary = Diagonal([1.0, 2.0, -2.0, 1.0]),
	differentiation_matrix_interior = Diagonal([2.0, 2.0]),
	quadrature_matrix = Diagonal([1.0, 1.0, 1.0, 1.0]))

# Create a StandardKSCell for a 2D case
cell_2D = StandardKSCell(
	p = (3, 4),
	level = 1,
	continuity_order = 0,
	nodes_with_boundary = ([-1.0, 0.0, 1.0], [-1.0, -0.5, 0.5, 1.0]),
	nodes_interior = ([0.0], [-0.5, 0.5]),
	weights_with_boundary = ([2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 1.0]),
	weights_interior = ([2.0], [2.0, 2.0]),
	differentiation_matrix_with_boundary = (Diagonal([2.0, -2.0, 2.0]), Diagonal([1.0, 2.0, -2.0, 1.0])),
	differentiation_matrix_interior = (Diagonal([2.0]), Diagonal([2.0, 2.0])),
	quadrature_matrix = (Diagonal([2.0]), Diagonal([1.0, 1.0, 1.0, 1.0])),
	higher_order_diff_matrices_with_boundary = (([Diagonal([2.0, -2.0, 2.0])],), ([Diagonal([1.0, 2.0, -2.0, 1.0])],)),
	higher_order_diff_matrices_interior = (([Diagonal([2.0])],), ([Diagonal([2.0, 2.0])],)))
# See Also
- @ref StandardSpectralProperties

"""
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
			Union{Nothing, AbstractVector{<:AbstractMatrix}}, N
		} = ntuple(_ -> nothing, N),
		higher_order_diff_matrices_interior::NTuple{
			Union{Nothing, AbstractVector{<:AbstractMatrix}}, N
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
					"Length of nodes_with_boundary[\$i] ($(length(nodes_with_boundary[i]))) must be p[\$i] + 2 ($(p[i] + 2))"
				),
			)
			length(nodes_interior[i]) == p[i] || throw(
				DimensionMismatch(
					"Length of nodes_interior[\$i] ($(length(nodes_interior[i]))) must be p[\$i] ($(p[i]))"
				),
			)
			length(weights_with_boundary[i]) == p[i] + 2 || throw(
				DimensionMismatch(
					"Length of weights_with_boundary[\$i] ($(length(weights_with_boundary[i]))) must be p[\$i] + 2 ($(p[i] + 2))"
				),
			)
			length(weights_interior[i]) == p[i] || throw(
				DimensionMismatch(
					"Length of weights_interior[\$i] ($(length(weights_interior[i]))) must be p[\$i] ($(p[i]))"
				),
			)
			size(differentiation_matrix_with_boundary[i]) == (p[i] + 2, p[i] + 2) || throw(
				DimensionMismatch(
					"Size of differentiation_matrix_with_boundary[\$i] ($(size(differentiation_matrix_with_boundary[i]))) must be (p[\$i] + 2, p[\$i] + 2)"
				),
			)
			size(differentiation_matrix_interior[i]) == (p[i], p[i]) || throw(
				DimensionMismatch(
					"Size of differentiation_matrix_interior[\$i] (\$(size(differentiation_matrix_interior[\$i]))) must be (p[\$i], p[\$i])"
				),
			)
			size(quadrature_matrix[i]) == (p[i], p[i]) || throw(
				DimensionMismatch(
					"Size of quadrature_matrix[\$i] (\$(size(quadrature_matrix[\$i]))) must be (p[\$i], p[\$i])"
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
@doc raw"""
	mutable struct KSCell{T <: Number, N}

Represents a cell in the Finite Cell Method (FCM). The cell is associated with a specific
level of refinement and polynomial degree.

# Fields
- `id::Int`: Unique identifier for the cell.
- `p::NTuple{N, Int}`: Polynomial degree in each dimension.
- `level::Int`: Refinement level of the cell.
- `continuity_order::NTuple{N, Int}`: The order of continuity \(C^k\), where \(k\) is the
  number of continuous derivatives.
- `standard_cell_key::Tuple{NTuple{N, Int}, Int}`: Key to look up standard cells in the
  manager, based on polynomial degree and level.
- `neighbors::Dict{Symbol, Int}`: Neighbors in each dimension, mapped by directional symbols
  (e.g., `:left`, `:right`).
- `node_map::Dict{NTuple{N, Int}, Int}`: Mapping from local node indices to global node
  indices.
- `tensor_product_mask::NTuple{N, AbstractArray{Bool}}`: Mask indicating active regions for
  tensor products in each dimension.
- `boundary_connectivity::Dict{Symbol, Int}`: Boundary connectivity information, indicating
  adjacent cells.
- `error_estimate::T`: Error estimate associated with this cell.
- `legendre_decay_rate::T`: Legendre decay rate for this cell.
- `is_leaf::Bool`: Flag indicating if this is a leaf cell in the hierarchical structure.
- `is_fictitious::Bool`: Flag for fictitious cells used at boundaries.
- `refinement_options::Union{Nothing, Dict{Symbol, Any}}`: Options for adaptive refinement
  decisions.
- `parent_id::Union{Nothing, Int}`: Parent cell ID for hierarchical refinement.
- `child_ids::Union{Nothing, Vector{Int}}`: Child cell IDs for hierarchical refinement.
- `b_spline_coefficients::Union{Nothing, NTuple{N, AbstractVector{T}}}`: B-spline
  coefficients for each dimension.
- `b_spline_knots::Union{Nothing, NTuple{N, AbstractVector{T}}}`: Knot vectors for the
  B-splines in each dimension.

# Mathematical Description
The polynomial degree \(p_k\) in each dimension determines the approximation order within the
cell. The tensor product mask \(M_k\) identifies the active regions within the cell for each
dimension \(k\). The error estimate \(e\) and the Legendre decay rate \(\lambda\) are used in
adaptive refinement strategies.

# Constructor
	KSCell(id::Int, p::NTuple{N, Int}, level::Int, continuity_order::NTuple{N, Int},
		   standard_cell_key::Tuple{NTuple{N, Int}, Int}, neighbors::Dict{Symbol, Int},
		   node_map::Dict{NTuple{N, Int}, Int}, tensor_product_mask::NTuple{N, AbstractArray{Bool}},
		   boundary_connectivity::Dict{Symbol, Int}, error_estimate::T, legendre_decay_rate::T,
		   is_leaf::Bool, is_fictitious::Bool, refinement_options::Union{Nothing, Dict{Symbol, Any}},
		   parent_id::Union{Nothing, Int}, child_ids::Union{Nothing, Vector{Int}},
		   b_spline_coefficients::Union{Nothing, NTuple{N, AbstractVector{T}}} = nothing,
		   b_spline_knots::Union{Nothing, NTuple{N, AbstractVector{T}}} = nothing) where {T <: Number, N}

Constructs a KSCell with the given parameters.

# Index Explanation
- `id`: Unique identifier for the cell.
- `p`: Polynomial degree in each dimension.
- `level`: Refinement level of the cell.
- `continuity_order`: The order of continuity \(C^k\), where \(k\) is the number of
  continuous derivatives.
- `standard_cell_key`: Key to look up standard cells in the manager, based on polynomial
  degree and level.
- `neighbors`: Neighbors in each dimension, mapped by directional symbols (e.g., `:left`,
  `:right`).
- `node_map`: Mapping from local node indices to global node indices.
- `tensor_product_mask`: Mask indicating active regions for tensor products in each
  dimension.
- `boundary_connectivity`: Boundary connectivity information, indicating adjacent cells.
- `error_estimate`: Error estimate associated with this cell.
- `legendre_decay_rate`: Legendre decay rate for this cell.
- `is_leaf`: Flag indicating if this is a leaf cell in the hierarchical structure.
- `is_fictitious`: Flag for fictitious cells used at boundaries.
- `refinement_options`: Options for adaptive refinement decisions.
- `parent_id`: Parent cell ID for hierarchical refinement.
- `child_ids`: Child cell IDs for hierarchical refinement.
- `b_spline_coefficients`: B-spline coefficients for each dimension.
- `b_spline_knots`: Knot vectors for the B-splines in each dimension.
"""

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
	nodes_with_boundary::NTuple{N, Vector{T}}  # Add this field

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
		nodes_with_boundary::NTuple{N, Vector{T}} = ntuple(d -> Vector{T}(), N),  # Default value for nodes_with_boundary
	) where {T <: Number, N}

		# Validation checks
		id > 0 || throw(ArgumentError("Cell ID must be positive"))
		level >= 0 || throw(ArgumentError("Level must be non-negative"))
		length(p) == N ||
			throw(ArgumentError("Length of polynomial degree tuple must match N"))
		length(continuity_order) == N ||
			throw(ArgumentError("Length of continuity order tuple must match N"))

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
			nodes_with_boundary,
		)
	end
end

# Outer constructor for flexible initialization
function KSCell{T}(;
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
	nodes_with_boundary::NTuple{N, Vector{T}} = ntuple(d -> Vector{T}(), N),  # Default value for nodes_with_boundary
) where {T, N}

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
		nodes_with_boundary,
	)
end
@doc raw"""
	mutable struct KSMesh{T <: Number, N} <: AbstractKSMesh{T, N}

A mutable struct representing a mesh in the KS framework.

# Fields
- `cells::AbstractVector{KSCell{T, N}}`: A vector of KSCell objects.
- `tensor_product_masks::Vector{NTuple{N, AbstractVector{Bool}}}`: Tensor product masks for
  the cells.
- `node_map::Dict{Tuple{Int, NTuple{N, Int}}, Int}`: A mapping from local to global node
  indices.
- `standard_cell_keys::Set{Tuple{NTuple{N, Int}, Int}}`: A set of keys for standard cells.
- `global_error_estimate::T`: A global error estimate.
- `physical_cells::Vector{Int}`: A vector of physical cell indices.
- `boundary_cells::Dict{Symbol, Vector{Int}}`: A dictionary of boundary cells.
- `physical_domain::Function`: A function defining the physical domain.
- `characteristic_function::Union{Nothing, Function}`: A characteristic function.
- `continuity_constraints::Union{Nothing, Dict{Symbol, AbstractMatrix{T}}}`: Continuity
  constraints.
- `transformation_data::Union{Nothing, AbstractKSTransform}`: Transformation data.

# Mathematical Description
The global error estimate \( \epsilon \) for the mesh is typically the sum or maximum of the
local error estimates \( \epsilon_i \) for each cell:

\[ \epsilon = \sum_{i=1}^{n} \epsilon_i \]

or

\[ \epsilon = \max_{i=1}^{n} \epsilon_i \]

The tensor product masks and location matrices are crucial in assembling global system
matrices from local cell matrices.

# Inner Constructor
	KSMesh{T, N}(cells::AbstractVector{KSCell{T, N}}, tensor_product_masks::Vector{NTuple{N, AbstractVector{Bool}}},
				 node_map::Dict{Tuple{Int, NTuple{N, Int}}, Int}, standard_cell_keys::Set{Tuple{NTuple{N, Int}, Int}},
				 global_error_estimate::T, physical_cells::Vector{Int}, boundary_cells::Dict{Symbol, Vector{Int}},
				 physical_domain::Function, characteristic_function::Union{Nothing, Function},
				 continuity_constraints::Union{Nothing, Dict{Symbol, AbstractMatrix{T}}},
				 transformation_data::Union{Nothing, AbstractKSTransform}) where {T <: Number, N}

Constructs a `KSMesh` object with the given parameters. Validates the input data to ensure
consistency.

# Arguments
- `cells::AbstractVector{KSCell{T, N}}`: A vector of KSCell objects.
- `tensor_product_masks::Vector{NTuple{N, AbstractVector{Bool}}}`: Tensor product masks for
  the cells.
- `node_map::Dict{Tuple{Int, NTuple{N, Int}}, Int}`: A mapping from local to global node
  indices.
- `standard_cell_keys::Set{Tuple{NTuple{N, Int}, Int}}`: A set of keys for standard cells.
- `global_error_estimate::T`: A global error estimate.
- `physical_cells::Vector{Int}`: A vector of physical cell indices.
- `boundary_cells::Dict{Symbol, Vector{Int}}`: A dictionary of boundary cells.
- `physical_domain::Function`: A function defining the physical domain.
- `characteristic_function::Union{Nothing, Function}`: A characteristic function.
- `continuity_constraints::Union{Nothing, Dict{Symbol, AbstractMatrix{T}}}`: Continuity
  constraints.
- `transformation_data::Union{Nothing, AbstractKSTransform}`: Transformation data.

# Throws
- `ArgumentError` if `cells` or `physical_cells` are empty, or if there are duplicate cell
  IDs or invalid cell IDs in boundary cells.

# Outer Constructor
	KSMesh{T, N}(cells::AbstractVector{KSCell{T, N}}, global_error_estimate::Real = 0.0,
				 boundary_cells::Dict{Symbol, Vector{Int}} = Dict{Symbol, Vector{Int}}(),
				 characteristic_function::Union{Nothing, Function} = nothing,
				 continuity_constraints::Union{Nothing, Dict{Symbol, AbstractMatrix{T}}} = nothing,
				 transformation_data::Union{Nothing, AbstractKSTransform} = nothing,
				 physical_domain::Function) where {T <: Number, N}

Constructs a `KSMesh` object with the given parameters. Automatically constructs the node
map, tensor product masks, standard cells, and detects physical cells within the provided
physical domain.

# Arguments
- `cells::AbstractVector{KSCell{T, N}}`: A vector of KSCell objects.
- `global_error_estimate::Real`: A global error estimate (default is 0.0).
- `boundary_cells::Dict{Symbol, Vector{Int}}`: A dictionary of boundary cells (default is an
  empty dictionary).
- `characteristic_function::Union{Nothing, Function}`: A characteristic function (default is
  `nothing`).
- `continuity_constraints::Union{Nothing, Dict{Symbol, AbstractMatrix{T}}}`: Continuity
  constraints (default is `nothing`).
- `transformation_data::Union{Nothing, AbstractKSTransform}`: Transformation data (default is
  `nothing`).
- `physical_domain::Function`: A function defining the physical domain.

# Returns
- `KSMesh{T, N}`: A new `KSMesh` object.

# See also
- @ref KSCell
- @ref StandardKSCell
"""

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
"""
	get_unique_standard_cell_keys(mesh::KSMesh)

Return the unique standard cell keys from the given mesh.

# Arguments
- `mesh::KSMesh`: The mesh object containing standard cell keys.

# Returns
- A collection of unique standard cell keys.
"""
function get_unique_standard_cell_keys(mesh::KSMesh)
	return mesh.standard_cell_keys
end

"""
	detect_physical_cells(cells::Vector{KSCell{T, N}}, physical_domain::Function)::Vector{Int} where {T <: Number, N}

Detect and return the indices of physical cells within the given vector of cells.

# Arguments
- `cells::Vector{KSCell{T, N}}`: A vector of cells to be checked.
- `physical_domain::Function`: A function that defines the physical domain.

# Returns
- A vector of integers representing the indices of physical cells.
"""
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

@doc raw"""
	is_physical_cell(cell::KSCell{T, N}, physical_domain::Function)::Bool where {T <: Number, N}

Determine if a given cell is within the physical domain.

# Arguments
- `cell::KSCell{T, N}`: The cell to be checked.
- `physical_domain::Function`: A function that defines the physical domain.

# Returns
- A boolean indicating whether the cell is within the physical domain.

# Mathematical Description
Given a cell \( c \) and a physical domain function \( D \), the function checks if any
corner of the cell lies within the physical domain. Mathematically, this can be described as:

\[ \text{is\_physical\_cell}(c, D) = \exists x \in \text{corners}(c) \text{ such that } D(x)
= \text{true} \]

where \( \text{corners}(c) \) represents the set of corner points of the cell \( c \).

# Logical Operators
- `any`: Checks if any element in the collection satisfies the given condition.
- `.`: Element-wise application of the function over the collection.

# Example
```julia
cell = KSCell(...)
physical_domain = x -> x[1] >= 0 && x[2] >= 0  # Example physical domain function
is_physical_cell(cell, physical_domain)
"""
function is_physical_cell(
	cell::KSCell{T, N},
	physical_domain::Function,
)::Bool where {T <: Number, N}
	corners = get_cell_corners(cell)
	result = any(physical_domain.(corners))
	return result
end

@doc raw"""
	get_cell_corners(cell::KSCell{T, N}) where {T <: Number, N}

Get the corners of a given cell, assuming it is defined on a unit hypercube \([0, 1]^N\).

# Arguments
- `cell::KSCell{T, N}`: The cell for which to get the corners.

# Returns
- A vector of tuples representing the corners of the cell.

# Mathematical Description
Given a cell defined on a unit hypercube \([0, 1]^N\), the corners of the cell are the
vertices of the hypercube. For an \(N\)-dimensional hypercube, there are \(2^N\) corners.
Each corner can be represented as an \(N\)-tuple where each element is either 0 or 1.

# Example
For a 2-dimensional cell, the corners are: \[ (0, 0), (0, 1), (1, 0), (1, 1) \]

For a 3-dimensional cell, the corners are: \[ (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1,
0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1) \]
"""
function get_cell_corners(cell::KSCell{T, N}) where {T <: Number, N}
	# Assuming the cell is defined on a unit hypercube [0, 1]^N
	corners = Vector{NTuple{N, T}}()
	for corner_indices in Iterators.product(ntuple(i -> [0, 1], N)...)
		push!(corners, corner_indices)
	end
	return corners
end

@doc raw"""
	KSBasisFunction

Represents a basis function in the system.

# Fields
- `id::Int`: The unique identifier for the basis function.
- `function_handle::Function`: The function associated with the basis function.
- `contribution::Number`: The contribution of this basis function to the system.
- `is_removable::Bool`: A flag indicating whether this basis function is removable.

# Mathematical Description
A basis function \( \phi_i(x) \) is a function that is part of a set of functions used to
represent other functions in the system. The contribution of the basis function to the system
can be represented as a scalar value \( c_i \).

# Example
For a basis function \( \phi_i(x) \) with an identifier \( i \), the contribution to the
system can be represented as: \[ c_i \cdot \phi_i(x) \] where \( c_i \) is the contribution
of the basis function.
"""
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
@doc raw"""
	struct KSProblem{T <: Number, N} <: AbstractKSProblem

Represents a problem in the KS system.

# Fields
- `equation::Function`: The equation defining the problem.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{Nothing, AbstractVector{T}, Function}`: The initial conditions
  for the problem.
- `tspan::Union{Nothing, Tuple{T, T}}`: The time span for the problem.
- `continuity_order::Int`: The order of continuity required for the solution.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KS problem is defined by a differential equation \( \mathcal{L}(u) = f \) where \(
\mathcal{L} \) is a differential operator, \( u \) is the unknown function, and \( f \) is a
given function. The problem is subject to boundary conditions and initial conditions.

# Example
For a 1-dimensional problem, the domain might be \([a, b]\) and the equation could be: \[
\frac{d^2 u}{dx^2} + k u = 0 \] with boundary conditions \( u(a) = u_a \) and \( u(b) = u_b
\).

# Constructor
	KSProblem(; equation::Function,
			   boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
			   domain::NTuple{N, Tuple{T1, T2}},
			   coordinate_system::AbstractKSCoordinateSystem,
			   initial_conditions::Union{Nothing, AbstractVector, Function} = nothing,
			   tspan::Union{Nothing, Tuple{<:Number, <:Number}} = nothing,
			   continuity_order::Int = 2,
			   num_vars::Int = 1) where {N, T1 <: Number, T2 <: Number}

Constructs a KSProblem with the given parameters.
"""
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
@doc raw"""
	struct KSPDEProblem{T <: Number, N} <: AbstractKSProblem

Represents a partial differential equation (PDE) problem in the KS system.

# Fields
- `pde::Function`: The PDE defining the problem.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSPDEProblem is defined by a partial differential equation \( \mathcal{L}(u) = f \) where
\( \mathcal{L} \) is a differential operator, \( u \) is the unknown function, and \( f \) is
a given function. The problem is subject to boundary conditions and initial conditions.

# Example
## 1D Example
For a 1-dimensional PDE problem, the domain might be \([a, b]\) and the equation could be: \[
\frac{\partial^2 u}{\partial x^2} + k u = 0 \] with boundary conditions \( u(a) = u_a \) and
\( u(b) = u_b \), and initial conditions \( u(x, 0) = u_0(x) \).

## 3D Example
For a 3-dimensional PDE problem, the domain might be \([a, b] \times [c, d] \times [e, f]\)
and the equation could be: \[ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2
u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} + k u = 0 \] with boundary conditions \(
u(a, y, z) = u_a(y, z) \), \( u(b, y, z) = u_b(y, z) \), \( u(x, c, z) = u_c(x, z) \), \(
u(x, d, z) = u_d(x, z) \), \( u(x, y, e) = u_e(x, y) \), and \( u(x, y, f) = u_f(x, y) \),
and initial conditions \( u(x, y, z, 0) = u_0(x, y, z) \).

# Constructor
	KSPDEProblem(; pde::Function,
				  boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
				  domain::NTuple{N, Tuple{T, T}},
				  initial_conditions::Union{AbstractVector{T}, Function},
				  tspan::Tuple{T, T},
				  coordinate_system::AbstractKSCoordinateSystem,
				  num_vars::Int = 1) where {T <: Number, N}

Constructs a KSPDEProblem with the given parameters.
"""
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
		domain_T = convert.(Tuple{T,T}, domain)
		initial_conditions_T = if initial_conditions isa AbstractVector
			convert(Vector{T}, initial_conditions)
		else
			initial_conditions
		end
		tspan_T = tspan === nothing ? nothing : convert(Tuple{T,T}, tspan)

		return new{T, N}(
			pde,
			boundary_conditions,
			domain_T,
			initial_conditions_T,
			tspan_T,
			coordinate_system,
			num_vars
		)
	end
end
@doc raw"""
	struct KSODEProblem{T <: Number, N} <: AbstractKSProblem

Defines an ODE problem with boundary conditions, domain, initial conditions, time span, and
coordinate system.

# Fields
- `ode::Function`: The ordinary differential equation function.
- `boundary_conditions::Function`: The boundary conditions function.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain as a tuple of intervals.
- `    initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions
  vector.
- `tspan::Tuple{T, T}`: The time span as a tuple (start, end).
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system.
- `num_vars::Int`: The number of variables.

# Mathematical Description
The ODE problem is defined as: \[ \frac{du}{dt} = \text{ode}(u, t, x) \] with boundary
conditions: \[ \text{boundary\_conditions}(u, t, x) = 0 \] over the domain: \[ \text{domain}
= \{(x_1, x_2, \ldots, x_N) \mid x_i \in [a_i, b_i]\} \] with initial conditions: \[ u(x,
t_0) = \text{initial\_conditions}(x) \] and time span: \[ t \in [t_0, t_f] \]
"""
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

@doc raw"""
	struct KSDAEProblem{T <: Number, N} <: AbstractKSProblem

Represents a differential-algebraic equation (DAE) problem in the KS system.

# Fields
- `dae::Function`: The DAE defining the problem.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.
- `num_algebraic_vars::Int`: The number of algebraic variables in the problem.

# Mathematical Description
A KSDAEProblem is defined by a differential-algebraic equation \( \mathcal{F}(t, u, \dot{u})
= 0 \) where \( \mathcal{F} \) is a function of time \( t \), the unknown function \( u \),
and its time derivative \( \dot{u} \). The problem is subject to boundary conditions and
initial conditions.

# Example
For a 1-dimensional DAE problem, the domain might be \([a, b]\) and the equation could be: \[
\mathcal{F}(t, u, \dot{u}) = \frac{d^2 u}{dt^2} + k u = 0 \] with boundary conditions \( u(a)
= u_a \) and \( u(b) = u_b \), and initial conditions \( u(t_0) = u_0 \).

# Constructor
	KSDAEProblem(; dae::Function,
				  boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
				  domain::NTuple{N, Tuple{T, T}},
				  initial_conditions::Union{AbstractVector{T}, Function},
				  tspan::Tuple{T, T},
				  coordinate_system::AbstractKSCoordinateSystem,
				  num_vars::Int = length(initial_conditions),
				  num_algebraic_vars::Int = 0) where {T <: Number, N}

Constructs a KSDAEProblem with the given parameters.

# Index Explanation
- `dae`: The function representing the differential-algebraic equation.
- `boundary_conditions`: A vector of boundary conditions that the solution must satisfy.
- `domain`: The spatial domain of the problem, represented as a tuple of intervals.
- `initial_conditions`: The initial conditions for the problem, which can be a vector of
  values or a function.
- `tspan`: The time span over which the problem is defined.
- `coordinate_system`: The coordinate system used in the problem.
- `num_vars`: The total number of variables in the problem.
- `num_algebraic_vars`: The number of algebraic variables in the problem, which must be
  non-negative and less than `num_vars`.
"""
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
		domain_T = convert.(Tuple{T,T}, domain)
		initial_conditions_T = if initial_conditions isa AbstractVector
			convert(Vector{T}, initial_conditions)
		else
			initial_conditions
		end
		tspan_T = convert(Tuple{T,T}, tspan)

		return new{T, N}(
			dae,
			boundary_conditions,
			domain_T,
			initial_conditions_T,
			tspan_T,
			coordinate_system,
			num_vars,
			num_algebraic_vars
		)
	end
end
@doc raw"""
	struct KSBVDAEProblem{T <: Number, N} <: AbstractKSProblem

Represents a boundary value differential-algebraic equation (BVDAE) problem in the KS system.

# Fields
- `f::Function`: The differential part of the BVDAE.
- `g::Function`: The algebraic part of the BVDAE.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `algebraic_vars::AbstractVector{Bool}`: A vector indicating which variables are algebraic.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSBVDAEProblem is defined by a system of differential-algebraic equations: \[ \begin{cases}
\mathcal{F}(t, u, \dot{u}) = 0 & \text{(differential part)} \\
\mathcal{G}(t, u) = 0 & \text{(algebraic part)} \end{cases} \] where \( \mathcal{F} \) and \(
\mathcal{G} \) are functions of time \( t \), the unknown function \( u \), and its time
derivative \( \dot{u} \). The problem is subject to boundary conditions and initial
conditions.

# Example
For a 1-dimensional BVDAE problem, the domain might be \([a, b]\) and the equations could be:
\[ \begin{cases} \frac{d^2 u}{dt^2} + k u = 0 & \text{(differential part)} \\
u(t) - v(t) = 0 & \text{(algebraic part)} \end{cases} \] with boundary conditions \( u(a) =
u_a \) and \( u(b) = u_b \), and initial conditions \( u(t_0) = u_0 \).

# Constructor
	KSBVDAEProblem(; f::Function, g::Function,
					boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
					domain::NTuple{N, Tuple{T1, T2}},
					initial_conditions::AbstractVector{T3},
					algebraic_vars::AbstractVector{Bool},
					tspan::Tuple{T4, T5},
					coordinate_system::AbstractKSCoordinateSystem,
					num_vars::Int = length(initial_conditions)) where {N, T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number, T5 <: Number}

Constructs a KSBVDAEProblem with the given parameters.

# Index Explanation
- `f`: The function representing the differential part of the BVDAE.
- `g`: The function representing the algebraic part of the BVDAE.
- `boundary_conditions`: A vector of boundary conditions that the solution must satisfy.
- `domain`: The spatial domain of the problem, represented as a tuple of intervals.
- `initial_conditions`: The initial conditions for the problem, which can be a vector of
  values or a function.
- `algebraic_vars`: A vector indicating which variables are algebraic (true) and which are
  differential (false).
- `tspan`: The time span over which the problem is defined.
- `coordinate_system`: The coordinate system used in the problem.
- `num_vars`: The total number of variables in the problem.
"""
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
@doc raw"""
	struct KSIDEProblem{T <: Number, N} <: AbstractKSProblem

Represents an integral-differential equation (IDE) problem in the KS system.

# Fields
- `ide::Function`: The integral-differential equation defining the problem.
- `kernel::Function`: The kernel function used in the integral part of the IDE.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSIDEProblem is defined by an integral-differential equation: \[ \mathcal{L}(u) +
\int_{\Omega} K(x, s) u(s) \, ds = f(x) \] where \( \mathcal{L} \) is a differential
operator, \( K \) is the kernel function, \( u \) is the unknown function, and \( f \) is a
given function. The problem is subject to boundary conditions and initial conditions.

# Example
For a 1-dimensional IDE problem, the domain might be \([a, b]\) and the equation could be: \[
\frac{d^2 u}{dx^2} + \int_a^b K(x, s) u(s) \, ds = f(x) \] with boundary conditions \( u(a) =
u_a \) and \( u(b) = u_b \), and initial conditions \( u(x, 0) = u_0(x) \).

# Constructor
	KSIDEProblem(; ide::Function, kernel::Function,
				  boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
				  domain::NTuple{N, Tuple{T, T}},
				  initial_conditions::Union{AbstractVector{T}, Function},
				  tspan::Tuple{T, T},
				  coordinate_system::AbstractKSCoordinateSystem,
				  num_vars::Int = length(initial_conditions)) where {T <: Number, N}

Constructs a KSIDEProblem with the given parameters.

# Index Explanation
- `ide`: The function representing the integral-differential equation.
- `kernel`: The kernel function used in the integral part of the IDE.
- `boundary_conditions`: A vector of boundary conditions that the solution must satisfy.
- `domain`: The spatial domain of the problem, represented as a tuple of intervals.
- `initial_conditions`: The initial conditions for the problem, which can be a vector of
  values or a function.
- `tspan`: The time span over which the problem is defined.
- `coordinate_system`: The coordinate system used in the problem.
- `num_vars`: The total number of variables in the problem.
"""
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

@doc raw"""
	struct KSPIDEProblem{T <: Number, N} <: AbstractKSProblem

Represents a partial integral-differential equation (PIDE) problem in the KS system.

# Fields
- `pide::Function`: The partial integral-differential equation defining the problem.
- `kernel::Function`: The kernel function used in the integral part of the PIDE.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSPIDEProblem is defined by a partial integral-differential equation: \[ \mathcal{L}(u) +
\int_{\Omega} K(x, s) u(s) \, ds = f(x) \] where \( \mathcal{L} \) is a differential
operator, \( K \) is the kernel function, \( u \) is the unknown function, and \( f \) is a
given function. The problem is subject to boundary conditions and initial conditions.

# Example
For a 1-dimensional PIDE problem, the domain might be \([a, b]\) and the equation could be:
\[ \frac{\partial^2 u}{\partial x^2} + \int_a^b K(x, s) u(s) \, ds = f(x) \] with boundary
conditions \( u(a) = u_a \) and \( u(b) = u_b \), and initial conditions \( u(x, 0) = u_0(x)
\).

# Constructor
	KSPIDEProblem(; pide::Function, kernel::Function,
				  boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
				  domain::NTuple{N, Tuple{T, T}},
				  initial_conditions::Union{AbstractVector{T}, Function},
				  tspan::Tuple{T, T},
				  coordinate_system::AbstractKSCoordinateSystem,
				  num_vars::Int = isa(initial_conditions, AbstractVector) ? length(initial_conditions) : 1) where {T <: Number, N}

Constructs a KSPIDEProblem with the given parameters.

# Index Explanation
- `pide`: The function representing the partial integral-differential equation.
- `kernel`: The kernel function used in the integral part of the PIDE.
- `boundary_conditions`: A vector of boundary conditions that the solution must satisfy.
- `domain`: The spatial domain of the problem, represented as a tuple of intervals.
- `initial_conditions`: The initial conditions for the problem, which can be a vector of
  values or a function.
- `tspan`: The time span over which the problem is defined.
- `coordinate_system`: The coordinate system used in the problem.
- `num_vars`: The total number of variables in the problem.
"""
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
@doc raw"""
	struct KSMovingBoundaryPDEProblem{T <: Number, N} <: AbstractKSProblem

Represents a partial differential equation (PDE) problem with moving boundaries in the KS
system.

# Fields
- `pde::Function`: The PDE defining the problem.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `boundary_motion::Function`: The function defining the motion of the boundaries.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSMovingBoundaryPDEProblem is defined by a partial differential equation \( \mathcal{L}(u)
= f \) where \( \mathcal{L} \) is a differential operator, \( u \) is the unknown function,
and \( f \) is a given function. The problem is subject to boundary conditions, initial
conditions, and moving boundaries.

# Example
For a 1-dimensional PDE problem with moving boundaries, the domain might be \([a(t), b(t)]\)
and the equation could be: \[ \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial
x^2} \] with boundary conditions \( u(a(t), t) = u_a(t) \) and \( u(b(t), t) = u_b(t) \),
initial conditions \( u(x, 0) = u_0(x) \), and boundary motion functions \( a(t) \) and \(
b(t) \).

# Constructor
	KSMovingBoundaryPDEProblem(; pde::Function,
							   boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
							   domain::NTuple{N, Tuple{T, T}},
							   initial_conditions::Union{AbstractVector{T}, Function},
							   tspan::Tuple{T, T},
							   boundary_motion::Function,
							   coordinate_system::AbstractKSCoordinateSystem,
							   num_vars::Int = length(initial_conditions)) where {T <: Number, N}

Constructs a KSMovingBoundaryPDEProblem with the given parameters.

# Index Explanation
- `pde`: The function representing the partial differential equation.
- `boundary_conditions`: A vector of boundary conditions that the solution must satisfy.
- `domain`: The spatial domain of the problem, represented as a tuple of intervals.
- `initial_conditions`: The initial conditions for the problem, which can be a vector of
  values or a function.
- `tspan`: The time span over which the problem is defined.
- `boundary_motion`: The function defining the motion of the boundaries.
- `coordinate_system`: The coordinate system used in the problem.
- `num_vars`: The total number of variables in the problem.
"""
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

@doc raw"""
	struct KSBVPProblem{T <: Number, N} <: AbstractKSProblem

Represents a boundary value problem (BVP) in the KS system.

# Fields
- `bvp::Function`: The boundary value problem defining the problem.
- `boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}`: The boundary
  conditions for the problem.
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem, defined as a tuple of
  intervals.
- `initial_guess::Union{AbstractVector{T}, Function}`: The initial guess for the solution.
- `tspan::Tuple{T, T}`: The time span for the problem.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system used in the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSBVPProblem is defined by a boundary value problem: \[ \mathcal{L}(u) = f \] where \(
\mathcal{L} \) is a differential operator, \( u \) is the unknown function, and \( f \) is a
given function. The problem is subject to boundary conditions and an initial guess for the
solution.

# Example
For a 1-dimensional BVP, the domain might be \([a, b]\) and the equation could be: \[
\frac{d^2 u}{dx^2} + k u = 0 \] with boundary conditions \( u(a) = u_a \) and \( u(b) = u_b
\), and an initial guess \( u_0(x) \).

# Constructor
	KSBVPProblem(; bvp::Function,
				  boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
				  domain::NTuple{N, Tuple{T, T}},
				  initial_guess::Union{AbstractVector{T}, Function},
				  tspan::Tuple{T, T},
				  coordinate_system::AbstractKSCoordinateSystem,
				  num_vars::Int = isa(initial_guess, AbstractVector) ? length(initial_guess) : 1) where {T <: Number, N}

Constructs a KSBVPProblem with the given parameters.

# Index Explanation
- `bvp`: The function representing the boundary value problem.
- `boundary_conditions`: A vector of boundary conditions that the solution must satisfy.
- `domain`: The spatial domain of the problem, represented as a tuple of intervals.
- `initial_guess`: The initial guess for the solution, which can be a vector of values or a
  function.
- `tspan`: The time span over which the problem is defined.
- `coordinate_system`: The coordinate system used in the problem.
- `num_vars`: The total number of variables in the problem.
"""
struct KSBVPProblem{T <: Number, N} <: AbstractKSProblem
	bvp::Function
	boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition}
	domain::NTuple{N, Tuple{T, T}}
	initial_guess::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}
	coordinate_system::AbstractKSCoordinateSystem
	num_vars::Int

	function KSBVPProblem(;
		bvp::Function,
		boundary_conditions::AbstractVector{<:AbstractKSBoundaryCondition},
		domain::NTuple{N, Tuple{T, T}},
		initial_guess::Union{AbstractVector{T}, Function},
		tspan::Tuple{T, T},
		coordinate_system::AbstractKSCoordinateSystem,
		num_vars::Int = isa(initial_guess, AbstractVector) ? length(initial_guess) : 1,
	) where {T <: Number, N}
		if !isa(bvp, Function)
			throw(ArgumentError("BVP must be a Function, but got $(typeof(bvp))"))
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
		if !(isa(initial_guess, AbstractVector{T}) || isa(initial_guess, Function))
			throw(
				ArgumentError(
					"initial_guess must be an AbstractVector or Function, but got $(typeof(initial_guess))"
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
		domain_T = convert.(Tuple{T,T}, domain)
		initial_guess_T = if initial_guess isa AbstractVector
			convert(Vector{T}, initial_guess)
		else
			initial_guess
		end
		tspan_T = tspan === nothing ? nothing : convert(Tuple{T,T}, tspan)

		return new{T, N}(
			bvp,
			boundary_conditions,
			domain_T,
			initial_guess_T,
			tspan_T,
			coordinate_system,
			num_vars
		)
	end
end
@doc raw"""
	struct KSCoupledProblem <: AbstractKSProblem

Represents a coupled problem in the KS system, where multiple sub-problems are coupled
together.

# Fields
- `problems::AbstractVector{<:AbstractKSProblem}`: A vector of sub-problems that are coupled
  together.
- `coupling_terms::Matrix{Union{Nothing, Function}}`: A matrix representing the coupling
  terms between the sub-problems.
- `num_vars::Int`: The total number of variables in the coupled problem.

# Mathematical Description
A KSCoupledProblem is defined by a set of sub-problems \( \{P_i\} \) and coupling terms \(
C_{ij} \). Each sub-problem \( P_i \) is an instance of `AbstractKSProblem`. The coupling
terms \( C_{ij} \) define how the sub-problems are coupled together. If \( C_{ij} \) is a
function, it represents a coupling term between sub-problem \( P_i \) and sub-problem \( P_j
\). If \( C_{ij} \) is `nothing`, there is no direct coupling between \( P_i \) and \( P_j
\).

# Example
Consider two sub-problems \( P_1 \) and \( P_2 \) with coupling terms \( C_{12} \) and \(
C_{21} \). The coupled problem can be represented as: \[ \begin{cases} P_1 \\
P_2 \\
\end{cases} \] with coupling terms: \[ C = \begin{pmatrix} \text{nothing} & C_{12} \\
C_{21} & \text{nothing} \end{pmatrix} \]

# Constructor
	KSCoupledProblem(; problems::AbstractVector{<:AbstractKSProblem},
					  coupling_terms::Matrix{Union{Nothing, Function}})

Constructs a KSCoupledProblem with the given parameters.

# Index Explanation
- `problems`: A vector of sub-problems that are coupled together.
- `coupling_terms`: A matrix representing the coupling terms between the sub-problems.
- `num_vars`: The total number of variables in the coupled problem, calculated as the sum of
  the number of variables in each sub-problem.
"""
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
@doc raw"""
	struct KSDiscretizedProblem{T <: Number, N, M, MT <: AbstractMatrix} <: AbstractKSProblem

Represents a discretized problem in the KS system, where the problem is discretized in both
time and space.

# Fields
- `time_nodes::AbstractVector{T}`: The discretized time nodes.
- `spatial_nodes::NTuple{N, AbstractVector{T}}`: The discretized spatial nodes for each
  dimension.
- `system_matrix::MT`: The system matrix representing the discretized problem.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the
  problem.
- `problem_functions::NTuple{M, Function}`: The functions defining the problem.
- `num_vars::Int`: The number of variables in the problem.

# Mathematical Description
A KSDiscretizedProblem is defined by discretizing a continuous problem in both time and
space. The discretized problem can be represented as a system of equations: \[ A \mathbf{u} =
\mathbf{b} \] where \( A \) is the system matrix, \( \mathbf{u} \) is the vector of unknowns,
and \( \mathbf{b} \) is the vector of known values. The problem is subject to initial
conditions and defined by a set of problem functions.

# Example
For a 1-dimensional problem, the time nodes might be \(\{t_0, t_1, \ldots, t_n\}\) and the
spatial nodes might be \(\{x_0, x_1, \ldots, x_m\}\). The system matrix \( A \) represents
the discretized differential operator, and the initial conditions \( u(x, t_0) = u_0(x) \).

# Constructor
	KSDiscretizedProblem(time_nodes::AbstractVector{T}, spatial_nodes::NTuple{N, AbstractVector{T}},
						 system_matrix::MT, initial_conditions::Union{AbstractVector{T}, Function},
						 problem_functions::NTuple{M, Function}; num_vars::Int = 1) where {T <: Number, N, M, MT <: AbstractMatrix}

Constructs a KSDiscretizedProblem with the given parameters.

# Index Explanation
- `time_nodes`: The discretized time nodes.
- `spatial_nodes`: The discretized spatial nodes for each dimension.
- `system_matrix`: The system matrix representing the discretized problem.
- `initial_conditions`: The initial conditions for the problem, which can be a vector of
  values or a function.
- `problem_functions`: The functions defining the problem.
- `num_vars`: The total number of variables in the problem.
"""
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
@doc raw"""
	struct KSOptimalControlProblem{T <: Number} <: AbstractKSOptimizationProblem

Represents an optimal control problem in the KS system.

# Fields
- `state_equations::AbstractVector{<:Function}`: The state equations defining the dynamics of
  the system.
- `cost_functions::AbstractVector{<:Function}`: The cost functions to be minimized.
- `terminal_cost::Function`: The terminal cost function to be minimized at the final time.
- `initial_state::AbstractVector{T}`: The initial state of the system.
- `t_span::Tuple{T, T}`: The time span for the control problem.
- `control_bounds::AbstractVector{NTuple{2, T}}`: The bounds on the control variables.
- `num_vars::Int`: The number of state variables in the problem.

# Mathematical Description
A KSOptimalControlProblem is defined by a set of state equations, cost functions, and control
bounds. The goal is to find the control inputs that minimize the cost functions while
satisfying the state equations and control bounds.

The state equations are given by: \[ \dot{x}(t) = f(x(t), u(t), t) \] where \( x(t) \) is the
state vector, \( u(t) \) is the control vector, and \( t \) is time.

The cost functions are given by: \[ J = \int_{t_0}^{t_f} L(x(t), u(t), t) \, dt +
\phi(x(t_f)) \] where \( L \) is the running cost and \( \phi \) is the terminal cost.

The control bounds are given by: \[ u_{\min} \leq u(t) \leq u_{\max} \]

# Constructor
	KSOptimalControlProblem(state_equations::AbstractVector{<:Function},
							cost_functions::AbstractVector{<:Function},
							terminal_cost::Function,
							initial_state::AbstractVector{T},
							t_span::Tuple{T, T},
							control_bounds::AbstractVector{NTuple{2, T}}) where {T <: Number}

Constructs a KSOptimalControlProblem with the given parameters.

# Index Explanation
- `state_equations`: The state equations defining the dynamics of the system.
- `cost_functions`: The cost functions to be minimized.
- `terminal_cost`: The terminal cost function to be minimized at the final time.
- `initial_state`: The initial state of the system.
- `t_span`: The time span for the control problem.
- `control_bounds`: The bounds on the control variables.
- `num_vars`: The number of state variables in the problem.
"""
struct KSOptimalControlProblem{T <: Number} <: AbstractKSOptimizationProblem
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

# Cartesian Coordinates

@doc raw"""
	struct KSCartesianCoordinates{T <: Number, N} <: AbstractKSCoordinateSystem

The `KSCartesianCoordinates` struct represents a Cartesian coordinate system in `N`
dimensions.

## Fields
- `ranges::NTuple{N, Tuple{T, T}}`: The ranges of each dimension in the Cartesian coordinate
  system.
- `active::NTuple{N, Bool}`: A tuple indicating whether each dimension is active or not.
- `domains::NTuple{N, Tuple{T, T}}`: The domains of each dimension in the Cartesian
  coordinate system.

## Constructors
- `KSCartesianCoordinates{T, N}(ranges::NTuple{N, Tuple{T, T}})`: Creates a new
  `KSCartesianCoordinates` object with the specified ranges for each dimension.

	Arguments:
	- `ranges::NTuple{N, Tuple{T, T}}`: The ranges of each dimension in the Cartesian
	  coordinate system.

# Mathematical Description
The Cartesian coordinate system is defined by the ranges of each dimension: \[ \text{ranges}
= \{(a_i, b_i) \mid i = 1, 2, \ldots, N\} \] where \(a_i\) and \(b_i\) are the lower and
upper bounds of the \(i\)-th dimension, respectively.

The `active` field indicates whether each dimension is active: \[ \text{active} = \{
\text{true} \text{ if dimension } i \text{ is active, false otherwise} \} \]

The `domains` field is initialized to be the same as `ranges`: \[ \text{domains} =
\text{ranges} \]

# Errors
Throws an `ArgumentError` if any of the ranges are invalid.
"""
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
		NTuple{N, Tuple{T1, T2}}},
) where {N, T1 <: Number, T2 <: Number}
	T = promote_type(T1, T2)
	promoted_ranges = map(r -> (convert(T, r[1]), convert(T, r[2])), ranges)
	return KSCartesianCoordinates{T, length(ranges)}(promoted_ranges)
end

@doc raw"""
	validate_range(range::Tuple{T, T}, name::String, min_val::T = -Inf, max_val::T = Inf) where T <: Number

Validates a range tuple to ensure the lower bound is less than the upper bound and within
specified limits.

# Arguments
- `range::Tuple{T, T}`: The range tuple to validate.
- `name::String`: The name of the range (used for error messages).
- `min_val::T`: The minimum allowable value for the lower bound (default is `-Inf`).
- `max_val::T`: The maximum allowable value for the upper bound (default is `Inf`).

# Returns
- A validated range tuple.

# Mathematical Description
The function checks the following conditions:
1. The lower bound must be less than the upper bound: \[ \text{lower} < \text{upper} \]
2. The lower bound must be greater than or equal to `min_val`: \[ \text{lower} \geq
\text{min_val} \]
3. The upper bound must be less than or equal to `max_val`: \[ \text{upper} \leq
\text{max_val} \]

# Errors
Throws an `ArgumentError` if any of the conditions are not met.
"""
function validate_range(
	range::Tuple{T, T},
	name::String,
	min_val::T = -Inf,
	max_val::T = Inf,
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

# Polar Coordinates
@doc raw"""
	struct KSPolarCoordinates{T <: Number} <: AbstractKSCoordinateSystem

A type representing a 2D polar coordinate system.

# Fields
- `r::Tuple{T, T}`: Radial distance tuple `(r_min, r_max)`.
- `theta::Union{Tuple{T, T}, Nothing}`: Angular range tuple `(theta_min, theta_max)` in
  radians. If `nothing`, `theta` is inactive.
- `active::NTuple{2, Bool}`: A tuple indicating the active status of `r` and `theta` (`(true,
  true)` or `(true, false)`).
- `domains::NTuple{2, Union{Tuple{T, T}, Nothing}}`: A tuple containing the validated domains
  for `r` and `theta`.

# Mathematical Description
In a 2D polar coordinate system, a point is represented by its radial distance \( r \) from
the origin and its angle \( \theta \) from the positive x-axis. The coordinates are given by:
\[ (x, y) = (r \cos(\theta), r \sin(\theta)) \] where \( r \) is the radial distance and \(
\theta \) is the angle in radians.

# Constructors
- `KSPolarCoordinates{T}(r::Tuple{T, T}, theta::Union{Tuple{T, T}, Nothing} = nothing) where
  T <: Number`: Inner constructor that validates and initializes the polar coordinate system.
- `KSPolarCoordinates(r::Tuple{T1, T2}, theta::Union{Tuple{T3, T4}, Nothing} = nothing) where
  {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number}`: Outer constructor that promotes
  types and handles inactive domains.

# Example
```julia
r_range = (0.0, 5.0)
theta_range = (0.0, π)
polar_coords = KSPolarCoordinates(r_range, theta_range)
"""
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
@doc raw"""
	struct KSCylindricalCoordinates{T <: Number} <: AbstractKSCoordinateSystem

A type representing a 3D cylindrical coordinate system.

# Fields
- `r::Tuple{T, T}`: Radial distance tuple `(r_min, r_max)`.
- `theta::Union{Tuple{T, T}, Nothing}`: Angular range tuple `(theta_min, theta_max)` in
  radians. If `nothing`, `theta` is inactive.
- `z::Union{Tuple{T, T}, Nothing}`: Axial distance tuple `(z_min, z_max)`. If `nothing`, `z`
  is inactive.
- `active::NTuple{3, Bool}`: A tuple indicating the active status of `r`, `theta`, and `z`
  (`(true, true, true)` or other combinations).
- `domains::NTuple{3, Union{Tuple{T, T}, Nothing}}`: A tuple containing the validated domains
  for `r`, `theta`, and `z`.

# Mathematical Description
The cylindrical coordinate system is a 3D coordinate system where a point in space is
represented by three coordinates: radial distance \( r \), azimuthal angle \( \theta \), and
axial distance \( z \). The relationships between cylindrical coordinates \((r, \theta, z)\)
and Cartesian coordinates \((x, y, z)\) are given by: \[ \begin{align*} x &= r \cos(\theta)
\\
y &= r \sin(\theta) \\
z &= z \end{align*} \] where \( r \) is the distance from the origin to the projection of the
point in the \( xy \)-plane, \( \theta \) is the angle between the positive \( x \)-axis and
the line connecting the origin to the projection of the point in the \( xy \)-plane, and \( z
\) is the height of the point above the \( xy \)-plane.

# Constructors
- `KSCylindricalCoordinates{T}(r::Tuple{T, T}, theta::Union{Tuple{T, T}, Nothing},
  z::Union{Tuple{T, T}, Nothing}) where T <: Number`: Inner constructor that validates and
  initializes the cylindrical coordinate system.
- `KSCylindricalCoordinates(r::Tuple{T1, T2}, theta::Union{Tuple{T3, T4}, Nothing},
  z::Union{Tuple{T5, T6}, Nothing}) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <:
  Number, T5 <: Number, T6 <: Number}`: Outer constructor that promotes types and handles
  inactive domains.

# Example
```julia
r_range = (0.0, 5.0)
theta_range = (0.0, π)
z_range = (0.0, 10.0)
cylindrical_coords = KSCylindricalCoordinates(r_range, theta_range, z_range)
"""
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
@doc raw"""
	struct KSSphericalCoordinates{T <: Number} <: AbstractKSCoordinateSystem

A type representing a 3D spherical coordinate system.

# Fields
- `r::Tuple{T, T}`: Radial distance tuple `(r_min, r_max)`.
- `theta::Union{Tuple{T, T}, Nothing}`: Polar angle range tuple `(theta_min, theta_max)` in
  radians. If `nothing`, `theta` is inactive.
- `phi::Union{Tuple{T, T}, Nothing}`: Azimuthal angle range tuple `(phi_min, phi_max)` in
  radians. If `nothing`, `phi` is inactive.
- `active::NTuple{3, Bool}`: A tuple indicating the active status of `r`, `theta`, and `phi`
  (`(true, true, true)` or other combinations).
- `domains::NTuple{3, Union{Tuple{T, T}, Nothing}}`: A tuple containing the validated domains
  for `r`, `theta`, and `phi`.

# Mathematical Description
The spherical coordinate system is a 3D coordinate system where a point in space is
represented by three coordinates: radial distance \( r \), polar angle \( \theta \), and
azimuthal angle \( \phi \). The relationships between spherical coordinates \((r, \theta,
\phi)\) and Cartesian coordinates \((x, y, z)\) are given by: \[ \begin{align*} x &= r
\sin(\theta) \cos(\phi) \\
y &= r \sin(\theta) \sin(\phi) \\
z &= r \cos(\theta) \end{align*} \] where \( r \) is the distance from the origin to the
point, \( \theta \) is the angle between the positive \( z \)-axis and the line connecting
the origin to the point, and \( \phi \) is the angle between the positive \( x \)-axis and
the projection of the line connecting the origin to the point in the \( xy \)-plane.

# Constructors
- `KSSphericalCoordinates{T}(r::Tuple{T, T}, theta::Union{Tuple{T, T}, Nothing},
  phi::Union{Tuple{T, T}, Nothing}) where T <: Number`: Inner constructor that validates and
  initializes the spherical coordinate system.
- `KSSphericalCoordinates(r::Tuple{T1, T2}, theta::Union{Tuple{T3, T4}, Nothing},
  phi::Union{Tuple{T5, T6}, Nothing}) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <:
  Number, T5 <: Number, T6 <: Number}`: Outer constructor that promotes types and handles
  inactive domains.

# Example
```julia
r_range = (0.0, 5.0)
theta_range = (0.0, π)
phi_range = (0.0, 2π)
spherical_coords = KSSphericalCoordinates(r_range, theta_range, phi_range)
"""
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

# Boundary Conditions

@doc raw"""
  on the boundary.
- `coordinate_system`: The coordinate system (Cartesian, Polar, Cylindrical, Spherical, etc.)
  that defines the domain. Default: `KSCartesianCoordinates(((0.0, 1.0),))`

# Example
```julia
# Apply u(x) = 1.0 on the boundary in a Cartesian coordinate system
bc = KSDirichletBC(boundary_value = x -> 1.0, coordinate_system = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0))))```
"""

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

@doc raw"""
	struct KSNeumannBC <: AbstractKSBoundaryCondition

Represents a Neumann boundary condition, where a specific flux is enforced on the boundary
(related to the normal derivative of the function).

# Fields
- `flux_value::Union{Function, AbstractArray}`: The function or array specifying the flux
  value \( q(x) \) on the boundary.
- `boundary_region::Function`: The function that defines the region of the boundary \(
  \partial \Omega \) where the condition is applied.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system that contains the
  domain where the boundary condition is applied.

# Mathematical Formulation
A Neumann boundary condition specifies that the normal derivative of the solution \( u(x) \)
on the boundary \( \partial \Omega \) equals a specific flux \( q(x) \):

\[ \frac{\partial u(x)}{\partial n} = q(x), \quad x \in \partial \Omega \]

where \( \frac{\partial u}{\partial n} \) is the derivative of \( u \) in the direction
normal to the boundary.

# Keyword Arguments
- `flux_value`: A function or array that returns the flux value \( q(x) \) (normal
  derivative) to be enforced on the boundary. Default: `x -> 0.0`
- `boundary_region`: A function that returns true for points on the boundary where the
  condition should be applied. Default: `x -> true`, meaning the condition applies everywhere
  on the boundary.
- `coordinate_system`: The coordinate system (Cartesian, Polar, Cylindrical, Spherical, etc.)
  that defines the domain. Default: `KSCartesianCoordinates(((0.0, 1.0),))`

# Example
```julia
# Apply ∂u/∂n = -2.0 on the boundary in a Cartesian coordinate system
bc = KSNeumannBC(flux_value = x -> -2.0, coordinate_system = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0))))```
```
"""

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

@doc raw"""
	struct KSRobinBC <: AbstractKSBoundaryCondition

Represents a Robin boundary condition, which is a combination of Dirichlet and Neumann
boundary conditions. This type is often used to model convective or radiative boundary
conditions.

# Mathematical Formulation
A Robin boundary condition is a weighted combination of Dirichlet and Neumann conditions:

\[ \alpha(x) \frac{\partial u(x)}{\partial n} + \beta(x) u(x) = f(x), \quad x \in \partial
\Omega \]

where:
- \( \alpha(x) \) is the Neumann coefficient (associated with the normal derivative \(
  \frac{\partial u(x)}{\partial n} \)),
- \( \beta(x) \) is the Dirichlet coefficient (associated with the boundary value \( u(x)
  \)),
- \( f(x) \) is the boundary value specified on the boundary.

# Fields
- `neumann_coefficient::Union{Function, AbstractArray}`: Represents \( \alpha(x) \), the
  coefficient function or array for the normal derivative term in the Robin condition.
- `dirichlet_coefficient::Union{Function, AbstractArray}`: Represents \( \beta(x) \), the
  coefficient function or array for the boundary value term in the Robin condition.
- `boundary_value::Union{Function, AbstractArray}`: Represents \( f(x) \), the function or
  array specifying the target value on the boundary.
- `boundary_region::Function`: A function that returns `true` for points on the boundary
  where the condition should be applied.
- `coordinate_system::AbstractKSCoordinateSystem`: The coordinate system that defines the
  domain on which the boundary condition is applied (e.g., Cartesian, Polar, Cylindrical,
  Spherical).

# Keyword Arguments
- `neumann_coefficient`: A function or array representing \( \alpha(x) \), the coefficient
  for the normal derivative term \( \frac{\partial u(x)}{\partial n} \). Default: `x -> 1.0`
- `dirichlet_coefficient`: A function or array representing \( \beta(x) \), the coefficient
  for the boundary value term \( u(x) \). Default: `x -> 1.0`
- `boundary_value`: A function or array representing \( f(x) \), the target value to be
  enforced on the boundary. Default: `x -> 0.0`
- `boundary_region`: A function that returns `true` for points on the boundary where the
  condition should be applied. Default: `x -> true`, meaning the condition applies everywhere
  on the boundary.
- `coordinate_system`: The coordinate system (e.g., Cartesian, Polar, Cylindrical, Spherical)
  that defines the domain where the boundary condition is applied. Default:
  `KSCartesianCoordinates(((0.0, 1.0),))`

# Example
```julia
# Apply 1 * ∂u/∂n + 2 * u = 3 on the boundary in a Cartesian coordinate system
bc = KSRobinBC(neumann_coefficient = x -> 1.0, dirichlet_coefficient = x -> 2.0, boundary_value = x -> 3.0, coordinate_system = KSCartesianCoordinates(((0.0, 1.0), (0.0, 1.0))))```
"""
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

@doc raw"""
	struct KSSolverOptions{T <: Number}

Represents solver options for the KitchenSink framework.

# Fields
- `max_iterations::Int`: Maximum number of iterations.
- `tolerance::T`: Tolerance for convergence.
- `adaptive::Bool`: Whether to use adaptive methods.
- `max_levels::Int`: Maximum number of levels in multigrid methods.
- `smoothness_threshold::T`: Threshold for smoothness in adaptive methods.
- `initial_elements::Int`: Initial number of elements.
- `initial_degree::Int`: Initial polynomial degree.
- `use_domain_decomposition::Bool`: Whether to use domain decomposition methods.
- `num_subdomains::Int`: Number of subdomains for domain decomposition.
- `use_strang_splitting::Bool`: Whether to use Strang splitting for time integration.
- `dt::T`: Time step for the solver.
- `num_steps::Int`: Number of time steps to perform.
- `legendre_threshold::T`: Threshold for Legendre polynomial decay rate to trigger
  refinement.

# Mathematical Description
The `KSSolverOptions` struct encapsulates various parameters used to control the behavior of
solvers in the KitchenSink framework. These parameters include convergence criteria, adaptive
methods, multigrid levels, and domain decomposition settings.

- **Convergence Criteria**: The solver iterates until the number of iterations reaches
`max_iterations` or the solution converges within the `tolerance` \( \epsilon \). \[
\text{Convergence:} \quad \| \text{residual} \| < \epsilon \]

- **Adaptive Methods**: If `adaptive` is true, the solver uses adaptive methods to refine the
solution based on the `smoothness_threshold`. \[ \text{Refinement Criterion:} \quad
\text{smoothness} < \text{smoothness_threshold} \]

- **Multigrid Methods**: The solver can use multigrid methods up to `max_levels` levels to
accelerate convergence. \[ \text{Multigrid Levels:} \quad L \leq \text{max_levels} \]

- **Domain Decomposition**: If `use_domain_decomposition` is true, the domain is decomposed
into `num_subdomains` subdomains. \[ \text{Subdomains:} \quad \Omega =
\bigcup_{i=1}^{\text{num_subdomains}} \Omega_i \]

- **Strang Splitting**: If `use_strang_splitting` is true, the solver uses Strang splitting
for time integration with a time step `dt`. \[ \text{Time Step:} \quad \Delta t = \text{dt}
\]

- **Legendre Polynomial Decay**: The solver refines the solution if the decay rate of
Legendre polynomials exceeds `legendre_threshold`. \[ \text{Refinement Criterion:} \quad
\text{decay rate} > \text{legendre_threshold} \]

# Constructor
	KSSolverOptions(max_iterations::Int, tolerance::T, adaptive::Bool, max_levels::Int,
					smoothness_threshold::T, initial_elements::Int, initial_degree::Int,
					use_domain_decomposition::Bool, num_subdomains::Int,
					use_strang_splitting::Bool, dt::T, num_steps::Int,
					legendre_threshold::T) where {T <: Number}

Constructs a `KSSolverOptions` instance with the given parameters.

# Index Explanation
- `max_iterations`: Maximum number of iterations.
- `tolerance`: Tolerance for convergence.
- `adaptive`: Whether to use adaptive methods.
- `max_levels`: Maximum number of levels in multigrid methods.
- `smoothness_threshold`: Threshold for smoothness in adaptive methods.
- `initial_elements`: Initial number of elements.
- `initial_degree`: Initial polynomial degree.
- `use_domain_decomposition`: Whether to use domain decomposition methods.
- `num_subdomains`: Number of subdomains for domain decomposition.
- `use_strang_splitting`: Whether to use Strang splitting for time integration.
- `dt`: Time step for the solver.
- `num_steps`: Number of time steps to perform.
- `legendre_threshold`: Threshold for Legendre polynomial decay rate to trigger refinement.
"""
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

# Solvers
@doc raw"""
	struct KSDirectSolver <: AbstractKSLinearSolver

Represents a direct linear solver.

# Fields
- `method::Symbol`: The method used for the direct solver.

# Mathematical Description
A direct linear solver is used to solve a system of linear equations of the form: \[ A
\mathbf{x} = \mathbf{b} \] where \( A \) is a matrix, \( \mathbf{x} \) is the vector of
unknowns, and \( \mathbf{b} \) is the vector of known values. The direct solver computes the
solution \( \mathbf{x} \) by directly inverting the matrix \( A \) or using factorization
methods.

# Methods
- `KSDirectSolver(method::Symbol)`: Constructs a `KSDirectSolver` with the specified method.

# Example
```julia
solver = KSDirectSolver(:LU)
```
"""
struct KSDirectSolver <: AbstractKSLinearSolver
	method::Symbol

	function KSDirectSolver(method::Symbol)
		return new(method)
	end
end
@doc raw"""
	struct KSIterativeSolver{T<:Number} <: AbstractKSLinearSolver

Represents an iterative linear solver.

# Fields
- `method::Symbol`: The method used for the iterative solver.
- `maxiter::Int`: The maximum number of iterations.
- `tolerance::T`: The tolerance for convergence.
- `preconditioner::Union{Nothing, Function}`: The preconditioner function, if any.

# Mathematical Description
An iterative linear solver aims to find the solution to a linear system of equations \(Ax =
b\) using iterative methods. The solver iteratively refines the solution \(x\) until the
residual \(r = b - Ax\) is sufficiently small, i.e., \(\|r\| < \text{tolerance}\).

# Example
For a linear system \(Ax = b\), an iterative solver might use methods such as Conjugate
Gradient (CG) or Generalized Minimal Residual (GMRES). The solver stops when the number of
iterations exceeds `maxiter` or the residual norm is less than `tolerance`.

# Constructor
	KSIterativeSolver(method::Symbol, maxiter::Int, tolerance::T, preconditioner::Union{Nothing, Function} = nothing) where {T <: Number}

Constructs a KSIterativeSolver with the given parameters.

# Index Explanation
- `method`: The method used for the iterative solver (e.g., `:CG` for Conjugate Gradient,
  `:GMRES` for Generalized Minimal Residual).
- `maxiter`: The maximum number of iterations allowed.
- `tolerance`: The convergence tolerance. The iteration stops when the residual norm is less
  than this value.
- `preconditioner`: The preconditioner function to accelerate convergence, if any. It should
  be a function that approximates the inverse of \(A\).

# Logical Operators
- The solver iterates until either the maximum number of iterations is reached (`maxiter`) or
  the residual norm is less than the specified tolerance (`tolerance`).
- If a preconditioner is provided, it is applied at each iteration to improve convergence.

# Example Usage
```julia
solver = KSIterativeSolver(:CG, 1000, 1e-6)
```
"""
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

@doc raw"""
	struct KSAMGSolver{T <: Number} <: AbstractKSLinearSolver

Represents an algebraic multigrid (AMG) solver.

# Fields
- `maxiter::Int`: The maximum number of iterations.
- `tolerance::T`: The tolerance for convergence.
- `preconditioner::Any`: The smoother method used in the AMG solver.

# Mathematical Description
An algebraic multigrid solver is an iterative method for solving large linear systems of
equations, particularly those arising from discretized partial differential equations. The
AMG solver operates by recursively coarsening the problem, solving on the coarsest level, and
then interpolating the solution back to the finer levels.

The solver iterates to find the solution \( \mathbf{x} \) to the linear system: \[ A
\mathbf{x} = \mathbf{b} \] where \( A \) is the system matrix and \( \mathbf{b} \) is the
right-hand side vector.

The convergence criterion is based on the tolerance \( \epsilon \): \[ \| A \mathbf{x} -
\mathbf{b} \| < \epsilon \] where \( \| \cdot \| \) denotes the norm.

# Constructor
	KSAMGSolver(maxiter::Int, tolerance::T, preconditioner) where {T <: Number}

Constructs a KSAMGSolver with the given parameters.

# Index Explanation
- `maxiter`: The maximum number of iterations allowed for the solver.
- `tolerance`: The convergence tolerance. The solver stops when the residual norm is less
  than this value.
- `preconditioner`: The smoother method used in the AMG solver, which can be any type of
  preconditioner.

# Example
To create an AMG solver with a maximum of 100 iterations, a tolerance of \(1e-6\), and a
Gauss-Seidel preconditioner:
```julia
solver = KSAMGSolver(100, 1e-6, :gauss_seidel)
```
"""
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
@doc raw"""
	tuple_if_active(dims...)

Filters out `nothing` values from the input dimensions and returns a tuple of the remaining
dimensions.

# Arguments
- `dims...`: A variable number of dimensions, which can include `nothing`.

# Returns
- A tuple containing all non-`nothing` dimensions.

# Mathematical Description
Given a set of dimensions \((d_1, d_2, \ldots, d_n)\), the function filters out any
dimensions that are `nothing` and returns a tuple of the remaining dimensions: \[
\text{tuple\_if\_active}(d_1, d_2, \ldots, d_n) = (d_i \mid d_i \neq \text{nothing}) \]

# Example
```julia
tuple_if_active(1, nothing, 3, nothing, 5)  # Returns (1, 3, 5)
tuple_if_active(nothing, nothing)           # Returns ()
tuple_if_active(2, 4, 6)                    # Returns (2, 4, 6)
```
"""

function tuple_if_active(dims...)
	return Tuple(filter(!isnothing, dims))
end

# Handle coordinate ranges
struct CoordinateRanges{T <: Number, N}
	ranges::NTuple{N, Tuple{T, T}}
	active::NTuple{N, Bool}

	function CoordinateRanges{T, N}(
		ranges::NTuple{N, Tuple{T, T}}, active::NTuple{N, Bool}
	) where {T <: Number, N}
		return new{T, N}(ranges, active)
	end
end
end  # module KSTypes
