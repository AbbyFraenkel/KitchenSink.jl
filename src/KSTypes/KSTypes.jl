module KSTypes

using StaticArrays, SparseArrays

# Exported types
export AbstractKSElement, AbstractKSMesh, AbstractKSProblem, AbstractKSCoordinateSystem
export AbstractKSBoundaryCondition, AbstractKSBasisFunction, AbstractKSSolver, AbstractKSOptimizationProblem
export AbstractKSLinearSolver, StandardElement, KSBasisFunction, KSElement, KSMesh, KSProblem
export KSODEProblem, KSBVDAEProblem, KSCoupledProblem, KSIDEProblem, KSPDEProblem, KSDAEProblem
export KSMovingBoundaryPDEProblem, KSPIDEProblem, KSDiscretizedProblem, KSOptimalControlProblem
export KSCartesianCoordinates, KSPolarCoordinates, KSCylindricalCoordinates, KSSphericalCoordinates
export KSDirichletBC, KSRobinBC, KSNeumannBC, KSSolverOptions, KSDirectSolver, KSIterativeSolver, KSAMGSolver

export is_callable, tuple_if_active, validate_range

# Abstract types

"""
	AbstractKSElement{T<:Number, N}

An abstract type representing an element in N-dimensional space.
"""
abstract type AbstractKSElement{T <: Number, N} end

"""
	AbstractKSMesh{T<:Number, N}

An abstract type representing a mesh in N-dimensional space.
"""
abstract type AbstractKSMesh{T <: Number, N} end

"""
	AbstractKSProblem

An abstract type representing a general problem in the KitchenSink framework.
"""
abstract type AbstractKSProblem end

"""
	AbstractKSCoordinateSystem

An abstract type representing a coordinate system.
"""
abstract type AbstractKSCoordinateSystem end

"""
	AbstractKSBoundaryCondition

An abstract type representing a boundary condition.
"""
abstract type AbstractKSBoundaryCondition end

"""
	AbstractKSBasisFunction

An abstract type representing a basis function.
"""
abstract type AbstractKSBasisFunction end

"""
	AbstractKSSolver

An abstract type representing a solver.
"""
abstract type AbstractKSSolver end

"""
	AbstractKSOptimizationProblem

An abstract type representing an optimization problem.
"""
abstract type AbstractKSOptimizationProblem end

"""
	AbstractKSLinearSolver

An abstract type representing a linear solver.
"""
abstract type AbstractKSLinearSolver end

# Helper functions

to_abstract_vector(x::AbstractVector) = x
to_abstract_vector(x::StaticArray) = collect(x)
to_abstract_vector(x) = [x...]

# Concrete types

"""
	StandardElement{T<:Number, N}

Represents precomputed data for elements of a specific level and polynomial degree.

# Fields
- `points_with_boundary::Vector{SVector{N, T}}`: Points including boundary points in reference space.
- `collocation_points::Vector{SVector{N, T}}`: Collocation points in reference space.
- `collocation_weights::Vector{T}`: Collocation weights for numerical integration.
- `differentiation_matrices::Vector{Matrix{T}}`: Differentiation matrices.
- `quadrature_matrices::Vector{Matrix{T}}`: Quadrature matrices.
- `level::Int`: The refinement level of the element.
"""
struct StandardElement{T <: Number, N}
	points_with_boundary::Vector{NTuple{N, T}}
	collocation_points::Vector{NTuple{N, T}}
	collocation_weights::Vector{T}
	differentiation_matrices::Vector{Matrix{T}}
	quadrature_matrices::Vector{Matrix{T}}
	level::Int

	function StandardElement(points_with_boundary::Vector{NTuple{N, T}},
							 collocation_points::Vector{NTuple{N, T}},
							 collocation_weights::Vector{T},
							 differentiation_matrices::Vector{Matrix{T}},
							 quadrature_matrices::Vector{Matrix{T}},
							 level::Int) where {T <: Number, N}
		if level < 0
			throw(ArgumentError("Level must be non-negative"))
		end
		new{T, N}(points_with_boundary, collocation_points, collocation_weights,
				  differentiation_matrices, quadrature_matrices, level)
	end
end

"""
	KSBasisFunction <: AbstractKSBasisFunction

Represents a basis function.

# Fields
- `id::Int`: An identifier for the basis function.
- `function_handle::Function`: The actual function representing the basis function.
"""
mutable struct KSBasisFunction <: AbstractKSBasisFunction
	id::Int
	function_handle::Function

	function KSBasisFunction(id::Int, function_handle::Function = x -> x)
		if id <= 0
			throw(ArgumentError("Basis function ID must be positive"))
		end
		if !isa(function_handle, Function)
			throw(ArgumentError("function_handle must be a Function, but got $(typeof(function_handle))"))
		end
		new(id, function_handle)
	end
end

"""
	KSElement{T<:Number, N} <: AbstractKSElement{T, N}

Represents an element in N-dimensional space.

# Fields
- `id::Int`: An identifier for the element.
- `level::Int`: The refinement level of the element.
- `polynomial_degree::NTuple{N, Int}`: The polynomial degree of the element.
- `parent::Union{Nothing, Int}`: The ID of the parent element.
- `children::Union{Nothing, Vector{Int}}`: IDs of child elements.
- `neighbors::Union{Nothing, Vector{Int}}`: IDs of neighboring elements.
- `is_leaf::Bool`: Flag indicating if the element is a leaf in the refinement tree.
- `error_estimate::T`: An estimate of the error associated with the element.
- `legendre_decay_rate::T`: The decay rate of Legendre polynomial coefficients for adaptivity.
"""
mutable struct KSElement{T <: Number, N} <: AbstractKSElement{T, N}
	id::Int
	level::Int
	polynomial_degree::NTuple{N, Int}
	parent::Union{Nothing, Int}
	children::Union{Nothing, Vector{Int}}
	neighbors::Union{Nothing, Vector{Int}}
	is_leaf::Bool
	error_estimate::T
	legendre_decay_rate::T

	function KSElement{T, N}(id::Int, level::Int, polynomial_degree::NTuple{N, Int};
							 parent::Union{Nothing, Int} = nothing,
							 children::Union{Nothing, Vector{Int}} = nothing,
							 neighbors::Union{Nothing, Vector{Int}} = nothing,
							 is_leaf::Bool = true,
							 error_estimate::Union{Nothing, T} = nothing,
							 legendre_decay_rate::Union{Nothing, T} = nothing) where {T <: Number, N}
		if id <= 0
			throw(ArgumentError("Element ID must be positive"))
		end
		if level < 0
			throw(ArgumentError("Element level must be non-negative"))
		end
		if any(p -> p < 0, polynomial_degree)
			throw(ArgumentError("Polynomial degrees must be non-negative"))
		end
		error_estimate = error_estimate === nothing ? zero(T) : error_estimate
		legendre_decay_rate = legendre_decay_rate === nothing ? zero(T) : legendre_decay_rate

		new{T, N}(id, level, polynomial_degree, parent, children, neighbors, is_leaf, error_estimate, legendre_decay_rate)
	end
end

# Convenience constructor for Float64
function KSElement(id::Int, level::Int, polynomial_degree::NTuple{N, Int}; kwargs...) where N
	KSElement{Float64, N}(id, level, polynomial_degree; kwargs...)
end


"""
	KSMesh{T <: Number, N} <: AbstractKSMesh{T, N}

Represents a mesh used in the KitchenSink package.

# Fields
- `elements::Vector{KSElement{T, N}}`: Vector of `KSElement` objects representing the elements in the mesh.
- `tensor_product_masks::Vector{AbstractArray{Bool, N}}`: Vector of abstract arrays of booleans with `N` dimensions, representing the tensor product masks.
- `location_matrices::Vector{Dict{Int, Int}}`: Vector of dictionaries mapping integers to integers, representing the location matrices.
- `global_error_estimate::T`: Global error estimate of type `T`.
"""
mutable struct KSMesh{T <: Number, N} <: AbstractKSMesh{T, N}
	elements::Vector{KSElement{T, N}}
	tensor_product_masks::Vector{AbstractArray{Bool, N}}
	location_matrices::Vector{Dict{Int, Int}}
	global_error_estimate::T

	function KSMesh{T, N}(elements::Vector{KSElement{T, N}},
						  tensor_product_masks::Vector{<:AbstractArray{Bool, N}} = Vector{AbstractArray{Bool, N}}(),
						  location_matrices::Vector{Dict{Int, Int}} = Vector{Dict{Int, Int}}(),
						  global_error_estimate::T = zero(T)) where {T <: Number, N}
		if isempty(elements)
			throw(ArgumentError("Elements vector cannot be empty"))
		end
		new{T, N}(elements, tensor_product_masks, location_matrices, global_error_estimate)
	end
end

# Convenience constructor
function KSMesh(elements::Vector{KSElement{T, N}},
				tensor_product_masks::Vector{<:AbstractArray{Bool, N}} = Vector{AbstractArray{Bool, N}}(),
				location_matrices::Vector{Dict{Int, Int}} = Vector{Dict{Int, Int}}(),
				global_error_estimate::T = zero(T)) where {T <: Number, N}
	KSMesh{T, N}(elements, tensor_product_masks, location_matrices, global_error_estimate)
end

# Constructor to accept Vector{BitMatrix}
function KSMesh(elements::Vector{KSElement{T, N}},
				tensor_product_masks::Vector{BitMatrix},
				location_matrices::Vector{Dict{Int, Int}},
				global_error_estimate::T) where {T <: Number, N}
	converted_masks = Vector{AbstractArray{Bool, N}}(undef, length(tensor_product_masks))
	for (i, mask) in enumerate(tensor_product_masks)
		converted_masks[i] = convert(Array{Bool, N}, mask)
	end
	KSMesh{T, N}(elements, converted_masks, location_matrices, global_error_estimate)
end

"""
	KSProblem{T <: Number, N} <: AbstractKSProblem

Represents a general problem in N-dimensional space in the KitchenSink framework.

# Fields
- `equation::Function`: The governing equation of the problem.
- `boundary_conditions::Union{Nothing, Function, Vector{Function}}`: The boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The domain over which the problem is defined.
- `initial_conditions::Union{Nothing, Vector{T}, Function}`: The initial conditions for the problem.
- `tspan::Union{Nothing, Tuple{T, T}}`: The time span over which the problem is defined.
- `continuity_order::Int`: The desired order of continuity (e.g., C^k continuity).
"""
mutable struct KSProblem{T <: Number, N} <: AbstractKSProblem
	equation::Function
	boundary_conditions::Union{Nothing, Function, Vector{Function}}
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{Nothing, Vector{T}, Function}
	tspan::Union{Nothing, Tuple{T, T}}
	continuity_order::Int

	function KSProblem(equation::Function,
					   boundary_conditions::Union{Nothing, Function, Vector{Function}},
					   domain::NTuple{N, Tuple{T, T}},
					   initial_conditions::Union{Nothing, Vector{T}, Function} = nothing,
					   tspan::Union{Nothing, Tuple{T, T}} = nothing,
					   continuity_order::Int = 2) where {T <: Number, N}
		# Type checks
		if !isa(equation, Function)
			throw(ArgumentError("equation must be a Function, but got $(typeof(equation))"))
		end
		if !(isa(boundary_conditions, Function) || isa(boundary_conditions, Vector{<:Function}) || isnothing(boundary_conditions))
			throw(ArgumentError("boundary_conditions must be a Function, Vector of Functions, or Nothing, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if continuity_order < 0
			throw(ArgumentError("Continuity order must be non-negative"))
		end

		new{T, N}(equation, boundary_conditions, domain, initial_conditions, tspan, continuity_order)
	end
end

"""
	KSODEProblem{T <: Number, N} <: AbstractKSProblem

Represents an ordinary differential equation (ODE) problem in N-dimensional space.

# Fields
- `ode::Function`: The ODE function defining the differential equation.
- `boundary_conditions::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::AbstractVector{T}`: The initial conditions for the problem.
- `tspan::Tuple{T, T}`: The time span over which the problem is defined.
"""
struct KSODEProblem{T <: Number, N} <: AbstractKSProblem
	ode::Function
	boundary_conditions::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::AbstractVector{T}
	tspan::Tuple{T, T}

	function KSODEProblem(ode::Function, boundary_conditions::Function,
						  domain::NTuple{N, Tuple{T, T}}, initial_conditions::AbstractVector{T},
						  tspan::Tuple{T, T}) where {T <: Number, N}
		# Type checks
		if !isa(ode, Function)
			throw(ArgumentError("ODE must be a Function, but got $(typeof(ode))"))
		end
		if !isa(boundary_conditions, Function)
			throw(ArgumentError("boundary_conditions must be a Function, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(ArgumentError("initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end

		new{T, N}(ode, boundary_conditions, domain, initial_conditions, tspan)
	end
end

"""
	KSBVDAEProblem{T <: Number, N} <: AbstractKSProblem

Represents a boundary value differential algebraic equation (BVDAE) problem in N-dimensional space.

# Fields
- `f::Function`: The differential equation function.
- `g::Function`: The algebraic constraint function.
- `bc::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::AbstractVector{T}`: The initial conditions for the problem.
- `algebraic_vars::AbstractVector{Bool}`: Flags indicating which variables are algebraic.
- `tspan::Tuple{T, T}`: The time span of the problem.
"""
struct KSBVDAEProblem{T <: Number, N} <: AbstractKSProblem
	f::Function
	g::Function
	bc::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::AbstractVector{T}
	algebraic_vars::AbstractVector{Bool}
	tspan::Tuple{T, T}

	function KSBVDAEProblem(f::Function, g::Function, bc::Function,
							domain::NTuple{N, Tuple{T, T}}, initial_conditions::AbstractVector{T},
							algebraic_vars::AbstractVector{Bool}, tspan::Tuple{T, T}) where {T <: Number, N}
		# Type checks
		if !isa(f, Function)
			throw(ArgumentError("f must be a Function, but got $(typeof(f))"))
		end
		if !isa(g, Function)
			throw(ArgumentError("g must be a Function, but got $(typeof(g))"))
		end
		if !isa(bc, Function)
			throw(ArgumentError("bc must be a Function, but got $(typeof(bc))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if length(initial_conditions) != length(algebraic_vars)
			throw(ArgumentError("Length of initial_conditions must match length of algebraic_vars"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end

		new{T, N}(f, g, bc, domain, initial_conditions, algebraic_vars, tspan)
	end
end

"""
	KSCoupledProblem

Represents a coupled problem consisting of multiple subproblems.

# Fields
- `problems::AbstractVector{<:AbstractKSProblem}`: Array of subproblems.
- `coupling_terms::Matrix{Union{Nothing, Function}}`: Matrix of coupling terms between subproblems.
"""
struct KSCoupledProblem
	problems::AbstractVector{<:AbstractKSProblem}
	coupling_terms::Matrix{Union{Nothing, Function}}

	function KSCoupledProblem(problems::AbstractVector{<:AbstractKSProblem},
							  coupling_terms::Matrix{Union{Nothing, Function}})
		# Type checks
		if !isa(problems, AbstractVector{<:AbstractKSProblem})
			throw(ArgumentError("problems must be an AbstractVector of AbstractKSProblem, but got $(typeof(problems))"))
		end
		if !isa(coupling_terms, Matrix{Union{Nothing, Function}})
			throw(ArgumentError("coupling_terms must be a Matrix of Union{Nothing, Function}, but got $(typeof(coupling_terms))"))
		end
		if size(coupling_terms, 1) != size(coupling_terms, 2) || size(coupling_terms, 1) != length(problems)
			throw(ArgumentError("Inconsistent coupling terms dimensions"))
		end
		new(problems, coupling_terms)
	end
end

"""
	KSIDEProblem{T <: Number, N} <: AbstractKSProblem

Represents an integro-differential equation (IDE) problem in N-dimensional space.

# Fields
- `f::Function`: The function defining the differential equation.
- `K::Function`: The kernel function defining the integral term.
- `boundary_conditions::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::AbstractVector{T}`: The initial conditions for the problem.
- `tspan::Tuple{T, T}`: The time span over which the problem is defined.
"""
struct KSIDEProblem{T <: Number, N} <: AbstractKSProblem
	f::Function
	K::Function
	boundary_conditions::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::AbstractVector{T}
	tspan::Tuple{T, T}

	function KSIDEProblem(f::Function, K::Function, boundary_conditions::Function,
						  domain::NTuple{N, Tuple{T, T}}, initial_conditions::AbstractVector{T},
						  tspan::Tuple{T, T}) where {T <: Number, N}
		# Type checks
		if !isa(f, Function)
			throw(ArgumentError("f must be a Function, but got $(typeof(f))"))
		end
		if !isa(K, Function)
			throw(ArgumentError("K must be a Function, but got $(typeof(K))"))
		end
		if !isa(boundary_conditions, Function)
			throw(ArgumentError("boundary_conditions must be a Function, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(ArgumentError("initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end

		new{T, N}(f, K, boundary_conditions, domain, initial_conditions, tspan)
	end
end

"""
	KSPDEProblem{T <: Number, N} <: AbstractKSProblem

Represents a partial differential equation (PDE) problem in N-dimensional space.

# Fields
- `pde::Function`: The PDE function defining the differential equation.
- `boundary_conditions::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the problem.
- `tspan::Tuple{T, T}`: The time span over which the problem is defined.
"""
struct KSPDEProblem{T <: Number, N} <: AbstractKSProblem
	pde::Function
	boundary_conditions::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}

	function KSPDEProblem(pde::Function, boundary_conditions::Function,
						  domain::NTuple{N, Tuple{T, T}}, initial_conditions::Union{AbstractVector{T}, Function},
						  tspan::Tuple{T, T}) where {T <: Number, N}
		# Type checks
		if !isa(pde, Function)
			throw(ArgumentError("PDE must be a Function, but got $(typeof(pde))"))
		end
		if !isa(boundary_conditions, Function)
			throw(ArgumentError("boundary_conditions must be a Function, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if !(isa(initial_conditions, AbstractVector{T}) || isa(initial_conditions, Function))
			throw(ArgumentError("initial_conditions must be an AbstractVector or Function, but got $(typeof(initial_conditions))"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end

		new{T, N}(pde, boundary_conditions, domain, initial_conditions, tspan)
	end
end

"""
	KSDAEProblem{T <: Number, N} <: AbstractKSProblem

Represents a differential algebraic equation (DAE) problem in N-dimensional space.

# Fields
- `dae::Function`: The DAE function defining the differential equation.
- `boundary_conditions::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::AbstractVector{T}`: The initial conditions for the problem.
- `tspan::Tuple{T, T}`: The time span over which the problem is defined.
"""
struct KSDAEProblem{T <: Number, N} <: AbstractKSProblem
	dae::Function
	boundary_conditions::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::AbstractVector{T}
	tspan::Tuple{T, T}

	function KSDAEProblem(dae::Function, boundary_conditions::Function,
						  domain::NTuple{N, Tuple{T, T}}, initial_conditions::AbstractVector{T},
						  tspan::Tuple{T, T}) where {T <: Number, N}
		# Type checks
		if !isa(dae, Function)
			throw(ArgumentError("DAE must be a Function, but got $(typeof(dae))"))
		end
		if !isa(boundary_conditions, Function)
			throw(ArgumentError("boundary_conditions must be a Function, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(ArgumentError("initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end

		new{T, N}(dae, boundary_conditions, domain, initial_conditions, tspan)
	end
end

"""
	KSMovingBoundaryPDEProblem{T <: Number, N} <: AbstractKSProblem

Represents a partial differential equation (PDE) problem with moving boundaries in N-dimensional space.

# Fields
- `pde::Function`: The PDE function defining the differential equation.
- `boundary_conditions::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::AbstractVector{T}`: The initial conditions for the problem.
- `tspan::Tuple{T, T}`: The time span over which the problem is defined.
- `boundary_motion::Function`: The function defining the motion of the boundary.
"""
struct KSMovingBoundaryPDEProblem{T <: Number, N} <: AbstractKSProblem
	pde::Function
	boundary_conditions::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::AbstractVector{T}
	tspan::Tuple{T, T}
	boundary_motion::Function

	function KSMovingBoundaryPDEProblem(pde::Function, boundary_conditions::Function,
										domain::NTuple{N, Tuple{T, T}}, initial_conditions::AbstractVector{T},
										tspan::Tuple{T, T}, boundary_motion::Function) where {T <: Number, N}
		# Type checks
		if !isa(pde, Function)
			throw(ArgumentError("PDE must be a Function, but got $(typeof(pde))"))
		end
		if !isa(boundary_conditions, Function)
			throw(ArgumentError("boundary_conditions must be a Function, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if !isa(initial_conditions, AbstractVector{T})
			throw(ArgumentError("initial_conditions must be a vector of $T, but got $(typeof(initial_conditions))"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end
		if !isa(boundary_motion, Function)
			throw(ArgumentError("boundary_motion must be a Function, but got $(typeof(boundary_motion))"))
		end

		new{T, N}(pde, boundary_conditions, domain, initial_conditions, tspan, boundary_motion)
	end
end

"""
	KSPIDEProblem{T <: Number, N} <: AbstractKSProblem

Represents a partial integro-differential equation (PIDE) problem in N-dimensional space.

# Fields
- `pide::Function`: The PIDE function defining the differential part of the equation.
- `kernel::Function`: The kernel function defining the integral part of the equation.
- `boundary_conditions::Function`: The function specifying the boundary conditions.
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain over which the problem is defined.
- `initial_conditions::Union{AbstractVector{T}, Function}`: The initial conditions for the problem.
- `tspan::Tuple{T, T}`: The time span over which the problem is defined.
"""
struct KSPIDEProblem{T <: Number, N} <: AbstractKSProblem
	pide::Function
	kernel::Function
	boundary_conditions::Function
	domain::NTuple{N, Tuple{T, T}}
	initial_conditions::Union{AbstractVector{T}, Function}
	tspan::Tuple{T, T}

	function KSPIDEProblem(pide::Function, kernel::Function, boundary_conditions::Function,
						   domain::NTuple{N, Tuple{T, T}}, initial_conditions::Union{AbstractVector{T}, Function},
						   tspan::Tuple{T, T}) where {T <: Number, N}
		# Type checks
		if !isa(pide, Function)
			throw(ArgumentError("PIDE must be a Function, but got $(typeof(pide))"))
		end
		if !isa(kernel, Function)
			throw(ArgumentError("kernel must be a Function, but got $(typeof(kernel))"))
		end
		if !isa(boundary_conditions, Function)
			throw(ArgumentError("boundary_conditions must be a Function, but got $(typeof(boundary_conditions))"))
		end
		if !isa(domain, NTuple{N, Tuple{T, T}})
			throw(ArgumentError("domain must be an NTuple of Tuples, but got $(typeof(domain))"))
		end
		if !(isa(initial_conditions, AbstractVector{T}) || isa(initial_conditions, Function))
			throw(ArgumentError("initial_conditions must be an AbstractVector or Function, but got $(typeof(initial_conditions))"))
		end
		if !isa(tspan, Tuple{T, T})
			throw(ArgumentError("tspan must be a Tuple of $T, but got $(typeof(tspan))"))
		end

		new{T, N}(pide, kernel, boundary_conditions, domain, initial_conditions, tspan)
	end
end

"""
	KSDiscretizedProblem{T <: Number, N, M, MT <: AbstractMatrix} <: AbstractKSProblem

Represents a discretized problem.

# Fields
- `time_nodes::AbstractVector{T}`: The time nodes of the discretization.
- `spatial_nodes::NTuple{N, AbstractVector{T}}`: The spatial nodes of the discretization.
- `system_matrix::MT`: The system matrix of the discretized problem.
- `initial_conditions::AbstractVector{T}`: The initial conditions.
- `problem_functions::NTuple{M, Function}`: The problem functions representing the discretized equations.
"""
mutable struct KSDiscretizedProblem{T <: Number, N, M, MT <: AbstractMatrix} <: AbstractKSProblem
	time_nodes::AbstractVector{T}
	spatial_nodes::NTuple{N, AbstractVector{T}}
	system_matrix::MT
	initial_conditions::AbstractVector{T}
	problem_functions::NTuple{M, Function}

	function KSDiscretizedProblem(time_nodes::AbstractVector{T}, spatial_nodes::NTuple{N, AbstractVector{T}},
								  system_matrix::MT, initial_conditions::AbstractVector{T},
								  problem_functions::NTuple{M, Function}) where {T <: Number, N, M, MT <: AbstractMatrix}
		# Type checks
		if length(time_nodes) == 0
			throw(ArgumentError("Time nodes cannot be empty"))
		end
		if length(spatial_nodes) != N
			throw(ArgumentError("Number of spatial node sets must match N"))
		end
		if !all(isa.(problem_functions, Function))
			throw(ArgumentError("All problem_functions must be callable"))
		end

		new{T, N, M, MT}(time_nodes, spatial_nodes, system_matrix, initial_conditions, problem_functions)
	end
end

"""
	KSOptimalControlProblem{T <: Number} <: AbstractKSOptimizationProblem

Represents an optimal control problem.

# Fields
- `state_equations::AbstractVector{<:Function}`: The state equations of the control problem.
- `cost_functions::AbstractVector{<:Function}`: The cost functions to be minimized.
- `terminal_cost::Function`: The terminal cost function.
- `initial_state::AbstractVector{T}`: The initial state vector.
- `time_span::Tuple{T, T}`: The time span of the control problem.
- `control_bounds::AbstractVector{NTuple{2, T}}`: The bounds on the control variables.
"""
struct KSOptimalControlProblem{T <: Number} <: AbstractKSOptimizationProblem
	state_equations::AbstractVector{<:Function}
	cost_functions::AbstractVector{<:Function}
	terminal_cost::Function
	initial_state::AbstractVector{T}
	time_span::Tuple{T, T}
	control_bounds::AbstractVector{NTuple{2, T}}

	function KSOptimalControlProblem(state_equations::AbstractVector{<:Function},
									 cost_functions::AbstractVector{<:Function},
									 terminal_cost::Function,
									 initial_state::AbstractVector{T},
									 time_span::Tuple{T, T},
									 control_bounds::AbstractVector{NTuple{2, T}}) where {T <: Number}
		# Type checks
		if length(initial_state) != length(state_equations)
			throw(ArgumentError("The number of initial states ($(length(initial_state))) must match the number of state equations ($(length(state_equations)))."))
		end
		if length(state_equations) != length(control_bounds)
			throw(ArgumentError("The number of state equations ($(length(state_equations))) must match the number of control bounds ($(length(control_bounds)))."))
		end
		if length(state_equations) != length(cost_functions)
			throw(ArgumentError("The number of state equations ($(length(state_equations))) must match the number of cost functions ($(length(cost_functions)))."))
		end

		new{T}(state_equations, cost_functions, terminal_cost, initial_state, time_span, control_bounds)
	end
end

"""
KSCartesianCoordinates{T, N} <: AbstractKSCoordinateSystem

Represents a Cartesian coordinate system.

# Fields
- `ranges::NTuple{N, Tuple{T, T}}`: A tuple containing the range for each dimension.
"""
struct KSCartesianCoordinates{T <: Number, N} <: AbstractKSCoordinateSystem
    ranges::NTuple{N, Tuple{T, T}}

    # Existing Constructors
    function KSCartesianCoordinates{N}(ranges::NTuple{N, Tuple{Number, Number}}) where N
        T = promote_type(map(el -> typeof(el[1]), ranges)..., map(el -> typeof(el[2]), ranges)...)
        promoted_ranges = ntuple(i -> (convert(T, ranges[i][1]), convert(T, ranges[i][2])), N)
        return new{T, N}(promoted_ranges)
    end

    function KSCartesianCoordinates(ranges::Tuple{Vararg{Tuple{Number, Number}}})
        N = length(ranges)
        T = promote_type(map(el -> typeof(el[1]), ranges)..., map(el -> typeof(el[2]), ranges)...)
        promoted_ranges = ntuple(i -> (convert(T, ranges[i][1]), convert(T, ranges[i][2])), Val(N))
        return new{T, N}(promoted_ranges)
    end

    function KSCartesianCoordinates(ranges::Vector{Tuple{Number, Number}})
        N = length(ranges)
        T = promote_type(map(el -> typeof(el[1]), ranges)..., map(el -> typeof(el[2]), ranges)...)
        promoted_ranges = ntuple(i -> (convert(T, ranges[i][1]), convert(T, ranges[i][2])), Val(N))
        return new{T, N}(promoted_ranges)
    end

    function KSCartesianCoordinates{T}(ranges::AbstractVector{<:Tuple{T, T}}) where T <: Number
        N = length(ranges)
        return new{T, N}(Tuple(ranges))
    end

    function KSCartesianCoordinates{T, N}(ranges::NTuple{N, Tuple{T, T}}) where {T <: Number, N}
        return new{T, N}(ranges)
    end

    # New Constructor for Tuple{FNumber, Number}
    function KSCartesianCoordinates(range::Tuple{Number, Number})
        T = promote_type(typeof(range[1]), typeof(range[2]))
        return new{T, 1}(NTuple{1, Tuple{T, T}}((convert(T, range[1]), convert(T, range[2]))))
    end
end


"""
KSPolarCoordinates{T} <: AbstractKSCoordinateSystem

Represents a polar coordinate system.

# Fields
- `r::Union{Tuple{T, T}, Nothing}`: The radial distance from the origin.
- `theta::Union{Tuple{T, T}, Nothing}`: The angle in radians from the positive x-axis.
"""
struct KSPolarCoordinates{T <: Number} <: AbstractKSCoordinateSystem
	r::Union{Tuple{T, T}, Nothing}
	theta::Union{Tuple{T, T}, Nothing}

	function KSPolarCoordinates(r::Union{Tuple{Number, Number}, Nothing}, theta::Union{Tuple{Number, Number}, Nothing})
		T = promote_type(r !== nothing ? typeof(r[1]) : Float64,
						 r !== nothing ? typeof(r[2]) : Float64,
						 theta !== nothing ? typeof(theta[1]) : Float64,
						 theta !== nothing ? typeof(theta[2]) : Float64)
		promoted_r = r !== nothing ? (convert(T, r[1]), convert(T, r[2])) : nothing
		promoted_theta = theta !== nothing ? (convert(T, theta[1]), convert(T, theta[2])) : nothing
		new{T}(promoted_r, promoted_theta)
	end
end

"""
KSCylindricalCoordinates{T} <: AbstractKSCoordinateSystem

Represents a cylindrical coordinate system.

# Fields
- `r::Union{Tuple{T, T}, Nothing}`: The radial distance from the z-axis.
- `theta::Union{Tuple{T, T}, Nothing}`: The azimuthal angle in radians from the positive x-axis.
- `z::Union{Tuple{T, T}, Nothing}`: The height above the xy-plane.
"""

struct KSCylindricalCoordinates{T <: Number} <: AbstractKSCoordinateSystem
	r::Union{Tuple{T, T}, Nothing}
	theta::Union{Tuple{T, T}, Nothing}
	z::Union{Tuple{T, T}, Nothing}

	function KSCylindricalCoordinates(r::Union{Tuple{Number, Number}, Nothing}, theta::Union{Tuple{Number, Number}, Nothing}, z::Union{Tuple{Number, Number}, Nothing})
		T = promote_type(r !== nothing ? typeof(r[1]) : Float64,
						 r !== nothing ? typeof(r[2]) : Float64,
						 theta !== nothing ? typeof(theta[1]) : Float64,
						 theta !== nothing ? typeof(theta[2]) : Float64,
						 z !== nothing ? typeof(z[1]) : Float64,
						 z !== nothing ? typeof(z[2]) : Float64)
		promoted_r = r !== nothing ? (convert(T, r[1]), convert(T, r[2])) : nothing
		promoted_theta = theta !== nothing ? (convert(T, theta[1]), convert(T, theta[2])) : nothing
		promoted_z = z !== nothing ? (convert(T, z[1]), convert(T, z[2])) : nothing
		new{T}(promoted_r, promoted_theta, promoted_z)
	end
end

"""
KSSphericalCoordinates{T} <: AbstractKSCoordinateSystem

Represents a spherical coordinate system.

# Fields
- `r::Union{Tuple{T, T}, Nothing}`: The radial distance from the origin.
- `theta::Union{Tuple{T, T}, Nothing}`: The azimuthal angle in radians from the positive x-axis.
- `phi::Union{Tuple{T, T}, Nothing}`: The polar angle in radians from the positive z-axis.
"""

struct KSSphericalCoordinates{T <: Number} <: AbstractKSCoordinateSystem
	r::Union{Tuple{T, T}, Nothing}
	theta::Union{Tuple{T, T}, Nothing}
	phi::Union{Tuple{T, T}, Nothing}

	function KSSphericalCoordinates(r::Union{Tuple{Number, Number}, Nothing}, theta::Union{Tuple{Number, Number}, Nothing}, phi::Union{Tuple{Number, Number}, Nothing})
		T = promote_type(r !== nothing ? typeof(r[1]) : Float64,
						 r !== nothing ? typeof(r[2]) : Float64,
						 theta !== nothing ? typeof(theta[1]) : Float64,
						 theta !== nothing ? typeof(theta[2]) : Float64,
						 phi !== nothing ? typeof(phi[1]) : Float64,
						 phi !== nothing ? typeof(phi[2]) : Float64)
		promoted_r = r !== nothing ? (convert(T, r[1]), convert(T, r[2])) : nothing
		promoted_theta = theta !== nothing ? (convert(T, theta[1]), convert(T, theta[2])) : nothing
		promoted_phi = phi !== nothing ? (convert(T, phi[1]), convert(T, phi[2])) : nothing
		new{T}(promoted_r, promoted_theta, promoted_phi)
	end
end
# Helper function to validate ranges
function validate_range(range::Union{Tuple{T, T}, Nothing}, name::String, min_val::T, max_val::Union{T, Nothing}) where T <: Number
	if range !== nothing
		if length(range) != 2
			throw(ArgumentError("Invalid $name range $(range): must be a tuple of length 2."))
		end
		if range[1] > range[2]
			throw(ArgumentError("Invalid $name range $(range): lower bound must be <= upper bound."))
		end
		if range[1] < min_val || (max_val !== nothing && range[2] > max_val)
			throw(ArgumentError("Invalid $name range $(range): must be within [$min_val, $max_val]."))
		end
	end
end

"""
	KSDirichletBC <: AbstractKSBoundaryCondition

Represents a Dirichlet boundary condition.

# Fields
- `value::Function`: The function specifying the boundary value.
- `boundary::Function`: The function specifying the boundary region.
"""
struct KSDirichletBC <: AbstractKSBoundaryCondition
	value::Function
	boundary::Function

	function KSDirichletBC(value::Function, boundary::Function)
		if !isa(value, Function)
			throw(ArgumentError("value must be a Function, but got $(typeof(value))"))
		end
		if !isa(boundary, Function)
			throw(ArgumentError("boundary must be a Function, but got $(typeof(boundary))"))
		end
		new(value, boundary)
	end
end

"""
	KSRobinBC <: AbstractKSBoundaryCondition

Represents a Robin boundary condition.

# Fields
- `alpha::Function`: The alpha coefficient function.
- `beta::Function`: The beta coefficient function.
- `value::Function`: The function specifying the boundary value.
- `boundary::Function`: The function specifying the boundary region.
"""
struct KSRobinBC <: AbstractKSBoundaryCondition
	alpha::Function
	beta::Function
	value::Function
	boundary::Function

	function KSRobinBC(alpha::Function, beta::Function, value::Function, boundary::Function)
		if !all(isa.([alpha, beta, value, boundary], Function))
			throw(ArgumentError("All arguments must be callable"))
		end
		new(alpha, beta, value, boundary)
	end
end

"""
	KSNeumannBC <: AbstractKSBoundaryCondition

Represents a Neumann boundary condition.

# Fields
- `flux::Function`: The function specifying the flux at the boundary.
- `boundary::Function`: The function specifying the boundary region.
"""
struct KSNeumannBC <: AbstractKSBoundaryCondition
	flux::Function
	boundary::Function

	function KSNeumannBC(flux::Function, boundary::Function)
		if !isa(flux, Function)
			throw(ArgumentError("flux must be a Function, but got $(typeof(flux))"))
		end
		if !isa(boundary, Function)
			throw(ArgumentError("boundary must be a Function, but got $(typeof(boundary))"))
		end
		new(flux, boundary)
	end
end

"""
	KSSolverOptions{T<:Number}

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
- `legendre_threshold::T`: Threshold for Legendre polynomial decay rate to trigger refinement.
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

	function KSSolverOptions(max_iterations::Int, tolerance::T, adaptive::Bool, max_levels::Int,
							 smoothness_threshold::T, initial_elements::Int, initial_degree::Int,
							 use_domain_decomposition::Bool, num_subdomains::Int,
							 use_strang_splitting::Bool, dt::T, num_steps::Int,
							 legendre_threshold::T) where {T <: Number}
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
		new{T}(max_iterations, tolerance, adaptive, max_levels, smoothness_threshold,
			   initial_elements, initial_degree, use_domain_decomposition, num_subdomains,
			   use_strang_splitting, dt, num_steps, legendre_threshold)
	end
end

"""
	KSDirectSolver <: AbstractKSLinearSolver

Represents a direct linear solver.

# Fields
- `method::Symbol`: The method used for the direct solver.
"""
struct KSDirectSolver <: AbstractKSLinearSolver
	method::Symbol

	function KSDirectSolver(method::Symbol)
		new(method)
	end
end

"""
	KSIterativeSolver{T<:Number} <: AbstractKSLinearSolver

Represents an iterative linear solver.

# Fields
- `method::Symbol`: The method used for the iterative solver.
- `max_iter::Int`: The maximum number of iterations.
- `tolerance::T`: The tolerance for convergence.
- `preconditioner::Union{Nothing, Function}`: The preconditioner function, if any.
"""
struct KSIterativeSolver{T <: Number} <: AbstractKSLinearSolver
	method::Symbol
	max_iter::Int
	tolerance::T
	preconditioner::Union{Nothing, Function}

	function KSIterativeSolver(method::Symbol, max_iter::Int, tolerance::T, preconditioner::Union{Nothing, Function} = nothing) where {T <: Number}
		if max_iter <= 0
			throw(ArgumentError("Maximum iterations must be positive"))
		end
		if tolerance <= 0
			throw(ArgumentError("Tolerance must be positive"))
		end
		if !isnothing(preconditioner) && !isa(preconditioner, Function)
			throw(ArgumentError("Preconditioner must be a Function or Nothing, but got $(typeof(preconditioner))"))
		end
		new{T}(method, max_iter, tolerance, preconditioner)
	end
end

"""
	KSAMGSolver{T<:Number} <: AbstractKSLinearSolver

Represents an algebraic multigrid solver.

# Fields
- `max_iter::Int`: The maximum number of iterations.
- `tolerance::T`: The tolerance for convergence.
- `smoother::Symbol`: The smoother method used in the AMG solver.
"""
struct KSAMGSolver{T <: Number} <: AbstractKSLinearSolver
	max_iter::Int
	tolerance::T
	smoother::Symbol

	function KSAMGSolver(max_iter::Int, tolerance::T, smoother::Symbol) where {T <: Number}
		if max_iter <= 0
			throw(ArgumentError("Maximum iterations must be positive"))
		end
		if tolerance <= 0
			throw(ArgumentError("Tolerance must be positive"))
		end
		new{T}(max_iter, tolerance, smoother)
	end
end

# Helper function to filter active dimensions
function tuple_if_active(dims...)
	return Tuple(filter(!isnothing, dims))
end

# Check if a function is callable
is_callable(f) = Base.isa(f, Function) || Base.isa(typeof(f), Function)

end  # module KSTypes
