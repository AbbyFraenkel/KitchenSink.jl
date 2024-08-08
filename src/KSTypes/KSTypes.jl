module KSTypes

using LinearAlgebra

"""
# KSTypes

Defines various types used in the KitchenSink package, including abstract and concrete types for systems, problems, solvers, meshes, elements, basis functions, boundary conditions, coordinate systems, and optimization.

"""

# Export all types
export AbstractKSSystem, AbstractKSProblem, AbstractKSSolver, AbstractKSMesh, AbstractKSElement
export AbstractKSBasisFunction, AbstractKSBoundaryCondition, AbstractKSCoordinateSystem
export AbstractKSPDEProblem, AbstractKSBVDAEProblem, AbstractKSIDEProblem, AbstractKSPoint
export AbstractKSLinearSolver, AbstractKSOptimizer, AbstractKSODEProblem, AbstractKSDAEProblem
export AbstractKSTimeSteppingMethod, AbstractKSOptimizationProblem, AbstractKSDiscretization
export KSPoint, KSBasisFunction, KSElement, KSMesh, KSProblem, KSSolverOptions, KSSolver
export KSCartesianCoordinates, KSPolarCoordinates, KSSphericalCoordinates, KSCylindricalCoordinates
export KSDirichletBC, KSNeumannBC, KSRobinBC, KSBVDAEProblem, KSCoupledProblem, KSIDEProblem
export KSODEProblem, KSPDEProblem, KSDAEProblem, KSMovingBoundaryPDEProblem, KSPIDEProblem
export KSDiscretizedProblem, KSTimeSteppingSolver, KSDirectSolver, KSIterativeSolver, KSAMGSolver
export KSGradientDescentOptimizer, KSNewtonOptimizer, KSLBFGSOptimizer, KSOptimalControlProblem
export KSMultiProblem

# Enums
@enum CoordinateSystem Cartesian Polar Spherical Cylindrical

# Abstract types

"""
    AbstractKSSystem

Abstract type for Kitchen Sink systems definitions, typically including multiple problems.
"""
abstract type AbstractKSSystem end

"""
    AbstractKSProblem{T, N}

Abstract type for Kitchen Sink problem definitions.
"""
abstract type AbstractKSProblem{T,N} <: AbstractKSSystem end

"""
    AbstractKSODEProblem{T, N}

Abstract type for ODE problems in the Kitchen Sink framework.
"""
abstract type AbstractKSODEProblem{T,N} <: AbstractKSProblem{T,N} end

"""
    AbstractKSPDEProblem{T, N}

Abstract type for PDE problems in the Kitchen Sink framework.
"""
abstract type AbstractKSPDEProblem{T,N} <: AbstractKSProblem{T,N} end

"""
    AbstractKSDAEProblem{T, N}

Abstract type for DAE problems in the Kitchen Sink framework.
"""
abstract type AbstractKSDAEProblem{T,N} <: AbstractKSProblem{T,N} end

"""
    AbstractKSBVDAEProblem{T, N}

Abstract type for Boundary Value Differential-Algebraic Equation problems.
"""
abstract type AbstractKSBVDAEProblem{T,N} <: AbstractKSDAEProblem{T,N} end

"""
    AbstractKSIDEProblem{T, N}

Abstract type for Integral-Differential Equation problems.
"""
abstract type AbstractKSIDEProblem{T,N} <: AbstractKSProblem{T,N} end

"""
    AbstractKSSolver{T}

Abstract type for Kitchen Sink solvers.
"""
abstract type AbstractKSSolver{T} end

"""
    AbstractKSMesh{T, N}

Abstract type for Kitchen Sink mesh representations.
"""
abstract type AbstractKSMesh{T,N} end

"""
    AbstractKSElement{T}

Abstract type for Kitchen Sink mesh elements.
"""
abstract type AbstractKSElement{T} <: AbstractKSMesh{T,1} end

"""
    AbstractKSBasisFunction

Abstract type for Kitchen Sink basis functions.
"""
abstract type AbstractKSBasisFunction end

"""
    AbstractKSBoundaryCondition{T}

Abstract type for Kitchen Sink boundary conditions.
"""
abstract type AbstractKSBoundaryCondition{T} end

"""
    AbstractKSCoordinateSystem{N}

Abstract type for Kitchen Sink coordinate systems.
"""
abstract type AbstractKSCoordinateSystem{N} end

"""
    AbstractKSLinearSolver{T}

Abstract type for linear solvers in the Kitchen Sink framework.
"""
abstract type AbstractKSLinearSolver{T} <: AbstractKSSolver{T} end

"""
    AbstractKSOptimizer{T}

Abstract base type for optimizers.
"""
abstract type AbstractKSOptimizer{T} <: AbstractKSSolver{T} end

"""
    AbstractKSPoint{T}

Abstract type for points in a grid.
"""
abstract type AbstractKSPoint{T} <: AbstractKSMesh{T,1} end

"""
    AbstractKSOptimizationProblem{T, N}

Abstract type for optimization problems.
"""
abstract type AbstractKSOptimizationProblem{T,N} <: AbstractKSProblem{T,N} end

"""
   AbstractKSDiscretization{T}

Abstract type for discretization.
"""
abstract type AbstractKSDiscretization{T} end

"""
   AbstractKSTimeSteppingMethod{T}

Abstract type for time stepping methods.
"""
abstract type AbstractKSTimeSteppingMethod{T} end

# Concrete types

"""
    KSPoint{T<:Real}

Represents a point in the Kitchen Sink framework, which could be a collocation point or a boundary point.

# Fields
- `coordinates::Vector{T}`: Coordinates of the point
- `weight::Union{Nothing, T}`: Optional weight for the point (used for collocation points)
"""
mutable struct KSPoint{T<:Real} <: AbstractKSPoint{T}
    coordinates::Vector{T}
    weight::Union{Nothing,T}
end

"""
    KSBasisFunction

Represents a basis function in the Kitchen Sink framework.

# Fields
- `id::Int`: Unique identifier for the basis function
- `function_handle::Function`: The actual basis function
- `is_removable::Bool`: Flag indicating if the basis function can be removed during adaptivity
- `degree::Int`: Polynomial degree of the basis function
"""
mutable struct KSBasisFunction <: AbstractKSBasisFunction
    id::Int
    function_handle::Function
    is_removable::Bool
    degree::Int
end

"""
    KSElement{T<:Real}

Represents an element in the Kitchen Sink mesh.

# Fields
- `id::Int`: Unique identifier for the element
- `points::Vector{KSPoint{T}}`: Points in the element (collocation and boundary points)
- `basis_functions::Vector{KSBasisFunction}`: Basis functions in the element
- `neighbors::Union{Nothing, Vector{KSElement{T}}}`: Neighboring elements
- `parent::Union{Nothing, KSElement{T}}`: Parent element
- `children::Union{Nothing, Vector{KSElement{T}}}`
- `level::Int`: Refinement level of the element
- `polynomial_degree::Int`: Polynomial degree of the element
- `error_estimate::T`: Local error estimate for this element
- `differentiation_matrices::Vector{Matrix{T}}`: Differentiation matrices for each basis function
"""
mutable struct KSElement{T<:Real} <: AbstractKSElement{T}
    id::Int
    points::Vector{KSPoint{T}}
    basis_functions::Vector{KSBasisFunction}
    neighbors::Union{Nothing,Vector{KSElement{T}}}
    parent::Union{Nothing,KSElement{T}}
    children::Union{Nothing,Vector{KSElement{T}}}
    level::Int
    polynomial_degree::Int
    error_estimate::T
    differentiation_matrices::Vector{Matrix{T}}
end
"""
    KSMesh{T<:Real, N}

Represents the entire mesh structure in the Kitchen Sink framework.

# Fields
- `elements::Vector{KSElement{T}}`: All elements in the mesh
- `tensor_product_masks::AbstractArray{Bool}`: Tensor product masks for each element
- `location_matrices::Vector{Dict{Int, Int}}`: Location matrices for each element
- `basis_functions::Vector{KSBasisFunction}`: All basis functions used in the mesh
- `global_error_estimate::Real`: Global error estimate for the entire mesh
- `dimensions::Int`: Number of dimensions in the mesh
"""
mutable struct KSMesh{T<:Real,N} <: AbstractKSMesh{T,N}
    elements::Vector{KSElement{T}}
    tensor_product_masks::AbstractArray{Bool}
    location_matrices::Vector{Dict{Int,Int}}
    basis_functions::Vector{KSBasisFunction}
    global_error_estimate::Real
    dimensions::Int
end

"""
    KSProblem{T<:Real, N, C<:AbstractKSCoordinateSystem{N}}

Represents a problem to be solved using the Kitchen Sink framework.

# Fields
- `equation::Function`: The differential operator of the problem
- `domain::NTuple{N, Tuple{T, T}}`: The domain of the problem
- `boundary_conditions::Function`: The boundary conditions of the problem
- `coordinate_system::C`: The coordinate system of the problem
"""
mutable struct KSProblem{T<:Real,N,C<:AbstractKSCoordinateSystem{N}} <: AbstractKSProblem{T,N}
    equation::Function
    domain::NTuple{N,Tuple{T,T}}
    boundary_conditions::Function
    coordinate_system::C
end

"""
    KSCartesianCoordinates{N, T<:Real}

A struct representing Cartesian coordinates in N-dimensional space.

# Fields
- `coordinates::NTuple{N, Union{T, Nothing}}`: Coordinates in the N-dimensional Cartesian system
"""
mutable struct KSCartesianCoordinates{N,T<:Real} <: AbstractKSCoordinateSystem{N}
    coordinates::NTuple{N,Union{T,Nothing}}
end

"""
    KSPolarCoordinates{T<:Real}

A struct representing polar coordinates in 2-dimensional space.

# Fields
- `r::Union{T, Nothing}`: Radial coordinate
- `θ::Union{T, Nothing}`: Angular coordinate
"""
mutable struct KSPolarCoordinates{T<:Real} <: AbstractKSCoordinateSystem{2}
    r::Union{T,Nothing}
    θ::Union{T,Nothing}
end

"""
    KSSphericalCoordinates{T<:Real}

A struct representing spherical coordinates in 3-dimensional space.

# Fields
- `r::Union{T, Nothing}`: Radial coordinate
- `θ::Union{T, Nothing}`: Polar angle
- `φ::Union{T, Nothing}`: Azimuthal angle
"""
mutable struct KSSphericalCoordinates{T<:Real} <: AbstractKSCoordinateSystem{3}
    r::Union{T,Nothing}
    θ::Union{T,Nothing}
    φ::Union{T,Nothing}
end

"""
    KSCylindricalCoordinates{T<:Real}

A struct representing cylindrical coordinates in 3-dimensional space.

# Fields
- `r::Union{T, Nothing}`: Radial coordinate
- `θ::Union{T, Nothing}`: Angular coordinate
- `z::Union{T, Nothing}`: Axial coordinate
"""
mutable struct KSCylindricalCoordinates{T<:Real} <: AbstractKSCoordinateSystem{3}
    r::Union{T,Nothing}
    θ::Union{T,Nothing}
    z::Union{T,Nothing}
end

"""
    KSDirichletBC{T<:Real}

Represents a Dirichlet boundary condition.

# Fields
- `value::Function`: Function defining the boundary value
- `boundary::Function`: Function defining the boundary region
"""
mutable struct KSDirichletBC{T<:Real} <: AbstractKSBoundaryCondition{T}
    value::Function
    boundary::Function
end

"""
    KSNeumannBC{T<:Real}

Represents a Neumann boundary condition.

# Fields
- `flux::Function`: Function defining the boundary flux
- `boundary::Function`: Function defining the boundary region
"""
mutable struct KSNeumannBC{T<:Real} <: AbstractKSBoundaryCondition{T}
    flux::Function
    boundary::Function
end

"""
    KSRobinBC{T<:Real}

Represents a Robin boundary condition.

# Fields
- `a::Function`: Function defining the a coefficient
- `b::Function`: Function defining the b coefficient
- `c::Function`: Function defining the c coefficient
- `boundary::Function`: Function defining the boundary region
"""
mutable struct KSRobinBC{T<:Real} <: AbstractKSBoundaryCondition{T}
    a::Function
    b::Function
    c::Function
    boundary::Function
end

"""
    KSSolverOptions{T<:Real}

Options for the Kitchen Sink solver.

# Fields
- `max_iterations::Int`: Maximum number of iterations
- `tolerance::Real`: Convergence tolerance
- `adaptive::Bool`: Whether to use adaptive refinement
- `max_levels::Int`: Maximum number of levels in the hierarchy
- `smoothness_threshold::T`: Threshold for smoothness in adaptive refinement
"""
mutable struct KSSolverOptions{T<:Real} <: AbstractKSSolver{T}
    max_iterations::Int
    tolerance::Real
    adaptive::Bool
    max_levels::Int
    smoothness_threshold::T
    initial_elements::Int
    initial_degree::Int
end

"""
    KSSolver{T<:Real}

The Kitchen Sink solver.

# Fields
- `options::KSSolverOptions{T}`: Solver options
- `current_solution::Vector{T}`: Current solution vector
- `error_estimates::Vector{T}`: Error estimates for each element
"""
mutable struct KSSolver{T<:Real} <: AbstractKSSolver{T}
    options::KSSolverOptions{T}
    current_solution::Vector{T}
    error_estimates::Vector{T}
end

"""
    KSOptimalControlProblem{T<:Real}

Represents an optimal control problem.

# Fields
- `state_equation::Function`: The state equation function
- `cost_function::Function`: The cost function
- `terminal_cost::Function`: The terminal cost function
- `initial_state::Vector{T}`: The initial state vector
- `time_span::Tuple{T, T}`: The time span of the problem
- `control_bounds::Vector{Tuple{T, T}}`: The bounds on the control variables
"""
mutable struct KSOptimalControlProblem{T<:Real} <: AbstractKSOptimizationProblem{T,1}
    state_equation::Function
    cost_function::Function
    terminal_cost::Function
    initial_state::Vector{T}
    time_span::Tuple{T,T}
    control_bounds::Vector{Tuple{T,T}}
end

"""
    KSGradientDescentOptimizer

Represents a gradient descent optimizer.

# Fields
- `learning_rate::Float64`: Learning rate for the optimizer
- `max_iterations::Int`: Maximum number of iterations
- `tolerance::Float64`: Convergence tolerance
"""
struct KSGradientDescentOptimizer <: AbstractKSOptimizer{Float64}
    learning_rate::Float64
    max_iterations::Int
    tolerance::Float64
end

"""
    KSNewtonOptimizer

Represents a Newton optimizer.

# Fields
- `max_iterations::Int`: Maximum number of iterations
- `tolerance::Float64`: Convergence tolerance
"""
struct KSNewtonOptimizer <: AbstractKSOptimizer{Float64}
    max_iterations::Int
    tolerance::Float64
end

"""
    KSLBFGSOptimizer

Represents an LBFGS optimizer.

# Fields
- `m::Int`: Number of corrections to store
- `max_iterations::Int`: Maximum number of iterations
- `tolerance::Float64`: Convergence tolerance
"""
struct KSLBFGSOptimizer <: AbstractKSOptimizer{Float64}
    m::Int  # Number of corrections to store
    max_iterations::Int
    tolerance::Float64
end

"""
    KSBVDAEProblem{T<:Real}

Represents a Boundary Value Differential-Algebraic Equation problem.

# Fields
- `f::Function`: Differential equations
- `g::Function`: Algebraic equations
- `bc::Function`: Boundary conditions
- `tspan::Tuple{T, T}`: Time span
- `y0::Vector{T}`: Initial values
- `algebraic_vars::Vector{Bool}`: Flags indicating algebraic variables
"""
mutable struct KSBVDAEProblem{T<:Real} <: AbstractKSBVDAEProblem{T,1}
    f::Function
    g::Function
    bc::Function
    tspan::Tuple{T,T}
    y0::Vector{T}
    algebraic_vars::Vector{Bool}
    initial_conditions::Vector{T}
end

"""
    KSCoupledProblem{T<:Real}

Represents a coupled problem in the Kitchen Sink framework.

# Fields
- `problems::Vector{AbstractKSProblem{T, 1}}`: List of problems to be coupled
- `coupling_terms::Matrix{Union{Function, Nothing}}`: Coupling terms between subproblems
"""
mutable struct KSCoupledProblem{T<:Real} <: AbstractKSProblem{T,1}
    problems::Vector{AbstractKSProblem{T,1}}
    coupling_terms::Matrix{Union{Function,Nothing}}
end

"""
    KSIDEProblem{T<:Real}

Represents an Integral-Differential Equation problem.

# Fields
- `f::Function`: IDE function dy/dt = f(t, y, integral)
- `K::Function`: Kernel function for the integral
- `boundary_conditions::Function`: Boundary conditions function
- `tspan::Tuple{T, T}`: Time span
- `initial_conditions::Vector{T}`: Initial values
"""
mutable struct KSIDEProblem{T<:Real} <: AbstractKSIDEProblem{T,1}
    f::Function
    K::Function
    boundary_conditions::Function
    tspan::Tuple{T,T}
    initial_conditions::Vector{T}
end

"""
    KSODEProblem{T<:Real}

Represents an ODE problem in the Kitchen Sink framework.

# Fields
- `ode::Function`: The ODE function
- `tspan::Tuple{T, T}`: Time span
- `initial_conditions::Vector{T}`: Initial conditions
"""
mutable struct KSODEProblem{T<:Real} <: AbstractKSODEProblem{T,1}
    ode::Function
    tspan::Tuple{T,T}
    initial_conditions::Vector{T}
end

"""
    KSPDEProblem{T<:Real}

Represents a PDE problem in the Kitchen Sink framework.

# Fields
- `pde::Function`: The PDE function
- `boundary_conditions::Function`: The boundary conditions function
- `domain::NTuple{2, Tuple{T, T}}`: The spatial domain
- `tspan::Tuple{T, T}`: Time span
- `initial_conditions::Union{Vector{T}, Function}`: Initial conditions
"""
mutable struct KSPDEProblem{T<:Real} <: AbstractKSPDEProblem{T,2}
    pde::Function
    boundary_conditions::Function
    tspan::Tuple{T,T}
    domain::NTuple{2,Tuple{T,T}}
    initial_conditions::Union{Vector{T},Function}
end

"""
    KSDAEProblem{T<:Real}

Represents a DAE problem in the Kitchen Sink framework.

# Fields
- `dae::Function`: The DAE function
- `tspan::Tuple{T, T}`: Time span
- `initial_conditions::Vector{T}`: Initial conditions
"""
mutable struct KSDAEProblem{T<:Real} <: AbstractKSDAEProblem{T,1}
    dae::Function
    tspan::Tuple{T,T}
    initial_conditions::Vector{T}
end

"""
    KSMovingBoundaryPDEProblem{T<:Real}

Represents a moving boundary PDE problem in the Kitchen Sink framework.

# Fields
- `pde::Function`: The PDE function
- `boundary_conditions::Function`: The boundary conditions function
- `domain::NTuple{2, Tuple{T, T}}`: The spatial domain
- `boundary_motion::Function`: The boundary motion function
- `tspan::Tuple{T, T}`: Time span
- `initial_conditions::Vector{T}`: Initial conditions
"""
mutable struct KSMovingBoundaryPDEProblem{T<:Real} <: AbstractKSPDEProblem{T,2}
    pde::Function
    boundary_conditions::Function
    tspan::Tuple{T,T}
    domain::NTuple{2,Tuple{T,T}}
    boundary_motion::Function
    initial_conditions::Vector{T}
end

"""
    KSPIDEProblem{T<:Real}

Represents a PIDE problem in the Kitchen Sink framework.

# Fields
- `pide::Function`: The PIDE function
- `K::Function`: The kernel function
- `boundary_conditions::Function`: The boundary conditions function
- `domain::NTuple{2, Tuple{T, T}}`: The spatial domain
- `tspan::Tuple{T, T}`: Time span
- `initial_conditions::Union{Vector{T}, Function}`: Initial conditions
"""
mutable struct KSPIDEProblem{T<:Real} <: AbstractKSPDEProblem{T,2}
    pide::Function
    K::Function
    boundary_conditions::Function
    tspan::Tuple{T,T}
    domain::NTuple{2,Tuple{T,T}}
    initial_conditions::Union{Vector{T},Function}
end

"""
    KSDiscretizedProblem{T<:Real}

Represents a discretized problem in the Kitchen Sink framework.

# Fields
- `time_nodes::Vector{T}`: Time nodes
- `spatial_nodes::Vector{Vector{T}}`: Spatial nodes
- `system_matrix::Matrix{T}`: System matrix
- `initial_conditions::Vector{T}`: Initial conditions
- `problem_functions::Vector{Function}`: Problem functions
"""
mutable struct KSDiscretizedProblem{T<:Real}
    time_nodes::Vector{T}
    spatial_nodes::Vector{Vector{T}}
    system_matrix::Matrix{T}
    initial_conditions::Vector{T}
    problem_functions::Vector{Function}
end

"""
    KSTimeSteppingSolver{T<:Real}

Represents a time-stepping solver configuration.

# Fields
- `method::Symbol`: The time-stepping method to use (:euler, :rk4, etc.)
- `dt::T`: Time step size
- `t_final::T`: Final time
- `tolerance::T`: Convergence tolerance
"""
mutable struct KSTimeSteppingSolver{T<:Real} <: AbstractKSSolver{T}
    method::Symbol
    dt::T
    t_final::T
    tolerance::T
end

"""
    KSDirectSolver{T<:Real}

Represents a direct solver configuration.

# Fields
- `method::Symbol`: The direct method to use (:lu, :qr, etc.)
- `tolerance::T`: Convergence tolerance
"""
mutable struct KSDirectSolver{T<:Real} <: AbstractKSLinearSolver{T}
    method::Symbol
    tolerance::T
end

"""
    KSIterativeSolver{T<:Real}

Represents an iterative solver configuration.

# Fields
- `method::Symbol`: The iterative method to use (:cg, :gmres, :bicgstab, etc.)
- `max_iter::Int`: Maximum number of iterations
- `tolerance::T`: Convergence tolerance
- `preconditioner::Union{Nothing, AbstractMatrix{T}}`: Preconditioner function, if any
"""
mutable struct KSIterativeSolver{T<:Real} <: AbstractKSLinearSolver{T}
    method::Symbol
    max_iter::Int
    tolerance::T
    preconditioner::Union{Nothing,AbstractMatrix{T}}
end

"""
    KSAMGSolver{T<:Real}

Represents an AMG solver configuration.

# Fields
- `max_iter::Int`: Maximum number of iterations
- `tolerance::T`: Convergence tolerance
- `smoother::Symbol`: The smoother to use (:jacobi, :gauss_seidel)
"""
mutable struct KSAMGSolver{T<:Real} <: AbstractKSLinearSolver{T}
    max_iter::Int
    tolerance::T
    smoother::Symbol
end

"""
    KSMultiProblem{T<:Real}

Represents a multi-problem in the Kitchen Sink framework.

# Fields
- `subproblems::Vector{AbstractKSProblem{T, 1}}`: Subproblems in the multi-problem
- `coupling_terms::Matrix{Union{Function, Nothing}}`: Coupling terms between subproblems
"""
mutable struct KSMultiProblem{T<:Real} <: AbstractKSProblem{T,1}
    subproblems::Vector{AbstractKSProblem{T,1}}
    coupling_terms::Matrix{Union{Function,Nothing}}
end

end # module KSTypes
