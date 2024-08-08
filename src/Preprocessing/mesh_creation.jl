"""
    create_mesh(domain::NTuple{N, Tuple{T, T}}, coord_system::AbstractKSCoordinateSystem{N},
                num_elements::NTuple{N, Int}, polynomial_degree::Int) where {N, T<:Real}

Create a mesh for the specified domain, coordinate system, number of elements, and polynomial degree.

# Arguments
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain.
- `coord_system::AbstractKSCoordinateSystem{N}`: The coordinate system.
- `num_elements::NTuple{N, Int}`: Number of elements in each dimension.
- `polynomial_degree::Int`: Polynomial degree for the basis functions.

# Returns
- `KSMesh{T, N}`: The created mesh.

# Throws
- `ArgumentError`: If any input argument is invalid.

# Example
```julia
domain = ((0.0, 1.0), (0.0, 1.0))
coord_system = KSCartesianCoordinates{2, Float64}((0.0, 0.0))
num_elements = (4, 4)
polynomial_degree = 2
mesh = create_mesh(domain, coord_system, num_elements, polynomial_degree)
```

See also: [`create_nodes`](@ref), [`create_elements`](@ref), [`KSMesh`](@ref)
"""
function create_mesh(domain::NTuple{N,Tuple{T,T}}, coord_system::AbstractKSCoordinateSystem{N},
    num_elements::NTuple{N,Int}, polynomial_degree::Int) where {N,T<:Real}
    if any(ne -> ne <= 0, num_elements)
        throw(ArgumentError("Number of elements must be positive in all dimensions"))
    end
    if polynomial_degree <= 0
        throw(ArgumentError("Polynomial degree must be positive"))
    end
    if N != length(domain) || N != length(num_elements)
        throw(ArgumentError("Dimension mismatch between domain, coordinate system, and number of elements"))
    end

    try
        nodes = create_nodes(domain, coord_system, num_elements)
        elements = create_elements(nodes, num_elements, polynomial_degree, coord_system)

        return KSMesh{T,N}(
            elements,
            [falses(polynomial_degree + 1) for _ in 1:length(elements)],
            [Dict{Int,Int}() for _ in 1:length(elements)],
            Vector{KSBasisFunction}(),
            zero(T),
            N
        )
    catch e
        throw(ErrorException("Failed to create mesh: $(e.msg)"))
    end
end

"""
    create_nodes(domain::NTuple{N, Tuple{T, T}}, coord_system::AbstractKSCoordinateSystem{N},
                 num_elements::NTuple{N, Int}) where {N, T<:Real}

Create nodes for the specified domain, coordinate system, and number of elements.

# Arguments
- `domain::NTuple{N, Tuple{T, T}}`: The spatial domain.
- `coord_system::AbstractKSCoordinateSystem{N}`: The coordinate system.
- `num_elements::NTuple{N, Int}`: Number of elements in each dimension.

# Returns
- `Vector{KSPoint{T}}`: The created nodes.

# Throws
- `ArgumentError`: If any input argument is invalid.

# Example
```julia
domain = ((0.0, 1.0), (0.0, 1.0))
coord_system = KSCartesianCoordinates{2, Float64}((0.0, 0.0))
num_elements = (4, 4)
nodes = create_nodes(domain, coord_system, num_elements)
```

See also: [`KSPoint`](@ref), [`SpectralMethods.create_nodes`](@ref)
"""
function create_nodes(domain::NTuple{N,Tuple{T,T}}, coord_system::AbstractKSCoordinateSystem{N},
    num_elements::NTuple{N,Int}) where {N,T<:Real}
    if any(ne -> ne <= 0, num_elements)
        throw(ArgumentError("Number of elements must be positive in all dimensions"))
    end
    if N != length(domain) || N != length(num_elements)
        throw(ArgumentError("Dimension mismatch between domain, coordinate system, and number of elements"))
    end

    nodes = Vector{KSPoint{T}}()
    node_id = 1

    try
        # Use SpectralMethods.create_nodes for each dimension
        dimension_nodes = [SpectralMethods.create_nodes(num_elements[d] + 1, domain[d]...)[1] for d in 1:N]

        for idx in CartesianIndices(num_elements .+ 1)
            coords = ntuple(d -> dimension_nodes[d][idx[d]], N)
            push!(nodes, KSPoint(node_id, CoordinateSystems.to_cartesian(coords, coord_system)))
            node_id += 1
        end

        return nodes
    catch e
        throw(ErrorException("Failed to create nodes: $(e.msg)"))
    end
end

"""
    create_elements(nodes::Vector{KSPoint{T}}, num_elements::NTuple{N, Int}, polynomial_degree::Int,
                    coord_system::AbstractKSCoordinateSystem{N}) where {N, T<:Real}

Create elements for the specified nodes, number of elements, polynomial degree, and coordinate system.

# Arguments
- `nodes::Vector{KSPoint{T}}`: The created nodes.
- `num_elements::NTuple{N, Int}`: Number of elements in each dimension.
- `polynomial_degree::Int`: Polynomial degree for the basis functions.
- `coord_system::AbstractKSCoordinateSystem{N}`: The coordinate system.

# Returns
- `Vector{KSElement{T, N}}`: The created elements.

# Throws
- `ArgumentError`: If any input argument is invalid.

# Example
```julia
domain = ((0.0, 1.0), (0.0, 1.0))
coord_system = KSCartesianCoordinates{2, Float64}((0.0, 0.0))
num_elements = (4, 4)
nodes = create_nodes(domain, coord_system, num_elements)
polynomial_degree = 2
elements = create_elements(nodes, num_elements, polynomial_degree, coord_system)
```

See also: [`KSElement`](@ref), [`create_collocation_points`](@ref), [`SpectralMethods.derivative_matrix`](@ref)
"""
function create_elements(nodes::Vector{KSPoint{T}}, num_elements::NTuple{N,Int}, polynomial_degree::Int,
    coord_system::AbstractKSCoordinateSystem{N}) where {N,T<:Real}
    if isempty(nodes)
        throw(ArgumentError("Nodes vector cannot be empty"))
    end
    if polynomial_degree <= 0
        throw(ArgumentError("Polynomial degree must be positive"))
    end
    if N != length(num_elements)
        throw(ArgumentError("Dimension mismatch between nodes and number of elements"))
    end

    elements = Vector{KSElement{T,N}}()
    element_id = 1

    try
        for idx in CartesianIndices(num_elements)
            element_nodes = [nodes[LinearIndices(num_elements .+ 1)[idx+offset]] for offset in CartesianIndices(ntuple(_ -> 0:1, N))]
            collocation_points = create_collocation_points(element_nodes, polynomial_degree, coord_system)

            # Create differentiation matrices using SpectralMethods
            diff_matrices = [SpectralMethods.derivative_matrix(getindex.(collocation_points.points, d)) for d in 1:N]

            push!(elements, KSElement{T,N}(
                element_id,
                element_nodes,
                Vector{KSBasisFunction}(),
                Vector{Union{Nothing,KSElement{T,N}}}(nothing, 2^N),
                nothing,
                Vector{KSElement{T,N}}(),
                0,
                polynomial_degree,
                zero(T),
                collocation_points,
                diff_matrices
            ))
            element_id += 1
        end

        return elements
    catch e
        throw(ErrorException("Failed to create elements: $(e.msg)"))
    end
end

"""
    create_collocation_points(nodes::Vector{KSPoint{T}}, polynomial_degree::Int,
                              coord_system::AbstractKSCoordinateSystem{N}) where {N, T<:Real}

Create collocation points for the specified nodes, polynomial degree, and coordinate system.

# Arguments
- `nodes::Vector{KSPoint{T}}`: The nodes of the element.
- `polynomial_degree::Int`: Polynomial degree for the basis functions.
- `coord_system::AbstractKSCoordinateSystem{N}`: The coordinate system.

# Returns
- `KSPoints{T}`: The created collocation points.

# Throws
- `ArgumentError`: If any input argument is invalid.

# Example
```julia
domain = ((0.0, 1.0), (0.0, 1.0))
coord_system = KSCartesianCoordinates{2, Float64}((0.0, 0.0))
num_elements = (4, 4)
nodes = create_nodes(domain, coord_system, num_elements)
polynomial_degree = 2
collocation_points = create_collocation_points(nodes[1:4], polynomial_degree, coord_system)
```

See also: [`KSPoints`](@ref), [`SpectralMethods.gauss_legendre_with_boundary_nd`](@ref)
"""
function create_collocation_points(nodes::Vector{KSPoint{T}}, polynomial_degree::Int,
    coord_system::AbstractKSCoordinateSystem{N}) where {N,T<:Real}
    if isempty(nodes)
        throw(ArgumentError("Nodes vector cannot be empty"))
    end
    if polynomial_degree <= 0
        throw(ArgumentError("Polynomial degree must be positive"))
    end

    try
        dim = length(nodes[1].coordinates)
        domain = ntuple(d -> (minimum(getindex.(getfield.(nodes, :coordinates), d)),
                maximum(getindex.(getfield.(nodes, :coordinates), d))), dim)

        points, weights = SpectralMethods.gauss_legendre_with_boundary_nd(polynomial_degree, dim, domain...)

        return KSPoints{T}(
            [CoordinateSystems.to_cartesian(p, coord_system) for p in points],
            weights
        )
    catch e
        throw(ErrorException("Failed to create collocation points: $(e.msg)"))
    end
end
