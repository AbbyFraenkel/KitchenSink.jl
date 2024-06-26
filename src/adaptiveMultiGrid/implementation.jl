
@doc """
    create_grid_hierarchy(ndims::Int, levels::Int, p::Int)::MultiLayerMesh

Create a hierarchy of grids for the multigrid solver.

# Arguments
- `ndims::Int`: Number of dimensions.
- `levels::Int`: Number of levels in the grid hierarchy.
- `p::Int`: Degree of the Legendre polynomials for the basis functions.

# Returns
- `mesh::MultiLayerMesh`: Multi-layer mesh with hierarchical grids.
"""
function create_grid_hierarchy(ndims::Int, levels::Int, p::Int)::MultiLayerMesh
    layers = Vector{GridLayer}(undef, levels)
    for l = 1:levels
        step = 2.0^(-l)
        ranges = ntuple(_ -> 0.0:step:1.0, ndims)
        points = collect(Iterators.product(ranges...))
        # Convert tuple to array directly instead of converting each point to a collection
        nodes = [collect(point) for point in points]

        basis_functions, _, _ = legendre_basis_functions(p)

        # Use array comprehension directly for elements
        elements = [
            Element(
                collect(node),
                basis_functions,
                collect(1:length(basis_functions)),
                nothing,
                l,
                0,
                Matrix{Float64}(undef, 0, 0),
                Matrix{Int}(undef, 0, 0),
            ) for node in nodes
        ]
        levels_arr = fill(l, length(nodes))
        is_leaf = fill(true, length(nodes))
        degrees = fill(1, length(nodes))
        layers[l] = GridLayer(
            nodes,
            elements,
            fill(-1, (length(nodes), ndims, 2)),
            levels_arr,
            is_leaf,
            degrees,
        )
    end
    return MultiLayerMesh(layers, ndims)
end

@doc """
    create_operators(mesh::MultiLayerMesh)::Vector{SparseMatrixCSC{Float64}}

Create the operators for each grid level.

# Arguments
- `mesh::MultiLayerMesh`: Multi-layer mesh with hierarchical grids.

# Returns
- `operators::Vector{SparseMatrixCSC{Float64}}`: Operators for each grid level.
"""
function create_operators(mesh::MultiLayerMesh)::Vector{SparseMatrixCSC{Float64}}
    operators = Vector{SparseMatrixCSC{Float64}}(undef, length(mesh.layers))
    for (level, layer) in enumerate(mesh.layers)
        n = length(layer.elements)
        A = spzeros(Float64, n, n)
        step = 2.0^(-level + 1)
        for i = 1:n
            A[i, i] = 2 * mesh.ndims
            for d = 1:mesh.ndims
                neighbor = deepcopy(layer.elements[i].nodes)
                neighbor[d] -= step
                idx = findfirst(x -> all(isapprox.(x.nodes, neighbor)), layer.elements)
                if idx !== nothing
                    A[i, idx] = -1
                end
                neighbor[d] += 2 * step
                idx = findfirst(x -> all(isapprox.(x.nodes, neighbor)), layer.elements)
                if idx !== nothing
                    A[i, idx] = -1
                end
            end
        end
        operators[level] = A
    end
    return operators
end


@doc """
    adaptive_multigrid_solver(problem::AbstractProblem, solver::Solver, levels::Int, error_threshold::Float64, p::Int)::Vector{Float64}

Solve the problem using an adaptive multigrid method.

# Arguments
- `problem::AbstractProblem`: Problem to be solved.
- `solver::Solver`: Solver configuration.
- `levels::Int`: Number of levels in the grid hierarchy.
- `error_threshold::Float64`: Error threshold for grid refinement.
- `p::Int`: Degree of the Legendre polynomials for the basis functions.

# Returns
- `solution::Vector{Float64}`: Solution vector.
"""
function adaptive_multigrid_solver(
    problem::AbstractProblem,
    solver::Solver,
    levels::Int,
    error_threshold::Float64,
    p::Int,
)::Vector{Float64}
    ndims = length(problem.domain)
    mesh = create_grid_hierarchy(ndims, levels, p)
    operators = create_operators(mesh)
    x = zeros(length(mesh.layers[1].elements))
    b = ones(length(mesh.layers[1].elements))
    boundary_conditions = [(1, 0.0), (length(b), 0.0)]
    A, b = apply_boundary_conditions!(operators[1], b, boundary_conditions)
    logger = ConsoleLogger()

    for iter = 1:solver.max_iterations
        @info logger "Iteration $iter started"
        x = v_cycle(1, levels, mesh, operators, b, x, 3, logger)
        residual = norm(b - A * x)
        if residual < solver.tolerance
            @info logger "Converged in $iter iterations with residual $residual"
            break
        end
        error_estimates = [norm(b - operators[i] * x) / norm(b) for i = 1:levels]
        marked_elements = findall(e -> e > error_threshold, error_estimates)
        for level = 1:levels-1
            mesh.layers[level+1] = refine_grid(mesh.layers[level], marked_elements)
            operators[level+1] = create_operators(mesh)[level+1]
        end
        @info logger "Marked elements for refinement: $marked_elements"
    end
    return x
end
