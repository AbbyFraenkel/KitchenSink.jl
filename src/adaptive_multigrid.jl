module AdaptiveMultigrid

export create_grid_hierarchy,
    create_operators,
    apply_boundary_conditions!,
    smooth,
    gauss_seidel,
    sor,
    restrict,
    prolong,
    v_cycle,
    adaptive_multigrid_solver

using LinearAlgebra
using SparseArrays
using Logging
using FastGaussQuadrature
using ..Types
using ..SpectralMethods

"""
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

"""
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

"""
    apply_boundary_conditions!(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, boundary_conditions::Vector{Tuple{Int, Float64}})::Tuple{SparseMatrixCSC{Float64}, Vector{Float64}}

Apply boundary conditions to the system matrix and right-hand side vector.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `boundary_conditions::Vector{Tuple{Int, Float64}}`: Boundary conditions specified as tuples of index and value.

# Returns
- `Tuple{SparseMatrixCSC{Float64}, Vector{Float64}}`: System matrix and right-hand side vector after applying boundary conditions.
"""
function apply_boundary_conditions!(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    boundary_conditions::Vector{Tuple{Int,Float64}},
)::Tuple{SparseMatrixCSC{Float64},Vector{Float64}}
    for (idx, value) in boundary_conditions
        A[idx, :] .= 0
        A[idx, idx] = 1
        b[idx] = value
    end
    return A, b
end

"""
    smooth(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, x::Vector{Float64}, iterations::Int)::Vector{Float64}

Perform Jacobi smoothing iterations.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `iterations::Int`: Number of smoothing iterations.

# Returns
- `x::Vector{Float64}`: Smoothed solution.
"""
function smooth(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iterations::Int,
)::Vector{Float64}
    D_vec = diagm(A)  # Extract the diagonal elements as a vector
    for _ = 1:iterations
        x = (b .- A * x .+ D_vec .* x) ./ D_vec
    end
    return x
end

"""
    gauss_seidel(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, x::Vector{Float64}, iterations::Int)::Vector{Float64}

Perform Gauss-Seidel smoothing iterations.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `iterations::Int`: Number of smoothing iterations.

# Returns
- `x::Vector{Float64}`: Smoothed solution.
"""
function gauss_seidel(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iterations::Int,
)::Vector{Float64}
    n = length(b)
    for _ = 1:iterations
        for i = 1:n
            sigma = sum(A[i, j] * x[j] for j = 1:n if j != i)
            x[i] = (b[i] - sigma) / A[i, i]
        end
    end
    return x
end

"""
    sor(A::SparseMatrixCSC{Float64}, b::Vector{Float64}, x::Vector{Float64}, omega::Float64, iterations::Int)::Vector{Float64}

Perform Successive Over-Relaxation (SOR) smoothing iterations.

# Arguments
- `A::SparseMatrixCSC{Float64}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `omega::Float64`: Relaxation parameter.
- `iterations::Int`: Number of smoothing iterations.

# Returns
- `x::Vector{Float64}`: Smoothed solution.
"""
function sor(
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    omega::Float64,
    iterations::Int,
)::Vector{Float64}
    @assert 0 < omega < 2 "Relaxation parameter ω must be in (0, 2)"
    n = length(b)
    for _ = 1:iterations
        for i = 1:n
            sigma = sum(A[i, j] * x[j] for j = 1:n if j != i)
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        end
    end
    return x
end

"""
    restrict(residual_fine::Array{Float64, N}) where N::Array{Float64, N}

Restrict the residual to a coarser grid.

# Arguments
- `residual_fine::Array{Float64, N}`: Residual on the fine grid.

# Returns
- `residual_coarse::Array{Float64, N}`: Residual on the coarse grid.
"""
function restrict(residual_fine::Array{Float64,N}) where {N}
    size_fine = size(residual_fine)
    size_coarse = div.(size_fine .+ 1, 2)
    residual_coarse = zeros(Float64, size_coarse)

    _, w_fine = gausslegendre(size_fine[1])
    w_fine = w_fine / sum(w_fine)

    weights = ntuple(d -> w_fine, N)

    for idx in CartesianIndices(residual_coarse)
        fine_indices = CartesianIndices(ntuple(d -> 2*idx[d]-1:2*idx[d], N)...)
        sum_weights = 0.0
        value_sum = 0.0

        for fi in fine_indices
            weight = prod(weights[d][fi[d]] for d = 1:N)
            value_sum += residual_fine[fi] * weight
            sum_weights += weight
        end

        residual_coarse[idx] = value_sum / sum_weights
    end

    return residual_coarse
end

"""
    prolong(correction_coarse::Array{Float64, N}) where N::Array{Float64, N}

Prolong the correction to a finer grid.

# Arguments
- `correction_coarse::Array{Float64, N}`: Correction on the coarse grid.

# Returns
- `correction_fine::Array{Float64, N}`: Correction on the fine grid.
"""
function prolong(correction_coarse::Array{Float64,N}) where {N}
    size_coarse = size(correction_coarse)
    size_fine = 2 .* (size_coarse .- 1) .+ 1
    correction_fine = zeros(Float64, size_fine)

    for idx in CartesianIndices(correction_coarse)
        fine_indices = CartesianIndices(ntuple(d -> 2*idx[d]-1:2*idx[d], N)...)
        correction_fine[fine_indices...] .= correction_coarse[idx]
        for d = 1:N
            neighbor_idx = copy(fine_indices)
            neighbor_idx[d] += 1
            if neighbor_idx[d] <= size_fine[d]
                correction_fine[neighbor_idx...] .=
                    0.5 * (correction_coarse[idx] + correction_coarse[neighbor_idx])
            end
        end
    end

    return correction_fine
end

"""
    v_cycle(level::Int, levels::Int, mesh::MultiLayerMesh, operators::Vector{SparseMatrixCSC{Float64}}, b::Vector{Float64}, x::Vector{Float64}, smoothing_iterations::Int, logger::ConsoleLogger)::Vector{Float64}

Perform a V-cycle multigrid iteration.

# Arguments
- `level::Int`: Current grid level.
- `levels::Int`: Total number of grid levels.
- `mesh::MultiLayerMesh`: Multi-layer mesh with hierarchical grids.
- `operators::Vector{SparseMatrixCSC{Float64}}`: Operators for each grid level.
- `b::Vector{Float64}`: Right-hand side vector.
- `x::Vector{Float64}`: Initial guess for the solution.
- `smoothing_iterations::Int`: Number of smoothing iterations.
- `logger::ConsoleLogger`: Logger for output.

# Returns
- `x::Vector{Float64}`: Solution vector after V-cycle.
"""
function v_cycle(
    level::Int,
    levels::Int,
    mesh::MultiLayerMesh,
    operators::Vector{SparseMatrixCSC{Float64}},
    b::Vector{Float64},
    x::Vector{Float64},
    smoothing_iterations::Int,
    logger::ConsoleLogger,
)::Vector{Float64}
    if level == levels
        x = operators[level] \ b
    else
        x = smooth(operators[level], b, x, smoothing_iterations)
        residual = b - operators[level] * x
        residual_coarse = restrict(residual)
        correction_coarse = zeros(length(residual_coarse))
        correction_coarse = v_cycle(
            level + 1,
            levels,
            mesh,
            operators,
            residual_coarse,
            correction_coarse,
            smoothing_iterations,
            logger,
        )
        correction_fine = prolong(correction_coarse)
        x .+= correction_fine
        x = smooth(operators[level], b, x, smoothing_iterations)
    end
    @info logger "V-cycle at level $level completed"
    return x
end

"""
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

"""
    refine_grid(layer::GridLayer, marked_elements::Vector{Int})::GridLayer

Refine the grid at the given level.

# Arguments
- `layer::GridLayer`: Grid layer at the current level.
- `marked_elements::Vector{Int}`: Indices of elements to be refined.

# Returns
- `refined_layer::GridLayer`: Refined grid layer.
"""
function refine_grid(layer::GridLayer, marked_elements::Vector{Int})::GridLayer
    refined_nodes = Vector{Vector{Float64}}()
    refined_elements = Vector{Element}()
    for i = 1:(length(layer.elements)-1)
        push!(refined_nodes, layer.elements[i].nodes)
        push!(refined_elements, layer.elements[i])
        if i in marked_elements
            midpoint = (layer.elements[i].nodes .+ layer.elements[i+1].nodes) ./ 2
            push!(refined_nodes, midpoint)
            new_element = deepcopy(layer.elements[i])
            new_element.nodes = midpoint
            push!(refined_elements, new_element)
        end
    end
    push!(refined_nodes, layer.elements[end].nodes)
    push!(refined_elements, layer.elements[end])

    new_layer = deepcopy(layer)
    new_layer.nodes = refined_nodes
    new_layer.elements = refined_elements
    new_layer.levels .= maximum(new_layer.levels) + 1

    return new_layer
end

end # module
