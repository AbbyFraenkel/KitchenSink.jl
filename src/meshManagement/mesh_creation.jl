@doc """
    ensure_higher_order_continuity!(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}, num_elements::Int, num_basis::Int, continuity_order::Int=2)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}

Ensure higher order continuity in the system by modifying the system matrix and the right-hand side vector.

# Arguments
- `A::SparseMatrixCSC{Float64,Int}`: System matrix.
- `b::Vector{Float64}`: Right-hand side vector.
- `num_elements::Int`: Number of elements in the mesh.
- `num_basis::Int`: Number of basis functions per element.
- `continuity_order::Int=2`: Desired order of continuity between elements (default is 2).

# Returns
- `Tuple{SparseMatrixCSC{Float64,Int}, Vector{Float64}}`: Modified system matrix and right-hand side vector to enforce continuity.

# Description
This function enforces higher order continuity between adjacent elements in the mesh by modifying the system matrix `A` and the right-hand side vector `b`. The continuity order is specified by `continuity_order`, which defaults to 2. The function iterates over the elements and adjusts the matrix and vector entries to ensure the desired continuity.
"""
function EnsureHigherOrderContinuity!(
    A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    num_elements::Int,
    num_basis::Int,
    continuity_order::Int = 2,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    for e = 1:(num_elements-1)
        for k = 1:continuity_order
            i1 = e * num_basis - (num_basis - k)
            i2 = e * num_basis + k

            A[i1, :] .-= A[i2, :]
            A[:, i1] .-= A[:, i2]
            b[i1] -= b[i2]
        end
    end
    return A, b
end


@inline function check_degrees_of_freedom(
    A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    num_dofs::Int,
)::Nothing
    @assert size(A, 1) == num_dofs "System matrix row size does not match number of degrees of freedom"
    @assert size(A, 2) == num_dofs "System matrix column size does not match number of degrees of freedom"
    @assert length(b) == num_dofs "Right-hand side vector size does not match number of degrees of freedom"
end

function MeshAssembly(
    ndims::Int,
    domain::Vector{Float64},
    num_elements::Int,
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    lambda::Float64,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    num_basis = length(basis_functions)
    num_dofs = num_elements^ndims * num_basis
    A = spzeros(Float64, num_dofs, num_dofs)
    b = zeros(Float64, num_dofs)

    Threads.@threads for el in CartesianIndices(ntuple(_ -> num_elements, ndims))
        for i = 1:num_basis
            for j = 1:num_basis
                A[
                    linear_index(el, i, num_elements, num_basis),
                    linear_index(el, j, num_elements, num_basis),
                ] += integrate_basis(
                    ndims,
                    domain,
                    basis_functions[i],
                    basis_functions[j],
                    collocation_points,
                    weights,
                )
            end
            b[linear_index(el, i, num_elements, num_basis)] += integrate_forcing(
                ndims,
                domain,
                basis_functions[i],
                collocation_points,
                weights,
            )
        end
    end

    A .+= lambda * I

    return A, b
end


function split_or_merge_elements(element_errors, mesh, max_error)
    # Split or merge elements based on the error
    new_mesh = []
    i = 1
    while i <= length(mesh)
        if element_errors[i] > max_error
            # Split the element
            new_mesh = vcat(new_mesh, split_element(mesh[i]))
        elseif i < length(mesh) && element_errors[i] < max_error / 2 &&
               element_errors[i + 1] < max_error / 2
            # Merge with the next element
            new_mesh = vcat(new_mesh, merge_elements(mesh[i], mesh[i + 1]))
            i += 1  # Skip the next element
        else
            # Keep the element as is
            new_mesh = vcat(new_mesh, mesh[i])
        end
        i += 1
    end
    return new_mesh
end



function adjust_polynomial_order(x_nodes, f, max_error, min_order, max_order)
  # x_nodes: Nodes for polynomial interpolation
  # f: Function to interpolate
  # max_error: Maximum allowable interpolation error
  # min_order: Minimum allowable polynomial order
  # max_order: Maximum allowable polynomial order

  # Compute the Lagrange polynomials
  L = Lagrange_polynomials(x_nodes)

  # Compute the interpolation error
  error = norm(L * f.(x_nodes) - f.(x_nodes), Inf)

  # Increase or decrease the polynomial order based on the error
  if error > max_error && length(x_nodes) < max_order
    # Increase the polynomial order by adding a node
    x_nodes = vcat(x_nodes, mean(x_nodes))
  elseif error < max_error / 2 && length(x_nodes) > min_order
    # Decrease the polynomial order by removing a node
    x_nodes = x_nodes[1:end-1]
  end

  return x_nodes
end

function update_mesh_and_basis_functions(root::TreeNode)
  # Adjust mesh and basis functions based on refined elements
  # This may involve updating the element connectivity, nodal coordinates, and basis function evaluations

  # Iterate over all elements
  for element in collect_elements(root)
    # Check if the element has been refined
    if element.refined
      # Update the element's connectivity, coordinates, and basis function evaluations
      # The specifics of this will depend on your implementation
      element.connectivity = update_connectivity(element)
      element.coordinates = update_coordinates(element)
      element.basis_functions = update_basis_functions(element)
    end
  end
end

function update_mesh_and_basis_functions(root::TreeNode)
    for element in collect_elements(root)
        if element.refined
            element.connectivity = update_connectivity(element)
            element.coordinates = update_coordinates(element)
            element.basis_functions = update_basis_functions(element)
        end
    end
end

function update_connectivity(element::Element)
    new_connectivity = []

    function add_connectivity(dims, indices)
        if dims == 1
            push!(new_connectivity, (indices[1], indices[1] + 1))
        else
            push!(new_connectivity, (indices..., indices[1:(end - 1)]..., indices[end] + 1))
        end
    end

    function iterate_dims(dims, sizes, indices)
        if dims == 0
            add_connectivity(length(indices), indices)
        else
            for i in 1:(sizes[dims] - 1)
                iterate_dims(dims - 1, sizes, [i, indices...])
            end
        end
    end

    dims = length(element.nodes)
    sizes = map(length, element.nodes)

    iterate_dims(dims, sizes, [])

    return new_connectivity
end

function update_coordinates(element::Element)
    new_coordinates = []

    for i in 1:length(element.nodes)
        coords = element.nodes[i]
        n = length(coords)
        new_coords = []

        for j in 1:(n - 1)
            mid = (coords[j] + coords[j + 1]) / 2
            push!(new_coords, coords[j])
            push!(new_coords, mid)
        end
        push!(new_coords, coords[end])
        push!(new_coordinates, new_coords)
    end

    return new_coordinates
end
