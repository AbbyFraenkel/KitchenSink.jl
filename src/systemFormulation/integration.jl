function localIntegralWeightsMatrix(w)
	n = length(w)
	# Initialize the local mass matrix
	m = spzeros(n + 1, n + 1)
	# Fill the diagonal of the local mass matrix with the weights
	m[1:n, 1:n] = spdiagm(0 => w)
  end
function globalIntegralWeightsMatrix(num_elements, num_nodes_per_element, w)
    N_j = num_nodes_per_element + 1
    # Determine the total number of nodes in the mesh
    num_nodes = num_elements * (N_j - 1) + 1

    # Initialize the global mass matrix
    M = spzeros(num_nodes, num_nodes)

    # Loop over all elements in the mesh
    for e in 1:num_elements
        # Compute the local mass matrix
        Me = localIntegralWeightsMatrix(w)

        # Insert the local mass matrix into the global mass matrix
        M[((e - 1) * (N_j - 1) + 1):((e - 1) * (N_j - 1) + N_j), ((e - 1) * (N_j - 1) + 1):((e - 1) * (N_j - 1) + N_j)] = Me
    end

    return M
end
