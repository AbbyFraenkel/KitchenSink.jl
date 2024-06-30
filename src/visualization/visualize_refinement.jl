function visualize_hp_refinement(mesh_nodes, p_orders)
    scatter(mesh_nodes, zeros(length(mesh_nodes)),
        label = "Mesh Nodes", color = :blue, markersize = 4)
    for (i, p) in enumerate(p_orders)
        pos = (mesh_nodes[i] + mesh_nodes[i + 1]) / 2
        annotate!(pos, 0.2, text("p=$p", 8, :center, :bottom))
    end
    plot!(title = "HP-Refinement Visualization",
        xlabel = "Domain", ylabel = "", legend = false)
    plot!(xlims = (minimum(mesh_nodes), maximum(mesh_nodes)),
        ylims = (-1, maximum(p_orders) + 1))
    return plot
end

function convergence_study()
    # Initialize the range of possible values for the number of elements and polynomial order
    elements_range = 10:10:100
    order_range = 1:1:10

    # Initialize an array to store the error for each combination of parameters
    errors = zeros(length(elements_range), length(order_range))

    # Loop over all combinations of parameters
    for (i, num_elements) in enumerate(elements_range)
        for (j, order) in enumerate(order_range)
            # Solve the PDE with the current combination of parameters
            solution = solve_pde(num_elements, order)

            # Compute the residual error of the solution
            error = residual_error_estimate(A, solution, f)

            # Store the error in the array
            errors[i, j] = error
        end
    end

    # Print the errors
    for (i, num_elements) in enumerate(elements_range)
        for (j, order) in enumerate(order_range)
            println("Number of elements: ",
                num_elements,
                ", Polynomial order: ",
                order,
                ", Error: ",
                errors[i, j])
        end
    end
end
