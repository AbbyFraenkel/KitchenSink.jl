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
