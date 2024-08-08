
"""
    plot_mesh_and_grid_with_adaptivity(mesh::KSMesh{T}) where {T}

Plot the mesh structure and grid, indicating polynomial order with colors.

# Arguments
- `mesh::KSMesh{T}`: The mesh to be plotted.
"""
function plot_mesh_and_grid_with_adaptivity(mesh::KSMesh{T}) where {T}
    dim = mesh.dimensions
    if dim == 2
        plot_mesh_and_grid_with_adaptivity_2d(mesh)
    elseif dim == 3
        plot_mesh_and_grid_with_adaptivity_3d(mesh)
    else
        error("Unsupported mesh dimensions: $dim. Only 2D and 3D meshes are supported.")
    end
end

function plot_mesh_and_grid_with_adaptivity_2d(mesh::KSMesh{T}) where {T}
    plot()
    for element in mesh.elements
        x = [node.coordinates[1] for node in element.points]
        y = [node.coordinates[2] for node in element.points]
        push!(x, x[1])  # Close the polygon
        push!(y, y[1])

        # Color by polynomial degree
        color = get_color_by_polynomial_order(element.polynomial_degree)
        plot!(x, y, seriestype=:shape, label="", linecolor=color, linewidth=2)

        # Plot nodes
        scatter!(x, y, label="", color=:red, markerstrokewidth=0.5)
    end
    xlabel!("x")
    ylabel!("y")
    title!("2D Mesh and Grid with Adaptivity")
    display(plot())
end

function plot_mesh_and_grid_with_adaptivity_3d(mesh::KSMesh{T}) where {T}
    plot()
    for element in mesh.elements
        x = [node.coordinates[1] for node in element.points]
        y = [node.coordinates[2] for node in element.points]
        z = [node.coordinates[3] for node in element.points]
        push!(x, x[1])  # Close the polygon
        push!(y, y[1])
        push!(z, z[1])

        # Color by polynomial degree
        color = get_color_by_polynomial_order(element.polynomial_degree)
        plot!(x, y, z, seriestype=:wireframe, label="", linecolor=color, linewidth=2)

        # Plot nodes
        scatter!(x, y, z, label="", color=:red, markerstrokewidth=0.5)
    end
    xlabel!("x")
    ylabel!("y")
    zlabel!("z")
    title!("3D Mesh and Grid with Adaptivity")
    display(plot())
end

"""
    get_color_by_polynomial_order(polynomial_degree::Int)::RGB

Get a color for a given polynomial order.

# Arguments
- `polynomial_degree::Int`: The polynomial order.

# Returns
- `RGB`: The color corresponding to the polynomial order.
"""
function get_color_by_polynomial_order(polynomial_degree::Int)::RGB
    colors = [RGB(0.0, 0.0, 1.0), RGB(0.0, 1.0, 0.0), RGB(1.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0), RGB(0.0, 1.0, 1.0)]
    return colors[mod(polynomial_degree - 1, length(colors))+1]
end


"""
    plot_mesh_and_grid(mesh::KSMesh{T}) where {T}

Plot the mesh structure and grid.

# Arguments
- `mesh::KSMesh{T}`: The mesh to be plotted.
"""
function plot_mesh_and_grid(mesh::KSMesh{T}) where {T}
    dim = mesh.dimensions
    if dim == 2
        plot_mesh_and_grid_2d(mesh)
    elseif dim == 3
        plot_mesh_and_grid_3d(mesh)
    else
        error("Unsupported mesh dimensions: $dim. Only 2D and 3D meshes are supported.")
    end
end
