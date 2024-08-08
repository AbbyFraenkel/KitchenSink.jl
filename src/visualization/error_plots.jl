"""
    plot_error_distribution(mesh::KSMesh{T}) where {T}

Plot the error distribution across the mesh.

# Arguments
- `mesh::KSMesh{T}`: The mesh with error estimates to be plotted.
"""
function plot_error_distribution(mesh::KSMesh{T}) where {T}
    dim = mesh.dimensions
    if dim == 2
        plot_error_distribution_2d(mesh)
    elseif dim == 3
        plot_error_distribution_3d(mesh)
    else
        error("Unsupported mesh dimensions: $dim. Only 2D and 3D meshes are supported.")
    end
end

function plot_error_distribution_2d(mesh::KSMesh{T}) where {T}
    plot()
    for element in mesh.elements
        x = [node.coordinates[1] for node in element.points]
        y = [node.coordinates[2] for node in element.points]
        push!(x, x[1])  # Close the polygon
        push!(y, y[1])

        # Color by error estimate
        color = get_color_by_error_estimate(element.error_estimate)
        plot!(x, y, seriestype=:shape, label="", linecolor=color, linewidth=2)

        # Plot nodes
        scatter!(x, y, label="", color=:red, markerstrokewidth=0.5)
    end
    xlabel!("x")
    ylabel!("y")
    title!("2D Error Distribution")
    display(plot())
end

function plot_error_distribution_3d(mesh::KSMesh{T}) where {T}
    plot()
    for element in mesh.elements
        x = [node.coordinates[1] for node in element.points]
        y = [node.coordinates[2] for node in element.points]
        z = [node.coordinates[3] for node in element.points]
        push!(x, x[1])  # Close the polygon
        push!(y, y[1])
        push!(z, z[1])

        # Color by error estimate
        color = get_color_by_error_estimate(element.error_estimate)
        plot!(x, y, z, seriestype=:wireframe, label="", linecolor=color, linewidth=2)

        # Plot nodes
        scatter!(x, y, z, label="", color=:red, markerstrokewidth=0.5)
    end
    xlabel!("x")
    ylabel!("y")
    zlabel!("z")
    title!("3D Error Distribution")
    display(plot())
end

"""
    get_color_by_error_estimate(error_estimate::Float64)::RGB

Get a color for a given error estimate.

# Arguments
- `error_estimate::Float64`: The error estimate.

# Returns
- `RGB`: The color corresponding to the error estimate.
"""
function get_color_by_error_estimate(error_estimate::Float64)::RGB
    if error_estimate < 0.1
        return RGB(0.0, 1.0, 0.0)  # Green for low error
    elseif error_estimate < 0.5
        return RGB(1.0, 1.0, 0.0)  # Yellow for medium error
    else
        return RGB(1.0, 0.0, 0.0)  # Red for high error
    end
end
