module Visualization

using Plots
using ..KSTypes
using ..CoordinateSystems

export plot_solution, plot_error_distribution, plot_mesh, get_node_coordinates

function plot_solution(u::AbstractVector{T}, mesh::KSMesh{T, N}) where {T <: Real, N}
    coords = get_node_coordinates(mesh)
    plot(coords, u, title = "Solution", xlabel = "x", ylabel = "u(x)", legend = false)
end

function plot_error_distribution(
        error::AbstractVector{T},
        mesh::KSMesh{T, N}
) where {T <: Real, N}
    coords = get_node_coordinates(mesh)
    scatter(
        coords,
        error,
        title = "Error Distribution",
        xlabel = "x",
        ylabel = "Error",
        yscale = :log10,
        legend = false
    )
end

function plot_mesh(mesh::KSMesh{T, N}) where {T <: Real, N}
    coords = get_node_coordinates(mesh)
    p = scatter(
        coords,
        zeros(length(coords)),
        title = "Mesh Structure",
        xlabel = "x",
        ylabel = "",
        yticks = [],
        legend = false
    )
    vline!(coords, color = :black, linestyle = :dash, label = "")
    return p
end


end # module Visualization
