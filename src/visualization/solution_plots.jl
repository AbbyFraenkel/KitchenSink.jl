
"""
    plot_solution(mesh::KSMesh{T}, solution::Vector{T}) where {T}

Plot the solution over the domain.

# Arguments
- `mesh::KSMesh{T}`: The mesh on which the solution is defined.
- `solution::Vector{T}`: The solution vector.
"""
function plot_solution(mesh::KSMesh{T}, solution::Vector{T}) where {T}
    scatter()
    for element in mesh.elements
        x = [node.coordinates[1] for node in element.points]
        y = [node.coordinates[2] for node in element.points]
        z = solution[get_global_indices(mesh, element)]
        scatter!(x, y, z, label="", markerstrokewidth=0, c=z, color=:viridis)
    end
    xlabel!("x")
    ylabel!("y")
    title!("Solution Plot")
    display(scatter())
end
