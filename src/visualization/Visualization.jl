module KSVisualization

using Plots, Makie, GLMakie, LaTeXStrings, DataFrames, StatsPlots, Measures, ColorSchemes
using ..KSTypes

export plot_error_distribution, plot_mesh_and_grid_with_adaptivity, plot_solution
export plot_eigenspectrum, plot_stability_region, interactive_solution_explorer
export animate_solution, plot_phase_portrait, plot_parametric_study, plot_sensitivity_analysis
export plot_reduced_model, plot_multiscale_solution, plot_convergence, plot_streamlines
export customize_plot!, add_colorbar!, create_subplot_grid

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

"""
    plot_eigenspectrum(A::AbstractMatrix; kwargs...)

Plot the eigenvalue spectrum of a matrix.
"""
function plot_eigenspectrum(A::AbstractMatrix;
    xlabel::String="Re(λ)", ylabel::String="Im(λ)",
    title::String="Eigenvalue Spectrum", kwargs...)
    eigvals = eigen(A).values
    scatter(real.(eigvals), imag.(eigvals), xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

"""
    plot_stability_region(method::Function, order::Int; kwargs...)

Plot the stability region of a numerical method.
"""
function plot_stability_region(method::Function, order::Int;
    xlabel::String="Re(z)", ylabel::String="Im(z)",
    title::String="Stability Region", kwargs...)
    x = range(-5, 5, length=100)
    y = range(-5, 5, length=100)
    z = [x + y * im for x in x, y in y]
    stability = abs.(method.(z, order)) .<= 1
    contour(x, y, stability, levels=[1], color=:black, xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

"""
    interactive_solution_explorer(problem::AbstractKSProblem, solution::Array, times::Vector{<:Real})

Create an interactive solution explorer for time-dependent problems.
"""
function interactive_solution_explorer(problem::AbstractKSProblem, solution::Array, times::Vector{<:Real})
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1])
    slider = Slider(fig[2, 1], range=1:length(times), startvalue=1)
    time_label = Label(fig[3, 1], @lift("Time: $(times[$(slider.value)])"))

    function update_plot(t_index)
        empty!(ax)
        if problem isa KSODEProblem || (problem isa KSPDEProblem && problem.domain isa Tuple{<:Real,<:Real})
            x = range(problem.domain[1], problem.domain[2], length=size(solution, 1))
            lines!(ax, x, solution[:, t_index])
        elseif problem isa KSPDEProblem && problem.domain isa NTuple{2,Tuple{<:Real,<:Real}}
            x = range(problem.domain[1][1], problem.domain[1][2], size(solution, 1))
            y = range(problem.domain[2][1], problem.domain[2][2], size(solution, 2))
            heatmap!(ax, x, y, solution[:, :, t_index])
        end
    end

    on(slider.value) do t_index
        update_plot(t_index)
    end

    update_plot(1)
    fig
end


"""
    animate_solution(problem::AbstractKSProblem, solution::Array, times::Vector{<:Real};
        fps::Int=30, kwargs...)

Animate the solution of a time-dependent problem.

# Arguments
- `problem::AbstractKSProblem`: The problem being solved.
- `solution::Array`: The solution array.
- `times::Vector{<:Real}`: The time points.
- `fps::Int=30`: Frames per second for the animation.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- An animation of the solution.
"""
function animate_solution(problem::AbstractKSProblem, solution::Array, times::Vector{<:Real};
    fps::Int=30, kwargs...)
    anim = @animate for (i, t) in enumerate(times)
        plot_solution(problem, solution[:, :, i], t; kwargs...)
    end
    return gif(anim, fps=fps)
end

"""
    plot_phase_portrait(problem::KSODEProblem, solutions::Vector{Vector{T}}; kwargs...) where T

Plot the phase portrait for a system of ODEs.

# Arguments
- `problem::KSODEProblem`: The ODE problem.
- `solutions::Vector{Vector{T}}`: The solutions.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A plot of the phase portrait.
"""
function plot_phase_portrait(problem::KSODEProblem, solutions::Vector{Vector{T}};
    xlabel::String="x", ylabel::String="y",
    title::String="Phase Portrait", kwargs...) where {T}
    if length(problem.initial_condition) != 2
        error("Phase portrait plotting is only supported for 2D systems")
    end

    plot(first.(solutions), last.(solutions), xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

"""
    plot_parametric_study(results::Dict, param_name::String; kwargs...)

Visualize the results of a parametric study.

# Arguments
- `results::Dict`: Dictionary containing parameter values and corresponding solutions.
- `param_name::String`: Name of the parameter being studied.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A plot showing the parametric study results.
"""
function plot_parametric_study(results::Dict, param_name::String;
    xlabel::String=param_name, ylabel::String="Solution norm",
    title::String="Parametric Study Results", kwargs...)
    params = [p[1] for p in keys(results)]
    solution_norms = [norm(sol) for sol in values(results)]
    scatter(params, solution_norms, xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

"""
    plot_sensitivity_analysis(sensitivities::Dict; kwargs...)

Create a heatmap visualization of sensitivity analysis results.

# Arguments
- `sensitivities::Dict`: Dictionary containing sensitivity values for different parameters.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A heatmap showing the sensitivity analysis results.
"""
function plot_sensitivity_analysis(sensitivities::Dict;
    title::String="Sensitivity Analysis", kwargs...)
    param_names = collect(keys(sensitivities))
    sensitivity_values = collect(values(sensitivities))
    heatmap(param_names, param_names, sensitivity_values,
        title=title, xlabel="Parameters", ylabel="Parameters"; kwargs...)
end

"""
    plot_reduced_model(full_solution::Vector, reduced_solution::Vector,
                       basis::Matrix, mean_snapshot::Vector; kwargs...)

Visualize the performance of a reduced-order model compared to the full model.

# Arguments
- `full_solution::Vector`: The solution from the full model.
- `reduced_solution::Vector`: The solution from the reduced-order model.
- `basis::Matrix`: The basis used for the reduced-order model.
- `mean_snapshot::Vector`: The mean snapshot used in the reduction process.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A plot comparing the full and reduced model solutions.
"""
function plot_reduced_model(full_solution::Vector, reduced_solution::Vector,
    basis::Matrix, mean_snapshot::Vector;
    xlabel::String="x", ylabel::String="u(x)",
    title::String="Full vs. Reduced Model", kwargs...)
    x = range(0, 1, length=length(full_solution))
    reconstructed_solution = basis * reduced_solution + mean_snapshot
    plot(x, [full_solution reconstructed_solution], label=["Full Model" "Reduced Model"],
        xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

"""
    plot_multiscale_solution(coarse_solution::Vector, fine_solutions::Vector{Vector{T}},
                             fine_regions::Vector{Tuple{T, T}}; kwargs...) where T

Visualize the results of a multiscale simulation.

# Arguments
- `coarse_solution::Vector`: The solution on the coarse scale.
- `fine_solutions::Vector{Vector{T}}`: The solutions on the fine scales.
- `fine_regions::Vector{Tuple{T, T}}`: The regions where fine scale solutions are applied.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A plot showing the multiscale solution.
"""
function plot_multiscale_solution(coarse_solution::Vector, fine_solutions::Vector{Vector{T}},
    fine_regions::Vector{Tuple{T,T}};
    xlabel::String="x", ylabel::String="u(x)",
    title::String="Multiscale Solution", kwargs...) where {T}
    x_coarse = range(0, 1, length=length(coarse_solution))
    plot(x_coarse, coarse_solution, label="Coarse Solution",
        xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)

    for (fine_sol, region) in zip(fine_solutions, fine_regions)
        x_fine = range(region[1], region[2], length=length(fine_sol))
        plot!(x_fine, fine_sol, label="Fine Solution ($(region[1]) - $(region[2]))")
    end
    plot!()
end

"""
    plot_convergence(errors::Vector{T}, iterations::Vector{Int}; kwargs...) where T

Plot the convergence history of an iterative method.

# Arguments
- `errors::Vector{T}`: The error values at each iteration.
- `iterations::Vector{Int}`: The iteration numbers.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A plot showing the convergence history.
"""
function plot_convergence(errors::Vector{T}, iterations::Vector{Int};
    xlabel::String="Iteration", ylabel::String="Error",
    title::String="Convergence History", kwargs...) where {T}
    plot(iterations, errors, xlabel=xlabel, ylabel=ylabel, title=title,
        yaxis=:log, marker=:circle; kwargs...)
end

"""
    plot_streamlines(u::Matrix, v::Matrix, x::Vector, y::Vector; kwargs...)

Plot streamlines for a vector field (useful for fluid dynamics problems).

# Arguments
- `u::Matrix`: The x-component of the vector field.
- `v::Matrix`: The y-component of the vector field.
- `x::Vector`: The x-coordinates.
- `y::Vector`: The y-coordinates.
- `kwargs...`: Additional keyword arguments for the plot.

# Returns
- A plot showing the streamlines of the vector field.
"""
function plot_streamlines(u::Matrix, v::Matrix, x::Vector, y::Vector;
    xlabel::String="x", ylabel::String="y",
    title::String="Streamlines", kwargs...)
    streamline(x, y, u', v', xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

"""
    customize_plot!(p::Plots.Plot; font_size::Int=12,
                    legend_position::Symbol=:topright, dpi::Int=300)

Customize the appearance of a plot.

# Arguments
- `p::Plots.Plot`: The plot to customize.
- `font_size::Int=12`: The font size for labels and titles.
- `legend_position::Symbol=:topright`: The position of the legend.
- `dpi::Int=300`: The resolution of the plot.

# Returns
- The customized plot.
"""
function customize_plot!(p::Plots.Plot; font_size::Int=12,
    legend_position::Symbol=:topright, dpi::Int=300)
    plot!(p, guidefont=font_size, tickfont=font_size, legendfont=font_size,
        legend=legend_position, dpi=dpi)
end

"""
    add_colorbar!(p::Plots.Plot; label::String="")

Add a colorbar to a plot.

# Arguments
- `p::Plots.Plot`: The plot to add the colorbar to.
- `label::String=""`: The label for the colorbar.

# Returns
- The plot with the added colorbar.
"""
function add_colorbar!(p::Plots.Plot; label::String="")
    plot!(p, colorbar=true, colorbar_title=label)
end

"""
    create_subplot_grid(n::Int, layout::Tuple{Int, Int}=(2, 2))

Create a grid of subplots.

# Arguments
- `n::Int`: The number of subplots.
- `layout::Tuple{Int, Int}=(2, 2)`: The layout of the subplots.

# Returns
- A plot with a grid of subplots.
"""
function create_subplot_grid(n::Int, layout::Tuple{Int,Int}=(2, 2))
    plots = [plot() for _ in 1:n]
    plot(plots..., layout=layout, size=(800, 600), link=:all)
end

end # module KSVisualization
