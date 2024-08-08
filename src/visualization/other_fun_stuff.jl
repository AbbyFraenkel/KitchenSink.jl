

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
"""
function plot_streamlines(u::Matrix, v::Matrix, x::Vector, y::Vector;
    xlabel::String="x", ylabel::String="y",
    title::String="Streamlines", kwargs...)
    streamline(x, y, u', v', xlabel=xlabel, ylabel=ylabel, title=title; kwargs...)
end

# Helper functions

"""
    customize_plot!(p::Plots.Plot; font_size::Int=12,
                    legend_position::Symbol=:topright, dpi::Int=300)

Customize the appearance of a plot.
"""
function customize_plot!(p::Plots.Plot; font_size::Int=12,
    legend_position::Symbol=:topright, dpi::Int=300)
    plot!(p, guidefont=font_size, tickfont=font_size, legendfont=font_size,
        legend=legend_position, dpi=dpi)
end

"""
    add_colorbar!(p::Plots.Plot; label::String="")

Add a colorbar to a plot.
"""
function add_colorbar!(p::Plots.Plot; label::String="")
    plot!(p, colorbar=true, colorbar_title=label)
end

"""
    create_subplot_grid(n::Int, layout::Tuple{Int, Int}=(2, 2))

Create a grid of subplots.
"""
function create_subplot_grid(n::Int, layout::Tuple{Int,Int}=(2, 2))
    plots = [plot() for _ in 1:n]
    plot(plots..., layout=layout, size=(800, 600), link=:all)
end
