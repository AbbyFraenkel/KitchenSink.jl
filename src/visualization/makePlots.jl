
function plot_solution(x::Vector{Float64}, u::Vector{Float64})::Nothing
    plot(x, u, lw = 2, label = "Solution")
    xlabel!("x")
    ylabel!("u(x)")
    title!("Numerical Solution")
    legend()
end

function plot_error_distribution(
    grids::Vector{Vector{Float64}},
    error_estimates::Vector{Float64},
)::Nothing
    bar(grids, error_estimates, lw = 2, label = "Error")
    xlabel!("Grid Points")
    ylabel!("Error Estimate")
    title!("Error Distribution")
    legend()
end
