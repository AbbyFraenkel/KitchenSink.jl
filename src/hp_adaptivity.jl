module HPAdaptivity

using ..Types
using LinearAlgebra
using SparseArrays
using ..ErrorEstimation
using ..OCFE

export hp_adaptivity_solver,
    goal_oriented_refinement, refine_by_superposition, compute_smoothness_indicator

function hp_adaptivity_solver(
    problem::Problem,
    solver::Solver,
    initial_solution::Vector{Float64},
)::Vector{Float64}
    solution = initial_solution
    for iter = 1:solver.max_iterations
        system = assemble_system(problem)
        solution = system.A \ system.b
        error_estimates = compute_error_estimates(
            solution,
            problem.basis_functions,
            problem.collocation_points,
            problem.weights,
        )
        if maximum(error_estimates) < solver.tolerance
            break
        end
        solution = refine_by_superposition(
            problem.domain,
            problem.num_elements,
            solution,
            length(problem.basis_functions),
            length(problem.basis_functions) + 1,
        )
    end
    return solution
end

function goal_oriented_refinement(
    elements::Vector{Element},
    goal_quantity::Function,
    tolerance::Float64,
)::Vector{Element}
    refined_elements = Element[]
    for el in elements
        error = estimate_error(el, goal_quantity)
        if error > tolerance
            append!(refined_elements, refine_element(el))
        else
            push!(refined_elements, el)
        end
    end
    return refined_elements
end

@inline function compute_smoothness_indicator(coefficients::Vector{Float64})::Float64
    decay_rate = norm(coefficients[end]) / norm(coefficients[1])
    return decay_rate
end

@inline function determine_refinement_strategy(
    smoothness_indicator::Float64,
    threshold::Float64,
)::Symbol
    if smoothness_indicator < threshold
        return :h_refinement
    else
        return :p_refinement
    end
end

function assemble_system(problem::Problem)::System
    num_basis = length(problem.basis_functions)
    A = spzeros(Float64, problem.num_elements * num_basis, problem.num_elements * num_basis)
    b = zeros(Float64, problem.num_elements * num_basis)

    Threads.@threads for i = 1:problem.num_elements
        for j = 1:num_basis
            A[i, j] = integrate_basis(
                length(problem.domain),
                problem.domain,
                problem.basis_functions[i],
                problem.basis_functions[j],
                problem.collocation_points,
                problem.weights,
            )
            b[i] = integrate_forcing(
                length(problem.domain),
                problem.domain,
                problem.basis_functions[i],
                problem.collocation_points,
                problem.weights,
            )
        end
    end

    return System(A, b)
end

function compute_error_estimates(
    solution::Vector{Float64},
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
)::Vector{Float64}
    error_estimates = zeros(Float64, length(solution))

    Threads.@threads for i = 1:length(solution)
        error_estimates[i] =
            norm(solution[i] - basis_functions[i](collocation_points[i])) * weights[i]
    end

    return error_estimates
end

function refine_element(el::Element)::Vector{Element}
    new_elements = [
        Element(
            el.nodes,
            el.basis_functions,
            el.active_basis_indices,
            el.parent,
            el.level + 1,
        ) for _ = 1:2
    ]
    return new_elements
end

end # module
