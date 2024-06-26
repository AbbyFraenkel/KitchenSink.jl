
function Parallel_assemble_system(problem::Problem)::System
    num_basis = length(problem.basis_functions)
    A = spzeros(Float64, problem.num_elements * num_basis, problem.num_elements * num_basis)
    b = zeros(Float64, problem.num_elements * num_basis)

    Threads.@threads for i in 1:problem.num_elements
        for j in 1:num_basis
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

function Parallel_compute_error_estimates(
    solution::Vector{Float64},
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
)::Vector{Float64}
    error_estimates = zeros(Float64, length(solution))

    Threads.@threads for i in 1:length(solution)
        error_estimates[i] =
            norm(solution[i] - basis_functions[i](collocation_points[i])) * weights[i]
    end

    return error_estimates
end
