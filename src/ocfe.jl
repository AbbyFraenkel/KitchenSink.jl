module OCFE

export hierarchical_basis_functions,
    assemble_system,
    assemble_system_with_delay,
    assemble_system_with_integral,
    apply_boundary_conditions!,
    refine_by_superposition,
    compute_smoothness_indicator,
    determine_refinement_strategy,
    interpolate_solution

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using ..Types

@inline function validate_inputs(ndims::Int, p::Int)::Nothing
    @assert ndims > 0 "Number of dimensions must be positive"
    @assert p > 0 "Polynomial order must be positive"
end

function hierarchical_basis_functions(
    ndims::Int,
    p::Int,
)::Tuple{Vector{Function},Vector{Float64},Vector{Float64}}
    validate_inputs(ndims, p)
    roots1d, weights1d = legendre(p)
    roots = collect(CartesianIndex(ntuple(_ -> roots1d, ndims))...)
    weights = prod(ntuple(_ -> weights1d, ndims))
    basis_functions = Vector{Function}()
    for idx in CartesianIndices(ntuple(_ -> p, ndims))
        push!(
            basis_functions,
            x -> prod(
                (x[d] .- roots1d[j]) ./ (roots1d[idx[d]] .- roots1d[j]) for d = 1:ndims
                for j = 1:p if idx[d] != j
            ),
        )
    end
    return basis_functions, roots, weights
end

function assemble_system(
    ndims::Int,
    domain::Vector{Float64},
    num_elements::Int,
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    validate_inputs(ndims, num_elements)
    num_basis = length(basis_functions)
    num_dofs = num_elements^ndims * num_basis
    A = spzeros(Float64, num_dofs, num_dofs)
    b = zeros(Float64, num_dofs)

    Threads.@threads for el in CartesianIndices(ntuple(_ -> num_elements, ndims))
        for i = 1:num_basis
            for j = 1:num_basis
                A[
                    linear_index(el, i, num_elements, num_basis),
                    linear_index(el, j, num_elements, num_basis),
                ] += integrate_basis(
                    ndims,
                    domain,
                    basis_functions[i],
                    basis_functions[j],
                    collocation_points,
                    weights,
                )
            end
            b[linear_index(el, i, num_elements, num_basis)] += integrate_forcing(
                ndims,
                domain,
                basis_functions[i],
                collocation_points,
                weights,
            )
        end
    end

    check_degrees_of_freedom(A, b, num_dofs)

    return A, b
end

function assemble_system_with_delay(
    ndims::Int,
    domain::Vector{Float64},
    num_elements::Int,
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    f::Function,
    τ::Float64,
    history::Function,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    validate_inputs(ndims, num_elements)
    num_basis = length(basis_functions)
    num_dofs = num_elements^ndims * num_basis
    A = spzeros(Float64, num_dofs, num_dofs)
    b = zeros(Float64, num_dofs)

    Threads.@threads for el in CartesianIndices(ntuple(_ -> num_elements, ndims))
        for i = 1:num_basis
            for j = 1:num_basis
                A[
                    linear_index(el, i, num_elements, num_basis),
                    linear_index(el, j, num_elements, num_basis),
                ] += integrate_basis(
                    ndims,
                    domain,
                    basis_functions[i],
                    basis_functions[j],
                    collocation_points,
                    weights,
                )
            end
            b[linear_index(el, i, num_elements, num_basis)] += integrate_forcing_with_delay(
                ndims,
                domain,
                basis_functions[i],
                collocation_points,
                weights,
                f,
                τ,
                history,
            )
        end
    end

    check_degrees_of_freedom(A, b, num_dofs)

    return A, b
end

function assemble_system_with_integral(
    ndims::Int,
    domain::Vector{Float64},
    num_elements::Int,
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    f::Function,
    K::Function,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    validate_inputs(ndims, num_elements)
    num_basis = length(basis_functions)
    num_dofs = num_elements^ndims * num_basis
    A = spzeros(Float64, num_dofs, num_dofs)
    b = zeros(Float64, num_dofs)

    Threads.@threads for el in CartesianIndices(ntuple(_ -> num_elements, ndims))
        for i = 1:num_basis
            for j = 1:num_basis
                A[
                    linear_index(el, i, num_elements, num_basis),
                    linear_index(el, j, num_elements, num_basis),
                ] += integrate_basis(
                    ndims,
                    domain,
                    basis_functions[i],
                    basis_functions[j],
                    collocation_points,
                    weights,
                )
            end
            b[linear_index(el, i, num_elements, num_basis)] +=
                integrate_forcing_with_integral(
                    ndims,
                    domain,
                    basis_functions[i],
                    collocation_points,
                    weights,
                    f,
                    K,
                )
        end
    end

    check_degrees_of_freedom(A, b, num_dofs)

    return A, b
end

@inline function linear_index(
    el::CartesianIndex,
    i::Int,
    num_elements::Int,
    num_basis::Int,
)::Int
    index = (el[1] - 1) * num_elements^(length(el) - 1) * num_basis + i
    for k = 2:length(el)
        index += (el[k] - 1) * num_elements^(length(el) - k) * num_basis
    end
    return index
end

@inline function integrate_basis(
    ndims::Int,
    domain::Vector{Float64},
    phi_i::Function,
    phi_j::Function,
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
)::Float64
    integral = 0.0
    for k = 1:length(collocation_points)
        pt = collocation_points[k]
        w = weights[k]
        integral += w * phi_i(pt) * phi_j(pt)
    end
    return integral
end

@inline function integrate_forcing(
    ndims::Int,
    domain::Vector{Float64},
    phi_i::Function,
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
)::Float64
    integral = 0.0
    for k = 1:length(collocation_points)
        pt = collocation_points[k]
        w = weights[k]
        integral += w * f(pt) * phi_i(pt)
    end
    return integral
end

function integrate_forcing_with_delay(
    ndims::Int,
    domain::Vector{Float64},
    phi_i::Function,
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    f::Function,
    τ::Float64,
    history::Function,
)::Float64
    integral = 0.0
    for k = 1:length(collocation_points)
        pt = collocation_points[k]
        w = weights[k]
        t_delayed = pt - τ
        u_delayed =
            t_delayed >= domain[1] ? interpolate_solution(t_delayed) : history(t_delayed)
        integral += w * f(pt) * phi_i(pt)
    end
    return integral
end

function integrate_forcing_with_integral(
    ndims::Int,
    domain::Vector{Float64},
    phi_i::Function,
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    f::Function,
    K::Function,
)::Float64
    integral = 0.0
    for k = 1:length(collocation_points)
        pt = collocation_points[k]
        w = weights[k]
        integral += w * f(pt) * phi_i(pt)
        for l = 1:k
            s = collocation_points[l]
            ws = weights[l]
            integral += ws * w * K(pt, s) * phi_i(pt)
        end
    end
    return integral
end

@inline function barycentric_weights(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    w = ones(Float64, n)
    for j = 1:n
        for k = 1:n
            if j != k
                w[j] /= (x[j] - x[k])
            end
        end
    end
    return w
end

@inline function barycentric_lagrange_interpolation(
    t::Float64,
    x::Vector{Float64},
    y::Vector{Float64},
    w::Vector{Float64},
)::Float64
    num = 0.0
    denom = 0.0
    for i = 1:length(x)
        if t == x[i]
            return y[i]
        else
            term = w[i] / (t - x[i])
            num += term * y[i]
            denom += term
        end
    end
    return num / denom
end

function interpolate_solution(
    t::Float64,
    known_points::Vector{Float64},
    known_solutions::Vector{Float64},
    w::Vector{Float64},
)::Float64
    return barycentric_lagrange_interpolation(t, known_points, known_solutions, w)
end

function apply_boundary_conditions!(
    A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    boundary_conditions::Vector{Tuple{Int,Float64}},
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
    for (idx, value) in boundary_conditions
        @assert 1 <= idx <= length(b) "Boundary condition index out of range"
        A[idx, :] .= 0
        A[idx, idx] = 1
        b[idx] = value
    end
    return A, b
end

function refine_by_superposition(
    domain::Vector{Float64},
    num_elements::Int,
    solution::Vector{Float64},
    p_old::Int,
    p_new::Int,
)::Vector{Float64}
    old_basis_functions, _, _ = hierarchical_basis_functions(length(domain), p_old)
    new_basis_functions, _, _ = hierarchical_basis_functions(length(domain), p_new)
    basis_functions = vcat(old_basis_functions, new_basis_functions)
    collocation_points, weights = legendre(p_new + 1)
    A, b = assemble_system(
        length(domain),
        domain,
        num_elements,
        basis_functions,
        collocation_points,
        weights,
    )
    boundary_conditions = []
    A, b = apply_boundary_conditions!(A, b, boundary_conditions)
    A, b = ensure_higher_order_continuity!(A, b, num_elements, length(new_basis_functions))
    new_solution = A \ b
    refined_solution = solution + new_solution[end-length(new_basis_functions)+1:end]
    return refined_solution
end

@inline function check_degrees_of_freedom(
    A::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64},
    num_dofs::Int,
)::Nothing
    @assert size(A, 1) == num_dofs "System matrix row size does not match number of degrees of freedom"
    @assert size(A, 2) == num_dofs "System matrix column size does not match number of degrees of freedom"
    @assert length(b) == num_dofs "Right-hand side vector size does not match number of degrees of freedom"
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

end # module OCFE
