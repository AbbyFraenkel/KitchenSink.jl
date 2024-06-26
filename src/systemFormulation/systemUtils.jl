@inline function validate_inputs(ndims::Int, p::Int)::Nothing
    @assert ndims > 0 "Number of dimensions must be positive"
    @assert p > 0 "Polynomial order must be positive"
end
@inline function initial_state(control::Vector{Float64})::Vector{Float64}
    return zeros(size(control))
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

