function ParallelMeshAssembly(
    ndims::Int,
    domain::Vector{Float64},
    num_elements::Int,
    basis_functions::Vector{Function},
    collocation_points::Vector{Float64},
    weights::Vector{Float64},
    lambda::Float64,
)::Tuple{SparseMatrixCSC{Float64,Int},Vector{Float64}}
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

    A .+= lambda * I

    return A, b
end


function ParallelCoordinateDescentRidge(
    A::Matrix{Float64},
    b::Vector{Float64},
    lambda1::Float64,
    tol::Float64 = 1e-6,
    max_iter::Int = 1000,
)::Vector{Float64}
    m, n = size(A)
    x = zeros(n)
    r = b - A * x

    for iter = 1:max_iter
        Threads.@threads for j = 1:n
            aj = A[:, j]
            x[j] = dot(aj, r) / dot(aj, aj)
            r .+= aj * (x[j] - x[j])
        end
        if norm(A * x - b) < tol
            break
        end
    end

    return x
end


function ParallelCoordinateDescentLasso(
    A::Matrix{Float64},
    b::Vector{Float64},
    lambda2::Float64,
    tol::Float64 = 1e-6,
    max_iter::Int = 1000,
)::Vector{Float64}
    m, n = size(A)
    x = zeros(n)
    r = b - A * x

    for iter = 1:max_iter
        Threads.@threads for j = 1:n
            aj = A[:, j]
            xj_old = x[j]
            rho = dot(aj, r + aj * xj_old)
            if rho < -lambda / 2
                x[j] = (rho + lambda / 2) / dot(aj, aj)
            elseif rho > lambda / 2
                x[j] = (rho - lambda / 2) / dot(aj, aj)
            else
                x[j] = 0
            end
            r .+= aj * (xj_old - x[j])
        end
        if norm(A * x - b) < tol
            break
        end
    end

    return x
end

function ParallelCoordinateDescentElasticNet(
    A::Matrix{Float64},
    b::Vector{Float64},
    lambda1::Float64,
    lambda2::Float64,
    tol::Float64 = 1e-6,
    max_iter::Int = 1000,
)::Vector{Float64}
    m, n = size(A)
    x = zeros(n)
    r = b - A * x

    for iter = 1:max_iter
        Threads.@threads for j = 1:n
            aj = A[:, j]
            xj_old = x[j]
            rho = dot(aj, r + aj * xj_old) + lambda2 * xj_old
            if rho < -lambda1 / 2
                x[j] = (rho + lambda1 / 2) / (dot(aj, aj) + lambda2)
            elseif rho > lambda1 / 2
                x[j] = (rho - lambda1 / 2) / (dot(aj, aj) + lambda2)
            else
                x[j] = 0
            end
            r .+= aj * (xj_old - x[j])
        end
        if norm(A * x - b) < tol
            break
        end
    end

    return x
end
