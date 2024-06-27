function CoordinateDescentRidge(
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


function CoordinateDescentLasso(
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

function CoordinateDescentElasticNet(
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
