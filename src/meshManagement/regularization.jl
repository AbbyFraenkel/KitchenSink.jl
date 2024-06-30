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


function find_optimal_lambda(x, y, λ_range, tolerance=1e-6)
    # Base case: if the range is small enough, return the middle value
    if length(λ_range) <= 3
        min_mse = Inf
        best_λn = λ_range[1]
        for λn in λ_range
            # ToDo: update sot that it calls a regularization function type
            mse = calculate_mse(x, y, 10.0^λn)
            if mse < min_mse
                min_mse = mse
                best_λn = λn
            end
        end
        return best_λn, min_mse
    end
    N = length(λ_range)

    # Define the start and end indices for the middle, left, and right ranges
    left_end = Int(floor(N / 3))
    middle_start = left_end + 1
    middle_end = N - left_end
    right_start = middle_end + 1

    # Define the middle, left, and right ranges
    left_range = λ_range[1:left_end]
    middle_range = λ_range[middle_start:middle_end]
    right_range = λ_range[right_start:end]

    # Calculate the MSE for the middle value in the range
    λn_middle = middle_range[div(length(middle_range), 2)+1]
    # λ_middle = 10.0^(λn_middle)
    # middle_range = λ_range[div(N, 2) - 1 : div(N, 2) + 1]

    mse_middle = calculate_mse(y, y_pred_middle)

    # Calculate the MSE for the point to the left of λn_middle
    λn_left = left_range[div(length(left_range), 2)+1]
    mse_left = calculate_mse(x, y, 10.0^λn_left)

    # Calculate the MSE for the point to the right of λn_middle
    λn_right = right_range[div(length(right_range), 2)+1]
    mse_right = calculate_mse(x, y, 10.0^λn_right)

    # Recursively apply the process to the half of the range that produced the smallest SSE
    if mse_middle <= min(mse_left, mse_right) + tolerance
        return find_optimal_lambda(x, y, middle_range, tolerance)
    elseif mse_left < mse_right
        return find_optimal_lambda(x, y, left_range, tolerance)
    else
        return find_optimal_lambda(x, y, right_range, tolerance)
    end
end
