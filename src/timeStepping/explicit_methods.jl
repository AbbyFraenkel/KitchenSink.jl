# """
#     forward_euler!(u, t, dt, f)

# Forward Euler method for time-stepping.

# # Arguments
# - `u`: State variable.
# - `t`: Current time.
# - `dt`: Time step.
# - `f`: Function representing the derivative of `u`.

# # Returns
# - Updated state variable after one time step.
# """
# function forward_euler!(u::AbstractVector, t::Real, dt::Real, f::Function)
#     return u + dt * f(u, t)
# end

# """
#     rk4!(u, t, dt, f)

# Runge-Kutta 4th order method for time-stepping.

# # Arguments
# - `u`: State variable.
# - `t`: Current time.
# - `dt`: Time step.
# - `f`: Function representing the derivative of `u`.

# # Returns
# - Updated state variable after one time step.
# """
# function rk4!(u::AbstractVector, t::Real, dt::Real, f::Function)
#     k1 = dt * f(u, t)
#     k2 = dt * f(u + 0.5 * k1, t + 0.5 * dt)
#     k3 = dt * f(u + 0.5 * k2, t + 0.5 * dt)
#     k4 = dt * f(u + k3, t + dt)
#     return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
# end

# """
#     rk45!(u, t, dt, f)

# Runge-Kutta-Fehlberg (RK45) method for adaptive time-stepping.

# # Arguments
# - `u`: State variable.
# - `t`: Current time.
# - `dt`: Initial time step.
# - `f`: Function representing the derivative of `u`.

# # Returns
# - `u_new`: Updated state variable.
# - `dt`: Adjusted time step based on error estimate.
# """
# function rk45!(u::AbstractVector, t::Real, dt::Real, f::Function)
#     # Coefficients for RK45
#     c2, c3, c4, c5, c6 = 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2
#     a21 = 1 / 4
#     a31, a32 = 3 / 32, 9 / 32
#     a41, a42, a43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
#     a51, a52, a53, a54 = 439 / 216, -8, 3680 / 513, -845 / 4104
#     a61, a62, a63, a64, a65 = -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40
#     b1, b2, b3, b4, b5, b6 = 16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55
#     b1s, b2s, b3s, b4s, b5s, b6s = 25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0

#     k1 = dt * f(u, t)
#     k2 = dt * f(u + a21 * k1, t + c2 * dt)
#     k3 = dt * f(u + a31 * k1 + a32 * k2, t + c3 * dt)
#     k4 = dt * f(u + a41 * k1 + a42 * k2 + a43 * k3, t + c4 * dt)
#     k5 = dt * f(u + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, t + c5 * dt)
#     k6 = dt * f(u + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, t + c6 * dt)

#     u_new = u + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
#     u_new_s = u + b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6

#     error_estimate = norm(u_new - u_new_s, Inf)
#     return u_new, error_estimate
# end



# function adams_bashforth!(u::AbstractVector{T}, t::T, dt::T, f::Function, order::Int, history::Vector{Vector{T}}) where T<:Real
#     coefficients = get_adams_bashforth_coefficients(order)
#     Threads.@threads for i in 1:length(u)
#         u[i] += dt * sum(coefficients[j] * history[j][i] for j in 1:order)
#     end
#     return u
# end

# function get_adams_bashforth_coefficients(order::Int)
#     if order == 1
#         return [1]
#     elseif order == 2
#         return [3/2, -1/2]
#     elseif order == 3
#         return [23/12, -16/12, 5/12]
#     elseif order == 4
#         return [55/24, -59/24, 37/24, -9/24]
#     else
#         error("Adams-Bashforth method of order $order not implemented")
#     end
# end
