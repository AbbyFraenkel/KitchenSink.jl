
"""
    v_cycle(hierarchy::Vector{KSMesh{T,N}}, f::Function, u::Vector{T}, level::Int) where {T<:Real,N}

Perform a V-cycle for the multi-level method.

# Arguments
- `hierarchy::Vector{KSMesh{T,N}}`: The mesh hierarchy.
- `f::Function`: The right-hand side function.
- `u::Vector{T}`: The current solution vector.
- `level::Int`: The current level in the hierarchy.

# Returns
- `Vector{T}`: The updated solution vector.

# Examples
```julia
updated_solution = v_cycle(hierarchy, f, initial_solution, 3)
```
"""
function v_cycle(hierarchy::Vector{KSMesh{T,N}}, f::Function, u::Vector{T}, level::Int) where {T<:Real,N}
    if level == 1
        return solve_coarse_problem(hierarchy[1], f)
    else
        u = smooth(hierarchy[level], f, u)
        r = compute_residual(hierarchy[level], f, u)
        r_coarse = restrict(r, hierarchy[level], hierarchy[level-1])
        e_coarse = v_cycle(hierarchy, x -> r_coarse, zeros(length(r_coarse)), level - 1)
        u += prolongate(e_coarse, hierarchy[level-1], hierarchy[level])
        return smooth(hierarchy[level], f, u)
    end
end


"""
    w_cycle(hierarchy::Vector{<:AbstractKSMesh}, f::Function, u::AbstractVector, level::Int) -> AbstractVector

Perform a W-cycle for the multi-level method.

# Arguments
- `hierarchy::Vector{<:AbstractKSMesh}`: The mesh hierarchy
- `f::Function`: The right-hand side function
- `u::AbstractVector`: The current solution vector
- `level::Int`: The current level in the hierarchy

# Returns
- `AbstractVector`: The updated solution vector

# Example
```julia
u_new = w_cycle(hierarchy, f, u, 3)
```
"""
function w_cycle(hierarchy::Vector{<:AbstractKSMesh}, f::Function, u::AbstractVector, level::Int)
    if level == 1
        return solve_coarse_problem(hierarchy[1], f)
    else
        u = smooth(hierarchy[level], f, u)
        r = compute_residual(hierarchy[level], f, u)
        r_coarse = restrict(r, hierarchy[level], hierarchy[level-1])
        e_coarse = w_cycle(hierarchy, x -> r_coarse, zeros(length(r_coarse)), level - 1)
        e_coarse = w_cycle(hierarchy, x -> r_coarse, e_coarse, level - 1)
        u += prolongate(e_coarse, hierarchy[level-1], hierarchy[level])
        return smooth(hierarchy[level], f, u)
    end
end

"""
    full_multigrid(hierarchy::Vector{<:AbstractKSMesh}, f::Function) -> AbstractVector

Perform a full multigrid cycle.

# Arguments
- `hierarchy::Vector{<:AbstractKSMesh}`: The mesh hierarchy
- `f::Function`: The right-hand side function

# Returns
- `AbstractVector`: The solution vector

# Example
```julia
u = full_multigrid(hierarchy, f)
```
"""
function full_multigrid(hierarchy::Vector{<:AbstractKSMesh}, f::Function)
    u = solve_coarse_problem(hierarchy[1], f)
    for level in 2:length(hierarchy)
        u = prolongate(u, hierarchy[level-1], hierarchy[level])
        u = v_cycle(hierarchy[1:level], f, u, level)
    end
    return u
end


"""
    geometric_multigrid(A::AbstractMatrix, b::AbstractVector, mesh_hierarchy::Vector{KSMesh};
                        cycle_type::Symbol=:v, max_iterations::Int=100, tolerance::Float64=1e-8) -> AbstractVector

Solves the linear system `Ax = b` using the geometric multigrid method.

# Arguments
- `A::AbstractMatrix`: The coefficient matrix.
- `b::AbstractVector`: The right-hand side vector.
- `mesh_hierarchy::Vector{KSMesh}`: A vector representing the mesh hierarchy for the multigrid method.
- `cycle_type::Symbol`: The type of multigrid cycle to use (`:v`, `:w`, or `:fmg`). Default is `:v`.
- `max_iterations::Int`: The maximum number of iterations to perform. Default is 100.
- `tolerance::Float64`: The convergence tolerance. Default is `1e-8`.

# Returns
- `AbstractVector`: The solution vector `x`.

# Description
This function applies the geometric multigrid method to solve the linear system `Ax = b`. It supports three types of multigrid cycles:
- `:v` for V-cycle
- `:w` for W-cycle
- `:fmg` for Full Multigrid (FMG)

The function iteratively refines the solution `x` until the residual `r = b - A * x` is less than the specified `tolerance` or the maximum number of iterations `max_iterations` is reached.

# Example
```julia
A = [4.0 -1.0 0.0; -1.0 4.0 -1.0; 0.0 -1.0 3.0]
b = [15.0, 10.0, 10.0]
mesh_hierarchy = [KSMesh(...), KSMesh(...)]  # Example mesh hierarchy
x = geometric_multigrid(A, b, mesh_hierarchy, cycle_type=:v, max_iterations=100, tolerance=1e-8)
Notes
The function will log an info message if it converges within the specified iterations.
If the function does not converge within the specified iterations, it will log a warning message. 
"""

function geometric_multigrid(A::AbstractMatrix, b::AbstractVector, mesh_hierarchy::Vector{KSMesh},
    cycle_type::Symbol=:v, max_iterations::Int=100, tolerance::Float64=1e-8)::AbstractVector
    x = zeros(length(b))
    r = b - A * x

    for iter in 1:max_iterations
        if cycle_type == :v
            x = v_cycle(mesh_hierarchy, t -> b - A * t, x, length(mesh_hierarchy))
        elseif cycle_type == :w
            x = w_cycle(mesh_hierarchy, t -> b - A * t, x, length(mesh_hierarchy))
        elseif cycle_type == :fmg
            x = full_multigrid(mesh_hierarchy, t -> b - A * t)
        else
            error("Unknown cycle type: $cycle_type")
        end

        r = b - A * x
        if norm(r) < tolerance
            @info "Converged in $iter iterations"
            return x
        end
    end

    @warn "Did not converge within $max_iterations iterations"
    return x
end


"""
    algebraic_multigrid(A::AbstractMatrix, b::AbstractVector, max_iterations::Int=100, tolerance::Float64=1e-8)::AbstractVector

Performs algebraic multigrid method to solve a linear system of equations.

# Arguments
- `A::AbstractMatrix`: The coefficient matrix of the linear system.
- `b::AbstractVector`: The right-hand side vector of the linear system.
- `max_iterations::Int=100`: The maximum number of iterations (default: 100).
- `tolerance::Float64=1e-8`: The tolerance for convergence (default: 1e-8).

# Returns
- `x::AbstractVector`: The solution vector.

"""
function algebraic_multigrid(A::AbstractMatrix, b::AbstractVector, max_iterations::Int=100, tolerance::Float64=1e-8)::AbstractVector
    ml = ruge_stuben(A)
    x, convergence_history = solve(ml, b, maxiter=max_iterations, tol=tolerance, log=true)

    if convergence_history.converged
        @info "Converged in $(convergence_history.niter) iterations"
    else
        @warn "Did not converge within $max_iterations iterations"
    end

    return x
end
