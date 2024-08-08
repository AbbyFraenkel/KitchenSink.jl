"""
    problem_complexity(problem::KSProblem{T,N})::Float64 where {T,N}

Estimate the complexity of a given problem.

# Arguments
- `problem::KSProblem{T,N}`: The problem definition.

# Returns
- `Float64`: The estimated complexity of the problem.
"""
function problem_complexity(problem::KSProblem{T,N})::Float64 where {T,N}
    # Initialize complexity score
    complexity = 0.0

    # 1. Analyze domain complexity
    domain_complexity = analyze_domain_complexity(problem.domain)
    complexity += 0.2 * domain_complexity

    # 2. Analyze equation complexity
    equation_complexity = analyze_equation_complexity(problem.equation)
    complexity += 0.4 * equation_complexity

    # 3. Analyze boundary condition complexity
    bc_complexity = analyze_boundary_condition_complexity(problem.boundary_conditions)
    complexity += 0.2 * bc_complexity

    # 4. Consider problem dimensionality
    dim_factor = 1 - exp(-0.5 * N)  # Increases with dimension, but with diminishing returns
    complexity += 0.2 * dim_factor

    # Ensure complexity is between 0 and 1
    return clamp(complexity, 0.0, 1.0)
end

"""
    analyze_domain_complexity(domain::NTuple{N,Tuple{T,T}}) where {N,T}

Analyze the complexity of the domain.

# Arguments
- `domain::NTuple{N,Tuple{T,T}}`: A tuple representing the domain bounds.

# Returns
- `Float64`: A score representing the domain complexity, normalized between 0 and 1.
"""
function analyze_domain_complexity(domain::NTuple{N,Tuple{T,T}}) where {N,T}
    # Check for non-rectangular domains or large aspect ratios
    aspect_ratios = [abs(d[2] - d[1]) for d in domain]
    max_ratio = maximum(aspect_ratios) / minimum(aspect_ratios)
    return min(log2(max_ratio) / 10, 1.0)  # Normalize to [0, 1]
end

"""
    analyze_equation_complexity(equation::Function)

Analyze the complexity of the equation.

# Arguments
- `equation::Function`: The function representing the equation.

# Returns
- `Float64`: A score representing the equation complexity, normalized between 0 and 1.
"""
function analyze_equation_complexity(equation::Function)
    # Use metaprogramming to analyze the equation's AST
    ast = Base.remove_linenums!(Meta.parse(string(equation)))

    # Count the number of operations and function calls
    op_count = count(x -> x isa Expr && x.head in [:call, :., :(=)], ast)

    # Analyze the types of operations (e.g., derivatives, nonlinear terms)
    has_derivatives = any(x -> x isa Expr && x.head == :call && x.args[1] in [:∂, :∇, :Δ], ast)
    has_nonlinear = any(x -> x isa Expr && x.head == :call && x.args[1] in [:^, :sin, :cos, :exp, :log], ast)

    complexity = log2(op_count + 1) / 10  # Base complexity on operation count
    complexity += has_derivatives ? 0.3 : 0
    complexity += has_nonlinear ? 0.3 : 0

    return min(complexity, 1.0)  # Normalize to [0, 1]
end

"""
    analyze_boundary_condition_complexity(bc::Function)

Analyze the complexity of the boundary conditions.

# Arguments
- `bc::Function`: The function representing the boundary conditions.

# Returns
- `Float64`: A score representing the boundary condition complexity, normalized between 0 and 1.
"""
function analyze_boundary_condition_complexity(bc::Function)
    # Similar to equation complexity, but for boundary conditions
    ast = Base.remove_linenums!(Meta.parse(string(bc)))

    op_count = count(x -> x isa Expr && x.head in [:call, :., :(=)], ast)
    has_nonlinear = any(x -> x isa Expr && x.head == :call && x.args[1] in [:^, :sin, :cos, :exp, :log], ast)

    complexity = log2(op_count + 1) / 10
    complexity += has_nonlinear ? 0.2 : 0

    return min(complexity, 1.0)  # Normalize to [0, 1]
end
