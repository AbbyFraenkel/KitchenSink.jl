module BoundaryConditions

using SparseArrays

export applyRobinBoundaryCondition, applyNeumannBoundaryCondition, applyDirichletBoundaryCondition

"""
    applyRobinBoundaryCondition(D::SparseMatrixCSC, F::Vector, h::Float64, k::Float64, g::Float64, location::Int)

Apply the Robin boundary condition at the specified location.

# Arguments
- `D::SparseMatrixCSC`: Coefficient matrix.
- `F::Vector`: Right-hand side vector.
- `h::Float64`: Robin condition coefficient.
- `k::Float64`: Robin condition coefficient.
- `g::Float64`: Source term.
- `location::Int`: Location index.

# Returns
- `D::SparseMatrixCSC`, `F::Vector`: Updated coefficient matrix and right-hand side vector.
"""
function applyRobinBoundaryCondition(D::SparseMatrixCSC, F::Vector, h::Float64, k::Float64, g::Float64, location::Int)
    D[location, location] += h + k / (mesh[location+1, 1] - mesh[location, 1])
    F[location] += g
    return D, F
end

"""
    applyNeumannBoundaryCondition(D::SparseMatrixCSC, F::Vector, q::Float64, location::Int)

Apply the Neumann boundary condition at the specified location.

# Arguments
- `D::SparseMatrixCSC`: Coefficient matrix.
- `F::Vector`: Right-hand side vector.
- `q::Float64`: Specified heat flux.
- `location::Int`: Location index.

# Returns
- `D::SparseMatrixCSC`, `F::Vector`: Updated coefficient matrix and right-hand side vector.
"""
function applyNeumannBoundaryCondition(D::SparseMatrixCSC, F::Vector, q::Float64, location::Int)
    D[location, (location-1):location] = [1, -1]
    F[location] += q
    return D, F
end

"""
    applyDirichletBoundaryCondition(D::SparseMatrixCSC, F::Vector, u::Float64, location::Int)

Apply the Dirichlet boundary condition at the specified location.

# Arguments
- `D::SparseMatrixCSC`: Coefficient matrix.
- `F::Vector`: Right-hand side vector.
- `u::Float64`: Specified value of the dependent variable.
- `location::Int`: Location index.

# Returns
- `D::SparseMatrixCSC`, `F::Vector`: Updated coefficient matrix and right-hand side vector.
"""
function applyDirichletBoundaryCondition(D::SparseMatrixCSC, F::Vector, u::Float64, location::Int)
    D[location, :] = 0
    D[location, location] = 1
    F[location] = u
    return D, F
end

end
