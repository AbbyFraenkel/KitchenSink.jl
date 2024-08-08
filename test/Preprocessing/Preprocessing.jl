using Test, LinearAlgebra, SparseArrays
# using ..KSTypes, ..Preprocessing, ..SpectralMethods, ..CoordinateSystems
using .KSTypes, .Preprocessing, .SpectralMethods, .CoordinateSystems

# Improved create_test_problem function
function create_test_problem(dim::Int, coord_system::Symbol)
    if dim == 1
        domain = ((0.0, 1.0),)
        f = x -> sin(π * x[1])
        u_exact = x -> sin(π * x[1]) / π^2
        bc = x -> 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{1,Float64}((0.0,)) :
                coord_system == :polar ? KSPolarCoordinates{Float64}(1.0, 0.0) :
                error("Unsupported 1D coordinate system")
        return KSProblem(x -> -sum(x .^ 2), domain, bc, coord), f, u_exact
    elseif dim == 2
        domain = ((0.0, 1.0), (0.0, 1.0))
        f = x -> 2π^2 * sin(π * x[1]) * sin(π * x[2])
        u_exact = x -> sin(π * x[1]) * sin(π * x[2])
        bc = x -> 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{2,Float64}((0.0, 0.0)) :
                coord_system == :polar ? KSPolarCoordinates{Float64}(1.0, 0.0) :
                error("Unsupported 2D coordinate system")
        return KSProblem(x -> -sum(x .^ 2), domain, bc, coord), f, u_exact
    elseif dim == 3
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        f = x -> 3π^2 * sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        u_exact = x -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        bc = x -> 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{3,Float64}((0.0, 0.0, 0.0)) :
                coord_system == :spherical ? KSSphericalCoordinates{Float64}(1.0, 0.0, 0.0) :
                error("Unsupported 3D coordinate system")
        return KSProblem(x -> -sum(x .^ 2), domain, bc, coord), f, u_exact
    else
        error("Unsupported dimension")
    end
end

@testset "Preprocessing" begin
    include("discretization.jl")
    include("mesh_creation.jl")
    include("tensor_product_masks.jl")
end
