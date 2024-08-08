
using Test, LinearAlgebra, SparseArrays
using ..KSTypes, ..MultiLevelMethods, ..SpectralMethods, ..AdaptiveMethods, ..Preprocessing, ..CoordinateSystems

# Helper function to create a test problem
function create_test_problem(dim::Int, coord_system::Symbol)
    if dim == 1
        domain = ((0.0, 1.0),)
        f(x) = sin(π * x[1])
        u_exact(x) = sin(π * x[1]) / π^2
        bc(x) = 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{1,Float64}((0.0,)) :
                coord_system == :polar ? KSPolarCoordinates{Float64}(1.0, 0.0) :
                error("Unsupported 1D coordinate system")
    elseif dim == 2
        domain = ((0.0, 1.0), (0.0, 1.0))
        f(x) = 2π^2 * sin(π * x[1]) * sin(π * x[2])
        u_exact(x) = sin(π * x[1]) * sin(π * x[2])
        bc(x) = 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{2,Float64}((0.0, 0.0)) :
                coord_system == :polar ? KSPolarCoordinates{Float64}(1.0, 0.0) :
                error("Unsupported 2D coordinate system")
    elseif dim == 3
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        f(x) = 3π^2 * sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        u_exact(x) = sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
        bc(x) = 0.0
        coord = coord_system == :cartesian ? KSCartesianCoordinates{3,Float64}((0.0, 0.0, 0.0)) :
                coord_system == :spherical ? KSSphericalCoordinates{Float64}(1.0, 0.0, 0.0) :
                error("Unsupported 3D coordinate system")
    else
        error("Unsupported dimension")
    end
    return KSProblem(x -> -sum(∇²(x)), domain, bc, coord), f, u_exact
end




function create_test_mesh(num_elements::Int, degree::Int)
    nodes, _ = create_nodes(degree + 1, 0.0, 1.0)
    elements = [KSElement(i, [KSPoint([nodes[j]]) for j in i:i+1], [], nothing, nothing, nothing, 0, degree, 0.0) for i in 1:num_elements]
    return KSMesh(elements, [], [], [], 0.0, 1)
end
