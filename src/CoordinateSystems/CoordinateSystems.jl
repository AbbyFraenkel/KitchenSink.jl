module CoordinateSystems

using ..KSTypes
using LinearAlgebra

export to_cartesian, from_cartesian, compute_jacobian, map_to_reference_element, map_from_reference_element
export case_mapping
# Helper function to scale a value based on a given range
function scale_value(value::T, range::Union{Tuple{T, T}, Nothing}) where T
    if range === nothing
        return value
    else
        return (value - range[1]) / (range[2] - range[1])
    end
end

# Helper function to unscale a value based on a given range
function unscale_value(scaled_value::T, range::Union{Tuple{T, T}, Nothing}) where T
    if range === nothing
        return scaled_value
    else
        return scaled_value * (range[2] - range[1]) + range[1]
    end
end

# Helper function to map a single coordinate
function map_coordinate(value::T, range::Union{Tuple{T, T}, Nothing}, is_angular::Bool = false) where {T}
    if range === nothing
        return value  # No mapping if the range is not specified
    else
        if is_angular
            wrapped = mod2pi(value - range[1]) / (range[2] - range[1])
            return 2 * wrapped - 1
        else
            return 2 * (value - range[1]) / (range[2] - range[1]) - 1
        end
    end
end

# Helper function to map a single coordinate back from reference
function map_coordinate_back(value::T, range::Union{Tuple{T, T}, Nothing}, is_angular::Bool = false) where T
    if range === nothing
        return value  # No mapping if the range is not specified
    else
        if is_angular
            normalized = (value + 1) / 2
            return normalized * (range[2] - range[1]) + range[1]
        else
            return 0.5 * (value + 1) * (range[2] - range[1]) + range[1]
        end
    end
end

# Updated to_cartesian functions
function to_cartesian(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r, theta = coords
    # Remove scaling for r and theta
    x = r * cos(theta)
    y = r * sin(theta)
    return (x, y)
end

function to_cartesian(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r, theta, phi = coords
    # Remove scaling for r, theta, and phi
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return (x, y, z)
end

function to_cartesian(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r, theta, z = coords
    # Remove scaling for r, theta, and z
    x = r * cos(theta)
    y = r * sin(theta)
    return (x, y, z)
end

# Updated from_cartesian functions
function from_cartesian(cartesian_coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(cartesian_coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    x, y = cartesian_coords
    r = hypot(x, y)
    theta = atan(y, x)
    # Remove unscaling for r and theta
    return (r, theta)
end

function from_cartesian(cartesian_coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(cartesian_coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    x, y, z = cartesian_coords
    r = sqrt(x^2 + y^2 + z^2)
    theta = acos(z / r)
    phi = atan(y, x)
    # Remove unscaling for r, theta, and phi
    return (r, theta, phi)
end

function from_cartesian(cartesian_coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(cartesian_coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    x, y, z = cartesian_coords
    r = hypot(x, y)
    theta = atan(y, x)
    # Remove unscaling for r, theta, and z
    return (r, theta, z)
end

function from_cartesian(coords::Tuple{Vararg{T, N}}, system::KSCartesianCoordinates{T, N}) where {T, N}
    return coords
end

function to_cartesian(coords::Tuple{Vararg{T, N}}, system::KSCartesianCoordinates{T, N}) where {T, N}
    return coords
end

# Jacobian and Mapping Functions
function compute_jacobian(coords::Tuple, system::KSCartesianCoordinates{T, N}) where {T, N}
    return I(N)
end

function map_to_reference_element(coords::Tuple, system::KSCartesianCoordinates{T, N}) where {T, N}
    if length(coords) != N
        throw(ArgumentError("Input coordinates must have $N elements."))
    end
    return ntuple(i -> 2 * (coords[i] - system.ranges[i][1]) / (system.ranges[i][2] - system.ranges[i][1]) - 1, N)
end

function map_from_reference_element(coords::Tuple, system::KSCartesianCoordinates{T, N}) where {T, N}
    if length(coords) != N
        throw(ArgumentError("Input coordinates must have $N elements."))
    end
    return ntuple(i -> 0.5 * (coords[i] + 1) * (system.ranges[i][2] - system.ranges[i][1]) + system.ranges[i][1], N)
end

function compute_jacobian(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r, theta = coords
    return [cos(theta) -r*sin(theta); sin(theta) r*cos(theta)]
end

function map_to_reference_element(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r, theta = coords
    r_ref = map_coordinate(r, system.r)
    theta_ref = map_coordinate(Float64(theta), system.theta, true)
    return (r_ref, theta_ref)
end

function map_from_reference_element(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r_ref, theta_ref = coords
    r = map_coordinate_back(r_ref, system.r)
    theta = map_coordinate_back(theta_ref, system.theta, true)
    return (r, theta)
end

function compute_jacobian(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r, theta, phi = coords
    return [sin(theta)*cos(phi) r*cos(theta)*cos(phi) -r*sin(theta)*sin(phi);
            sin(theta)*sin(phi) r*cos(theta)*sin(phi) r*sin(theta)*cos(phi);
            cos(theta) -r*sin(theta) 0]
end

function map_to_reference_element(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r, theta, phi = coords
    r_ref = map_coordinate(r, system.r)
    theta_ref = map_coordinate(Float64(theta), system.theta, true)
    phi_ref = map_coordinate(Float64(phi), system.phi, true)
    return (r_ref, theta_ref, phi_ref)
end

function map_from_reference_element(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r_ref, theta_ref, phi_ref = coords
    r = map_coordinate_back(r_ref, system.r)
    theta = map_coordinate_back(theta_ref, system.theta, true)
    phi = map_coordinate_back(phi_ref, system.phi, true)
    return (r, theta, phi)
end

function compute_jacobian(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r, theta, z = coords
    return [cos(theta) -r*sin(theta) 0;
            sin(theta) r*cos(theta) 0;
            0 0 1]
end

function map_to_reference_element(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r, theta, z = coords
    r_ref = map_coordinate(r, system.r)
    theta_ref = map_coordinate(Float64(theta), system.theta, true)
    z_ref = map_coordinate(z, system.z)
    return (r_ref, theta_ref, z_ref)
end

function map_from_reference_element(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r_ref, theta_ref, z_ref = coords
    r = map_coordinate_back(r_ref, system.r)
    theta = map_coordinate_back(theta_ref, system.theta, true)
    z = map_coordinate_back(z_ref, system.z)
    return (r, theta, z)
end

function case_mapping(bounds::Tuple{T, T}, coord_system::Symbol) where {T}
    if coord_system == :cartesian
        return map_from_reference_element(bounds, KSCartesianCoordinates((bounds,)))
    elseif coord_system == :polar
        return map_from_reference_element(bounds, KSPolarCoordinates(bounds))
    elseif coord_system == :cylindrical
        return map_from_reference_element(bounds, KSCylindricalCoordinates(bounds))
    elseif coord_system == :spherical
        return map_from_reference_element(bounds, KSSphericalCoordinates(bounds))
    else
        throw(ArgumentError("Unknown coordinate system: $coord_system"))
    end
end
end  # module CoordinateSystems
