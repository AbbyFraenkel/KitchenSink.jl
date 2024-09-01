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


# Function to convert to Cartesian coordinates
function to_cartesian(point::Tuple, coords::KSCartesianCoordinates)
    if length(point) != length(coords.ranges)
        throw(ArgumentError("Input coordinates must have $(length(coords.ranges)) elements."))
    end
    result = ntuple(i -> coords.active[i] ? point[i] : NaN, length(point))
    return result
end

# Function to convert from Cartesian coordinates
function from_cartesian(point::Tuple, coords::KSCartesianCoordinates)
    if length(point) != length(coords.ranges)
        throw(ArgumentError("Input coordinates must have $(length(coords.ranges)) elements."))
    end
    result = ntuple(i -> coords.active[i] ? point[i] : NaN, length(point))
    return result
end

# Polar to Cartesian
function to_cartesian(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r, theta = coords
    if !system.active[1]
        r = NaN
    end
    if !system.active[2]
        theta = NaN
    end
    x = r * cos(theta)
    y = r * sin(theta)
    return (x, y)
end

function from_cartesian(cartesian_coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(cartesian_coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    x, y = cartesian_coords
    r = hypot(x, y)
    theta = atan(y, x)
    if !system.active[1]
        r = NaN
    end
    if !system.active[2]
        theta = NaN
    end
    return (r, theta)
end

# Spherical to Cartesian
function to_cartesian(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r, theta, phi = coords
    if !system.active[1]
        r = NaN
    end
    if !system.active[2]
        theta = NaN
    end
    if !system.active[3]
        phi = NaN
    end
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return (x, y, z)
end

function from_cartesian(cartesian_coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(cartesian_coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    x, y, z = cartesian_coords
    r = sqrt(x^2 + y^2 + z^2)
    theta = acos(z / r)
    phi = atan(y, x)
    if !system.active[1]
        r = NaN
    end
    if !system.active[2]
        theta = NaN
    end
    if !system.active[3]
        phi = NaN
    end
    return (r, theta, phi)
end

# Cylindrical to Cartesian
function to_cartesian(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r, theta, z = coords
    if !system.active[1]
        r = NaN
    end
    if !system.active[2]
        theta = NaN
    end
    if !system.active[3]
        z = NaN
    end
    x = r * cos(theta)
    y = r * sin(theta)
    return (x, y, z)
end

function from_cartesian(cartesian_coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(cartesian_coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    x, y, z = cartesian_coords
    r = hypot(x, y)
    theta = atan(y, x)
    if !system.active[1]
        r = NaN
    end
    if !system.active[2]
        theta = NaN
    end
    if !system.active[3]
        z = NaN
    end
    return (r, theta, z)
end

# Jacobian and Mapping Functions

function compute_jacobian(coords::Tuple, system::KSCartesianCoordinates{T, N}) where {T, N}
    return I(N)
end

function compute_jacobian(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r, theta = coords
    if !system.active[1] || !system.active[2]
        return zeros(T, 2, 2)
    end
    return [cos(theta) -r*sin(theta); sin(theta) r*cos(theta)]
end

function compute_jacobian(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r, theta, phi = coords
    if !system.active[1] || !system.active[2] || !system.active[3]
        return zeros(T, 3, 3)
    end
    return [sin(theta)*cos(phi) r*cos(theta)*cos(phi) -r*sin(theta)*sin(phi);
            sin(theta)*sin(phi) r*cos(theta)*sin(phi) r*sin(theta)*cos(phi);
            cos(theta) -r*sin(theta) 0]
end

function compute_jacobian(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r, theta, z = coords
    if !system.active[1] || !system.active[2] || !system.active[3]
        return zeros(T, 3, 3)
    end
    return [cos(theta) -r*sin(theta) 0;
            sin(theta) r*cos(theta) 0;
            0 0 1]
end

# Map to/from reference element with inactive domain handling

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

function map_to_reference_element(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r, theta = coords
    if !system.active[1]
        r_ref = NaN
    else
        r_ref = map_coordinate(r, system.r)
    end
    if !system.active[2]
        theta_ref = NaN
    else
        theta_ref = map_coordinate(Float64(theta), system.theta, true)
    end
    return (r_ref, theta_ref)
end

function map_from_reference_element(coords::Tuple, system::KSPolarCoordinates{T}) where T
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have 2 elements for polar coordinates."))
    end
    r_ref, theta_ref = coords
    if !system.active[1]
        r = NaN
    else
        r = map_coordinate_back(r_ref, system.r)
    end
    if !system.active[2]
        theta = NaN
    else
        theta = map_coordinate_back(theta_ref, system.theta, true)
    end
    return (r, theta)
end

function map_to_reference_element(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r, theta, phi = coords
    if !system.active[1]
        r_ref = NaN
    else
        r_ref = map_coordinate(r, system.r)
    end
    if !system.active[2]
        theta_ref = NaN
    else
        theta_ref = map_coordinate(Float64(theta), system.theta, true)
    end
    if !system.active[3]
        phi_ref = NaN
    else
        phi_ref = map_coordinate(Float64(phi), system.phi, true)
    end
    return (r_ref, theta_ref, phi_ref)
end

function map_from_reference_element(coords::Tuple, system::KSSphericalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for spherical coordinates."))
    end
    r_ref, theta_ref, phi_ref = coords
    if !system.active[1]
        r = NaN
    else
        r = map_coordinate_back(r_ref, system.r)
    end
    if !system.active[2]
        theta = NaN
    else
        theta = map_coordinate_back(theta_ref, system.theta, true)
    end
    if !system.active[3]
        phi = NaN
    else
        phi = map_coordinate_back(phi_ref, system.phi, true)
    end
    return (r, theta, phi)
end

function map_to_reference_element(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r, theta, z = coords
    if !system.active[1]
        r_ref = NaN
    else
        r_ref = map_coordinate(r, system.r)
    end
    if !system.active[2]
        theta_ref = NaN
    else
        theta_ref = map_coordinate(Float64(theta), system.theta, true)
    end
    if !system.active[3]
        z_ref = NaN
    else
        z_ref = map_coordinate(z, system.z)
    end
    return (r_ref, theta_ref, z_ref)
end

function map_from_reference_element(coords::Tuple, system::KSCylindricalCoordinates{T}) where T
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have 3 elements for cylindrical coordinates."))
    end
    r_ref, theta_ref, z_ref = coords
    if !system.active[1]
        r = NaN
    else
        r = map_coordinate_back(r_ref, system.r)
    end
    if !system.active[2]
        theta = NaN
    else
        theta = map_coordinate_back(theta_ref, system.theta, true)
    end
    if !system.active[3]
        z = NaN
    else
        z = map_coordinate_back(z_ref, system.z)
    end
    return (r, theta, z)
end

# Handle case mapping based on coordinate system
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
