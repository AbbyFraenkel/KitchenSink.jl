"""
    to_cartesian(coords::AbstractVector{T}, coord_sys::AbstractKSCoordinateSystem{N}) where {N, T}

Converts coordinates from the given coordinate system to Cartesian coordinates.

# Arguments
- `coords::AbstractVector{T}`: The input coordinates.
- `coord_sys::AbstractKSCoordinateSystem{N}`: The input coordinate system.

# Returns
- The converted Cartesian coordinates.

# Example
```julia
to_cartesian([1.0, π / 4], KSPolarCoordinates{Float64}(1.0, π / 4))
```
"""
function to_cartesian(coords::AbstractVector{T}, coord_sys::AbstractKSCoordinateSystem{N}) where {N,T}
    if coord_sys isa KSCartesianCoordinates{N,T}
        return to_cartesian(coords, coord_sys::KSCartesianCoordinates{N,T})
    elseif coord_sys isa KSPolarCoordinates{T}
        return to_cartesian(coords, coord_sys::KSPolarCoordinates{T})
    elseif coord_sys isa KSSphericalCoordinates{T}
        return to_cartesian(coords, coord_sys::KSSphericalCoordinates{T})
    elseif coord_sys isa KSCylindricalCoordinates{T}
        return to_cartesian(coords, coord_sys::KSCylindricalCoordinates{T})
    else
        throw(ArgumentError("Unsupported coordinate system: $coord_sys"))
    end
end

function to_cartesian(coords::AbstractVector{T}, ::KSCartesianCoordinates{N,T}) where {N,T}
    if length(coords) != N
        throw(ArgumentError("Input coordinates must have length $N for Cartesian coordinates. If a dimension is not active, use 'nothing'."))
    end
    return coords
end

function to_cartesian(coords::AbstractVector{T}, ::KSPolarCoordinates{T}) where {T}
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have length 2 for Polar coordinates (r, θ). If a dimension is not active, use 'nothing'."))
    end
    r, θ = coords
    return [r * cos(θ), r * sin(θ)]
end

function to_cartesian(coords::AbstractVector{T}, ::KSSphericalCoordinates{T}) where {T}
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have length 3 for Spherical coordinates (r, θ, φ). If a dimension is not active, use 'nothing'."))
    end
    r, θ, φ = coords
    return [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)]
end

function to_cartesian(coords::AbstractVector{T}, ::KSCylindricalCoordinates{T}) where {T}
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have length 3 for Cylindrical coordinates (r, θ, z). If a dimension is not active, use 'nothing'."))
    end
    r, θ, z = coords
    return [r * cos(θ), r * sin(θ), z]
end

"""
    from_cartesian(coords::AbstractVector{T}, coord_sys::AbstractKSCoordinateSystem{N}) where {N, T}

Converts coordinates from Cartesian coordinates to the given coordinate system.

# Arguments
- `coords::AbstractVector{T}`: The input coordinates.
- `coord_sys::AbstractKSCoordinateSystem{N}`: The target coordinate system.

# Returns
- The converted coordinates in the target coordinate system.

# Example
```julia
from_cartesian([1.0, 1.0], KSPolarCoordinates{Float64}(1.0, π / 4))
```
"""
function from_cartesian(coords::AbstractVector{T}, coord_sys::AbstractKSCoordinateSystem{N}) where {N,T}
    if coord_sys isa KSCartesianCoordinates{N,T}
        return from_cartesian(coords, coord_sys::KSCartesianCoordinates{N,T})
    elseif coord_sys isa KSPolarCoordinates{T}
        return from_cartesian(coords, coord_sys::KSPolarCoordinates{T})
    elseif coord_sys isa KSSphericalCoordinates{T}
        return from_cartesian(coords, coord_sys::KSSphericalCoordinates{T})
    elseif coord_sys isa KSCylindricalCoordinates{T}
        return from_cartesian(coords, coord_sys::KSCylindricalCoordinates{T})
    else
        throw(ArgumentError("Unsupported coordinate system: $coord_sys"))
    end
end

function from_cartesian(coords::AbstractVector{T}, ::KSCartesianCoordinates{N,T}) where {N,T}
    if length(coords) != N
        throw(ArgumentError("Input coordinates must have length $N for Cartesian coordinates. If a dimension is not active, use 'nothing'."))
    end
    return coords
end

function from_cartesian(coords::AbstractVector{T}, ::KSPolarCoordinates{T}) where {T}
    if length(coords) != 2
        throw(ArgumentError("Input coordinates must have length 2 for Polar coordinates (r, θ). If a dimension is not active, use 'nothing'."))
    end
    x, y = coords
    return [sqrt(x^2 + y^2), atan(y, x)]
end

function from_cartesian(coords::AbstractVector{T}, ::KSSphericalCoordinates{T}) where {T}
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have length 3 for Spherical coordinates (r, θ, φ). If a dimension is not active, use 'nothing'."))
    end
    x, y, z = coords
    r = sqrt(x^2 + y^2 + z^2)
    θ = ifelse(r == 0, 0, acos(z / r))  # Handle division by zero
    φ = atan(y, x)
    return [r, θ, φ]
end

function from_cartesian(coords::AbstractVector{T}, ::KSCylindricalCoordinates{T}) where {T}
    if length(coords) != 3
        throw(ArgumentError("Input coordinates must have length 3 for Cylindrical coordinates (r, θ, z). If a dimension is not active, use 'nothing'."))
    end
    x, y, z = coords
    return [sqrt(x^2 + y^2), atan(y, x), z]
end
