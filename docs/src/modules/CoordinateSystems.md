

# Scaling and Mapping Functions
@doc raw"""
	scale_value(value::T, range::Union{Tuple{T, T}, Nothing}) where T

Scales a value to the range [0, 1].

# Arguments
- `value::T`: The value to be scaled.
- `range::Union{Tuple{T, T}, Nothing}`: The range (min, max) to scale the value from. If `nothing`, the value is returned as is.

# Returns
- The scaled value in the range [0, 1].

# Mathematical Description
If `range` is not `nothing`, the scaled value is computed as:
\[ \text{scaled\_value} = \frac{\text{value} - \text{range}[1]}{\text{range}[2] - \text{range}[1]} \]
"""

@doc raw"""
	unscale_value(scaled_value::T, range::Union{Tuple{T, T}, Nothing}) where T

Unscales a value from the range [0, 1] back to its original range.

# Arguments
- `scaled_value::T`: The scaled value in the range [0, 1].
- `range::Union{Tuple{T, T}, Nothing}`: The original range (min, max). If `nothing`, the scaled value is returned as is.

# Returns
- The unscaled value in the original range.

# Mathematical Description
If `range` is not `nothing`, the unscaled value is computed as:
\[ \text{value} = \text{scaled\_value} \times (\text{range}[2] - \text{range}[1]) + \text{range}[1] \]
"""


@doc raw"""
	map_coordinate(value::T, range::Union{Tuple{T, T}, Nothing}, is_angular::Bool) where T

Maps a value to the range [-1, 1].

# Arguments
- `value::T`: The value to be mapped.
- `range::Union{Tuple{T, T}, Nothing}`: The range (min, max) to map the value from. If `nothing`, the value is returned as is.
- `is_angular::Bool`: Indicates if the value is angular.

# Returns
- The mapped value in the range [-1, 1].

# Mathematical Description
If `range` is not `nothing` and `is_angular` is `false`, the mapped value is computed as:
\[ \text{mapped\_value} = 2 \times \frac{\text{value} - \text{range}[1]}{\text{range}[2] - \text{range}[1]} - 1 \]

If `is_angular` is `true`, the mapped value is computed as:
\[ \text{wrapped} = \frac{\text{mod}(\text{value} - \text{range}[1], 2\pi)}{\text{range}[2] - \text{range}[1]} \]
\[ \text{mapped\_value} = 2 \times \text{wrapped} - 1 \]
"""


@doc raw"""
	map_coordinate_back(value::T, range::Union{Tuple{T, T}, Nothing}, is_angular::Bool) where T

Maps a value from the range [-1, 1] back to its original range.

# Arguments
- `value::T`: The value to be mapped back.
- `range::Union{Tuple{T, T}, Nothing}`: The original range (min, max). If `nothing`, the value is returned as is.
- `is_angular::Bool`: Indicates if the value is angular.

# Returns
- The mapped value in the original range.

# Mathematical Description
If `range` is not `nothing` and `is_angular` is `false`, the mapped value is computed as:
\[ \text{value} = 0.5 \times (\text{mapped\_value} + 1) \times (\text{range}[2] - \text{range}[1]) + \text{range}[1] \]

If `is_angular` is `true`, the mapped value is computed as:
\[ \text{normalized} = \frac{\text{mapped\_value} + 1}{2} \]
\[ \text{value} = \text{normalized} \times (\text{range}[2] - \text{range}[1]) + \text{range}[1] \]
"""


# Coordinate Conversion Functions
@doc raw"""
	to_cartesian(point::NTuple{N, T}, coords::KSCartesianCoordinates{T, N}) where {T <: Number, N}

Converts a point in Cartesian coordinates to Cartesian coordinates (identity function).

# Arguments
- `point::NTuple{N, T}`: The point in Cartesian coordinates.
- `coords::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.

# Returns
- The same point in Cartesian coordinates.

# Mathematical Description
\[ \text{cartesian\_point} = \text{point} \]
"""


@doc raw"""
	from_cartesian(point::NTuple{N, T}, coords::KSCartesianCoordinates{T, N}) where {T <: Number, N}

Converts a point in Cartesian coordinates to Cartesian coordinates (identity function).

# Arguments
- `point::NTuple{N, T}`: The point in Cartesian coordinates.
- `coords::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.

# Returns
- The same point in Cartesian coordinates.

# Mathematical Description
\[ \text{cartesian\_point} = \text{point} \]
"""

@doc raw"""
	to_cartesian(coords::Tuple, coord_sys::KSCartesianCoordinates{T, N}) where {T, N}

Converts coordinates in Cartesian coordinates to Cartesian coordinates (identity function).

# Arguments
- `coords::Tuple`: The coordinates in Cartesian coordinates.
- `coord_sys::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.

# Returns
- The same coordinates in Cartesian coordinates.

# Mathematical Description
\[ \text{cartesian\_coords} = \text{coords} \]
"""


@doc raw"""
	to_cartesian(coords::Tuple, system::KSPolarCoordinates{T}) where T

Converts coordinates from polar to Cartesian coordinates.

# Arguments
- `coords::Tuple`: The coordinates in polar coordinates (r, theta).
- `system::KSPolarCoordinates{T}`: The polar coordinate system.

# Returns
- A tuple representing the coordinates in Cartesian coordinates (x, y).

# Mathematical Description
\[ x = r \cos(\theta) \]
\[ y = r \sin(\theta) \]
"""


@doc raw"""
	from_cartesian(cartesian_coords::Tuple, system::KSPolarCoordinates{T}) where T

Converts coordinates from Cartesian to polar coordinates.

# Arguments
- `cartesian_coords::Tuple`: The coordinates in Cartesian coordinates (x, y).
- `system::KSPolarCoordinates{T}`: The polar coordinate system.

# Returns
- A tuple representing the coordinates in polar coordinates (r, theta).

# Mathematical Description
\[ r = \sqrt{x^2 + y^2} \]
\[ \theta = \tan^{-1}\left(\frac{y}{x}\right) \]
"""

@doc raw"""
	to_cartesian(coords::Tuple, system::KSSphericalCoordinates{T}) where T

Converts coordinates from spherical to Cartesian coordinates.

# Arguments
- `coords::Tuple`: The coordinates in spherical coordinates (r, theta, phi).
- `system::KSSphericalCoordinates{T}`: The spherical coordinate system.

# Returns
- A tuple representing the coordinates in Cartesian coordinates (x, y, z).

# Mathematical Description
\[ x = r \sin(\theta) \cos(\phi) \]
\[ y = r \sin(\theta) \sin(\phi) \]
\[ z = r \cos(\theta) \]
"""

@doc raw"""
	from_cartesian(cartesian_coords::Tuple, system::KSSphericalCoordinates{T}) where T

Converts coordinates from Cartesian to spherical coordinates.

# Arguments
- `cartesian_coords::Tuple`: The coordinates in Cartesian coordinates (x, y, z).
- `system::KSSphericalCoordinates{T}`: The spherical coordinate system.

# Returns
- A tuple representing the coordinates in spherical coordinates (r, theta, phi).

# Mathematical Description
\[ r = \sqrt{x^2 + y^2 + z^2} \]
\[ \theta = \cos^{-1}\left(\frac{z}{r}\right) \]
\[ \phi = \tan^{-1}\left(\frac{y}{x}\right) \]
"""


@doc raw"""
	to_cartesian(coords::Tuple, system::KSCylindricalCoordinates{T}) where T

Converts coordinates from cylindrical to Cartesian coordinates.

# Arguments
- `coords::Tuple`: The coordinates in cylindrical coordinates (r, theta, z).
- `system::KSCylindricalCoordinates{T}`: The cylindrical coordinate system.

# Returns
- A tuple representing the coordinates in Cartesian coordinates (x, y, z).

# Mathematical Description
\[ x = r \cos(\theta) \]
\[ y = r \sin(\theta) \]
\[ z = z \]
"""


@doc raw"""
	from_cartesian(cartesian_coords::Tuple, system::KSCylindricalCoordinates{T}) where T

Converts coordinates from Cartesian to cylindrical coordinates.

# Arguments
- `cartesian_coords::Tuple`: The coordinates in Cartesian coordinates (x, y, z).
- `system::KSCylindricalCoordinates{T}`: The cylindrical coordinate system.

# Returns
- A tuple representing the coordinates in cylindrical coordinates (r, theta, z).

# Mathematical Description
\[ r = \sqrt{x^2 + y^2} \]
\[ \theta = \tan^{-1}\left(\frac{y}{x}\right) \]
\[ z = z \]
"""

# Mapping Functions

@doc raw"""
	map_to_reference_cell(coords::Tuple, system::KSCartesianCoordinates{T, N}) where {T, N}

Maps coordinates from the Cartesian coordinate system to the reference cell.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.

# Returns
- A tuple representing the coordinates in the reference cell.

# Mathematical Description
For each dimension \(i\):
\[ \text{mapped\_coord}[i] = \text{system.active}[i] ? \text{map\_coordinate}(\text{coords}[i], \text{system.ranges}[i], \text{false}) : \text{NaN} \]
"""

@doc raw"""
	map_from_reference_cell(coords::Tuple, system::KSCartesianCoordinates{T, N}) where {T, N}

Maps coordinates from the reference cell back to the Cartesian coordinate system.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.

# Returns
- A tuple representing the coordinates in the Cartesian coordinate system.

# Mathematical Description
For each dimension \(i\):
\[ \text{coord}[i] = \text{system.active}[i] ? \text{map\_coordinate\_back}(\text{coords}[i], \text{system.ranges}[i], \text{false}) : \text{NaN} \]
"""


@doc raw"""
	map_to_reference_cell(coords::Tuple, system::KSPolarCoordinates{T}) where T

Maps coordinates from the polar coordinate system to the reference cell.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSPolarCoordinates{T}`: The polar coordinate system.

# Returns
- A tuple representing the coordinates in the reference cell.

# Mathematical Description
\[ r_{\text{ref}} = \text{system.active}[1] ? \text{map\_coordinate}(r, \text{system.r}, \text{false}) : \text{NaN} \]
\[ \theta_{\text{ref}} = \text{system.active}[2] ? \text{map\_coordinate}(\theta, \text{system.theta}, \text{true}) : \text{NaN} \]
"""


@doc raw"""
	map_from_reference_cell(coords::Tuple, system::KSPolarCoordinates{T}) where T

Maps coordinates from the reference cell back to the polar coordinate system.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSPolarCoordinates{T}`: The polar coordinate system.

# Returns
- A tuple representing the coordinates in the polar coordinate system.

# Mathematical Description
\[ r = \text{system.active}[1] ? \text{map\_coordinate\_back}(r_{\text{ref}}, \text{system.r}, \text{false}) : \text{NaN} \]
\[ \theta = \text{system.active}[2] ? \text{map\_coordinate\_back}(\theta_{\text{ref}}, \text{system.theta}, \text{true}) : \text{NaN} \]
"""


@doc raw"""
	map_to_reference_cell(coords::Tuple, system::KSSphericalCoordinates{T}) where T

Maps coordinates from the spherical coordinate system to the reference cell.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSSphericalCoordinates{T}`: The spherical coordinate system.

# Returns
- A tuple representing the coordinates in the reference cell.

# Mathematical Description
\[ r_{\text{ref}} = \text{system.active}[1] ? \text{map\_coordinate}(r, \text{system.r}, \text{false}) : \text{NaN} \]
\[ \theta_{\text{ref}} = \text{system.active}[2] ? \text{map\_coordinate}(\theta, \text{system.theta}, \text{true}) : \text{NaN} \]
\[ \phi_{\text{ref}} = \text{system.active}[3] ? \text{map\_coordinate}(\phi, \text{system.phi}, \text{true}) : \text{NaN} \]
"""


@doc raw"""
	map_from_reference_cell(coords::Tuple, system::KSSphericalCoordinates{T}) where T

Maps coordinates from the reference cell back to the spherical coordinate system.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSSphericalCoordinates{T}`: The spherical coordinate system.

# Returns
- A tuple representing the coordinates in the spherical coordinate system.

# Mathematical Description
\[ r = \text{system.active}[1] ? \text{map\_coordinate\_back}(r_{\text{ref}}, \text{system.r}, \text{false}) : \text{NaN} \]
\[ \theta = \text{system.active}[2] ? \text{map\_coordinate\_back}(\theta_{\text{ref}}, \text{system.theta}, \text{true}) : \text{NaN} \]
\[ \phi = \text{system.active}[3] ? \text{map\_coordinate\_back}(\phi_{\text{ref}}, \text{system.phi}, \text{true}) : \text{NaN} \]
"""


@doc raw"""
	map_to_reference_cell(coords::Tuple, system::KSCylindricalCoordinates{T}) where T

Maps coordinates from the cylindrical coordinate system to the reference cell.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSCylindricalCoordinates{T}`: The cylindrical coordinate system.

# Returns
- A tuple representing the coordinates in the reference cell.

# Mathematical Description
\[ r_{\text{ref}} = \text{system.active}[1] ? \text{map\_coordinate}(r, \text{system.r}, \text{false}) : \text{NaN} \]
\[ \theta_{\text{ref}} = \text{system.active}[2] ? \text{map\_coordinate}(\theta, \text{system.theta}, \text{true}) : \text{NaN} \]
\[ z_{\text{ref}} = \text{system.active}[3] ? \text{map\_coordinate}(z, \text{system.z}, \text{false}) : \text{NaN} \]
"""

@doc raw"""
	map_from_reference_cell(coords::Tuple, system::KSCylindricalCoordinates{T}) where T

Maps coordinates from the reference cell back to the cylindrical coordinate system.

# Arguments
- `coords::Tuple`: The coordinates to be mapped.
- `system::KSCylindricalCoordinates{T}`: The cylindrical coordinate system.

# Returns
- A tuple representing the coordinates in the cylindrical coordinate system.

# Mathematical Description
\[ r = \text{system.active}[1] ? \text{map\_coordinate\_back}(r_{\text{ref}}, \text{system.r}, \text{false}) : \text{NaN} \]
\[ \theta = \text{system.active}[2] ? \text{map\_coordinate\_back}(\theta_{\text{ref}}, \text{system.theta}, \text{true}) : \text{NaN} \]
\[ z = \text{system.active}[3] ? \text{map\_coordinate\_back}(z_{\text{ref}}, \text{system.z}, \text{false}) : \text{NaN} \]
"""

@doc raw"""
	compute_normal_vector(coord_sys::KSCartesianCoordinates{T, N}, point::Union{NTuple{N, T}, Vector{T}},NTuple{N, T}) where {T <: Real, N}

Computes the normal vector at a given point in a Cartesian coordinate system.

# Arguments
- `coord_sys::KSCartesianCoordinates{T, N}`: The Cartesian coordinate system.
- `point::Union{NTuple{N, T}, Vector{T}},NTuple{N, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
For each dimension \(i\):
- If `coord_sys.active[i]` is `true` and \( \text{abs}(\text{point}[i] - \text{coord_sys.ranges}[i][1]) < \epsilon \), the normal vector is \((-1, 0, \ldots, 0)\).
- If `coord_sys.active[i]` is `true` and \( \text{abs}(\text{point}[i] - \text{coord_sys.ranges}[i][2]) < \epsilon \), the normal vector is \((1, 0, \ldots, 0)\).
"""


@doc raw"""
	compute_normal_vector(coord_sys::KSPolarCoordinates{T}, point::NTuple{2, T}) where T <: Real

Computes the normal vector at a given point in a polar coordinate system.

# Arguments
- `coord_sys::KSPolarCoordinates{T}`: The polar coordinate system.
- `point::NTuple{2, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
For radial boundaries:
- If \( r \leq \text{coord_sys.r}[1] + \epsilon \) or \( r \geq \text{coord_sys.r}[2] - \epsilon \), the normal vector is \((\cos(\theta), \sin(\theta))\).

For angular boundaries:
- If \( \theta \leq \text{coord_sys.theta}[1] + \epsilon \) or \( \theta \geq \text{coord_sys.theta}[2] - \epsilon \), the normal vector is \((- \sin(\theta), \cos(\theta))\).
"""

@doc raw"""
	compute_normal_vector(coord_sys::KSSphericalCoordinates{T}, point::NTuple{3, T}) where T <: Real

Computes the normal vector at a given point in a spherical coordinate system.

# Arguments
- `coord_sys::KSSphericalCoordinates{T}`: The spherical coordinate system.
- `point::NTuple{3, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
For radial boundaries:
- If \( r \leq \text{coord_sys.r}[1] + \epsilon \) or \( r \geq \text{coord_sys.r}[2] - \epsilon \), the normal vector is \((\sin(\theta) \cos(\phi), \sin(\theta) \sin(\phi), \cos(\theta))\).

For polar boundaries:
- If \( \theta \leq \text{coord_sys.theta}[1] + \epsilon \) or \( \theta \geq \text{coord_sys.theta}[2] - \epsilon \), the normal vector is \((- \cos(\theta) \cos(\phi), - \cos(\theta) \sin(\phi), \sin(\theta))\).

For azimuthal boundaries:
- If \( \phi \leq \text{coord_sys.phi}[1] + \epsilon \) or \( \phi \geq \text{coord_sys.phi}[2] - \epsilon \), the normal vector is \((- \sin(\phi), \cos(\phi), 0.0)\).
"""

@doc raw"""
	compute_normal_vector(coord_sys::KSCylindricalCoordinates{T}, point::NTuple{3, T}) where T <: Real

Computes the normal vector at a given point in a cylindrical coordinate system.

# Arguments
- `coord_sys::KSCylindricalCoordinates{T}`: The cylindrical coordinate system.
- `point::NTuple{3, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
For radial boundaries:
- If \( r \leq \text{coord_sys.r}[1] + \epsilon \) or \( r \geq \text{coord_sys.r}[2] - \epsilon \), the normal vector is \((\cos(\theta), \sin(\theta), 0.0)\).

For angular boundaries:
- If \( \theta \leq \text{coord_sys.theta}[1] + \epsilon \) or \( \theta \geq \text{coord_sys.theta}[2] - \epsilon \), the normal vector is \((- \sin(\theta), \cos(\theta), 0.0)\).

For z boundaries:
- If \( z \leq \text{coord_sys.z}[1] + \epsilon \) or \( z \geq \text{coord_sys.z}[2] - \epsilon \), the normal vector is \((0.0, 0.0, \pm 1.0)\).
"""
# 1D Normal Vector Computation

@doc raw"""
	compute_normal_vector(coord_sys::AbstractKSCoordinateSystem, x::NTuple{1, T}) where T <: Number

Computes the normal vector at a given point in a 1D coordinate system.

# Arguments
- `coord_sys::AbstractKSCoordinateSystem`: The 1D coordinate system.
- `x::NTuple{1, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
Delegates to the appropriate 1D coordinate system type.
"""

@doc raw"""
	compute_normal_vector(coord_sys::KSCartesianCoordinates, x::NTuple{1, T}) where T <: Number

Computes the normal vector at a given point in a 1D Cartesian coordinate system.

# Arguments
- `coord_sys::KSCartesianCoordinates`: The 1D Cartesian coordinate system.
- `x::NTuple{1, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
- If \( \text{abs}(x[1] - \text{coord_sys.ranges}[1][1]) < \epsilon \), the normal vector is \((-1.0)\).
- If \( \text{abs}(x[1] - \text{coord_sys.ranges}[1][2]) < \epsilon \), the normal vector is \((1.0)\).
"""

@doc raw"""
	compute_normal_vector(coord_sys::KSPolarCoordinates, x::NTuple{1, T}) where T <: Number

Computes the normal vector at a given point in a 1D polar coordinate system.

# Arguments
- `coord_sys::KSPolarCoordinates`: The 1D polar coordinate system.
- `x::NTuple{1, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
- If \( r \leq \text{coord_sys.r}[1] + \epsilon \) or \( r \geq \text{coord_sys.r}[2] - \epsilon \), the normal vector is \((1.0)\).
"""


@doc raw"""
	compute_normal_vector(coord_sys::KSSphericalCoordinates, x::NTuple{1, T}) where T <: Number

Computes the normal vector at a given point in a 1D spherical coordinate system.

# Arguments
- `coord_sys::KSSphericalCoordinates`: The 1D spherical coordinate system.
- `x::NTuple{1, T}`: The point at which to compute the normal vector.

# Returns
- A tuple representing the normal vector.

# Mathematical Description
- If \( r \leq \text{coord_sys.r}[1] + \epsilon \) or \( r \geq \text{coord_sys.r}[2] - \epsilon \), the normal vector is \((1.0)\).
"""
