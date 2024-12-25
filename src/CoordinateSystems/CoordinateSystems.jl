module CoordinateSystems

using LinearAlgebra, StaticArrays
using ..KSTypes

# Core coordinate transformation functions
export to_cartesian,
    from_cartesian,
    map_to_reference_cell,
    map_from_reference_cell,
    map_coordinate,
    map_coordinate_back

# Scaling and mapping utilities
export scale_value,
    unscale_value

# Normal vector computation
export compute_normal_vector

# Coordinate system operations
export get_coordinate_transform,
    is_identity_transform

# Helper functions
export check_valid_coordinates,
    mod2pi,
    map_to_reference_cell,
    map_from_reference_cell

# Coordinate system specific helpers
export cartesian_to_polar,
    polar_to_cartesian,
    cartesian_to_spherical,
    spherical_to_cartesian,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian

# Validation utilities
export check_coordinate_bounds,
    verify_coordinate_transform
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
function scale_value(value::T, range::Union{Tuple{T, T}, Nothing}) where {T}
	if range === nothing
		return value
	else
		return (value - range[1]) / (range[2] - range[1])
	end
end

@doc raw"""
	unscale_value(scaled_value::T, range::Union{Tuple{T, T}, Nothing}) where T

Scales a value from the range [0, 1] back to its original range.

# Arguments
- `scaled_value::T`: The scaled value in the range [0, 1].
- `range::Union{Tuple{T, T}, Nothing}`: The original range (min, max). If `nothing`, the scaled value is returned as is.

# Returns
- The unscaled value in the original range.

# Mathematical Description
If `range` is not `nothing`, the unscaled value is computed as:
\[ \text{value} = \text{scaled\_value} \times (\text{range}[2] - \text{range}[1]) + \text{range}[1] \]
"""
function unscale_value(scaled_value::T, range::Union{Tuple{T, T}, Nothing}) where {T}
	if range === nothing
		return scaled_value
	else
		return scaled_value * (range[2] - range[1]) + range[1]
	end
end

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
function map_coordinate(
	value::Real, range::Union{Nothing, Tuple{<:Real, <:Real}}, is_angular::Bool
)
	if range === nothing
		return convert(typeof(first(range)), value)
	end

	# Convert value to same type as range
	value = convert(typeof(first(range)), value)

	if is_angular
		wrapped = mod(value - range[1], 2π) / (range[2] - range[1])
		return 2 * wrapped - 1
	else
		return 2 * (value - range[1]) / (range[2] - range[1]) - 1
	end
end

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
function map_coordinate_back(
	value::Union{T, NTuple{N, T}, Vector{T}},
	range::Union{Tuple{T, T}, Nothing},
	is_angular::Bool) where {T, N}
	if range === nothing
		return value
	else
		if is_angular
			if value isa Tuple || value isa Vector
				return ntuple(i -> (value[i] + 1) / 2 * (range[2] - range[1]) + range[1], N)
			else
				normalized = (value + 1) / 2
				return normalized * (range[2] - range[1]) + range[1]
			end
		else
			if value isa Tuple || value isa Vector
				return ntuple(
					i -> 0.5 * (value[i] + 1) * (range[2] - range[1]) + range[1], N)
			else
				return 0.5 * (value + 1) * (range[2] - range[1]) + range[1]
			end
		end
	end
end

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
function to_cartesian(
	point::Union{NTuple{N, T}, Vector{T}},
	coords::KSCartesianCoordinates{T, N}) where {T <: Number, N}
	check_valid_coordinates(point)
	if length(point) != N
		throw(ArgumentError("Input point must have $N elements for cartesian coordinates."))
	end
	return point isa Vector{T} ? Tuple(point) : point
end

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
function from_cartesian(
	point::Union{NTuple{N, T}, Vector{T}},
	coords::KSCartesianCoordinates{T, N}) where {T <: Number, N}
	check_valid_coordinates(point)
	if length(point) != N
		throw(ArgumentError("Input point must have $N elements for cartesian coordinates."))
	end
	return point isa Vector{T} ? Tuple(point) : point
end

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
function to_cartesian(coords::Tuple, coord_sys::KSCartesianCoordinates{T, N}) where {T, N}
	check_valid_coordinates(coords)
	if length(coords) != N
		throw(
			ArgumentError(
				"Input coordinates must have $N elements for cartesian coordinates."
			),
		)
	end
	return coords
end

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
function to_cartesian(coords::Tuple, system::KSPolarCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	if length(coords) != 2
		throw(
			ArgumentError("Input coordinates must have 2 elements for polar coordinates.")
		)
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
function from_cartesian(
	cartesian_coords::Union{Tuple, AbstractVector},
	system::KSPolarCoordinates{T}) where {T}
	check_valid_coordinates(cartesian_coords)
	cartesian_coords = Tuple(cartesian_coords)
	if length(cartesian_coords) != 2
		throw(
			ArgumentError("Input coordinates must have 2 elements for polar coordinates.")
		)
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
function to_cartesian(
	coords::Union{Tuple, AbstractVector}, system::KSSphericalCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	coords = Tuple(coords)
	if length(coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for spherical coordinates."
			),
		)
	end
	coords = Tuple(coords)
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
function from_cartesian(
	cartesian_coords::Union{Tuple, AbstractVector},
	system::KSSphericalCoordinates{T}) where {T}
	check_valid_coordinates(cartesian_coords)
	cartesian_coords = Tuple(cartesian_coords)
	if length(cartesian_coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for spherical coordinates."
			),
		)
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
function to_cartesian(
	coords::Union{Tuple, AbstractVector}, system::KSCylindricalCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	coords = Tuple(coords)
	if length(coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for cylindrical coordinates."
			),
		)
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
function from_cartesian(
	cartesian_coords::Union{Tuple, AbstractVector},
	system::KSCylindricalCoordinates{T}) where {T}
	check_valid_coordinates(cartesian_coords)
	cartesian_coords = Tuple(cartesian_coords)
	if length(cartesian_coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for cylindrical coordinates."
			),
		)
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

# Coordinate Transform Functions
function get_coordinate_transform(coord_sys::AbstractKSCoordinateSystem)
	# Create transform functions based on coordinate system type
	if coord_sys isa KSCartesianCoordinates
		# For Cartesian, just use mapping functions directly
		return (x -> map_to_reference_cell(Tuple(x), coord_sys),
			x -> collect(map_from_reference_cell(x, coord_sys)))

	elseif coord_sys isa KSPolarCoordinates
		# For polar, transform through Cartesian coordinates
		return (x -> map_to_reference_cell(from_cartesian(x, coord_sys), coord_sys),
			x -> collect(to_cartesian(map_from_reference_cell(x, coord_sys), coord_sys)))

	elseif coord_sys isa KSSphericalCoordinates
		# For spherical, transform through Cartesian coordinates
		return (x -> map_to_reference_cell(from_cartesian(x, coord_sys), coord_sys),
			x -> collect(to_cartesian(map_from_reference_cell(x, coord_sys), coord_sys)))

	elseif coord_sys isa KSCylindricalCoordinates
		# For cylindrical, transform through Cartesian coordinates
		return (x -> map_to_reference_cell(from_cartesian(x, coord_sys), coord_sys),
			x -> collect(to_cartesian(map_from_reference_cell(x, coord_sys), coord_sys)))

	else
		throw(ArgumentError("Unsupported coordinate system type: $(typeof(coord_sys))"))
	end
end

function get_coordinate_transform(coord_sys::Any)
	throw(ArgumentError("Invalid coordinate system type: $(typeof(coord_sys))"))
end

function check_valid_coordinates(point)
	if any(isnan.(point)) || any(isinf.(point))
		throw(ArgumentError("Coordinates cannot contain NaN or Inf values"))
	end
end

function is_identity_transform(coord_sys::AbstractKSCoordinateSystem)
    if hasfield(typeof(coord_sys), :transformation_data)
        return isnothing(coord_sys.transformation_data)
    end
    return true
end

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
function map_to_reference_cell(
	coords::Tuple,
	system::KSCartesianCoordinates{T, N}) where {T, N}
	check_valid_coordinates(coords)
	if length(coords) != N
		throw(
			ArgumentError(
				"Input coordinates must have $N elements for cartesian coordinates."
			),
		)
	end
	return ntuple(
		i -> system.active[i] ? map_coordinate(coords[i], system.ranges[i], false) : NaN,
		N,
	)
end

"""
    map_to_reference_cell(point::AbstractVector{T}, coord_sys::KSCartesianCoordinates{T,N}) where {T,N}

Maps a point from physical coordinates to reference coordinates in a Cartesian system.
Converts Vector input to Tuple and delegates to existing method.
"""
function map_to_reference_cell(point::AbstractVector{T}, coord_sys::KSCartesianCoordinates{T,N}) where {T,N}
    length(point) == N || throw(DimensionMismatch("Point dimension ($(length(point))) must match coordinate system dimension ($N)"))
    # Convert Vector to Tuple and delegate to existing implementation
    return map_to_reference_cell(Tuple(point), coord_sys)
end

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
function map_from_reference_cell(
	coords::Union{NTuple{N, T}, Vector{T}},
	system::KSCartesianCoordinates{T, N}) where {T, N}
	check_valid_coordinates(coords)
	if length(coords) != N
		throw(
			ArgumentError(
				"Input coordinates must have $N elements for cartesian coordinates."
			),
		)
	end
	coords = Tuple(coords)
	return ntuple(
		i -> if system.active[i]
			map_coordinate_back(coords[i], system.ranges[i], false)
		else
			NaN
		end,
		N,
	)
end

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
function map_to_reference_cell(
	coords::Union{NTuple{2, T}, Vector{T}}, system::KSPolarCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	if length(coords) != 2
		throw(
			ArgumentError("Input coordinates must have 2 elements for polar coordinates.")
		)
	end
	coords = Tuple(coords)
	r, theta = coords
	r_ref = system.active[1] ? map_coordinate(r, system.r, false) : NaN
	theta_ref = system.active[2] ? map_coordinate(Float64(theta), system.theta, true) : NaN
	return (r_ref, theta_ref)
end

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
function map_from_reference_cell(
	coords::Union{NTuple{2, T}, Vector{T}}, system::KSPolarCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	if length(coords) != 2
		throw(
			ArgumentError("Input coordinates must have 2 elements for polar coordinates.")
		)
	end
	coords = Tuple(coords)
	r_ref, theta_ref = coords
	r = system.active[1] ? map_coordinate_back(r_ref, system.r, false) : NaN
	theta = system.active[2] ? map_coordinate_back(theta_ref, system.theta, true) : NaN
	return (r, theta)
end

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
function map_to_reference_cell(
	coords::Tuple,
	system::KSSphericalCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	if length(coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for spherical coordinates."
			),
		)
	end
	r, theta, phi = coords
	r_ref = system.active[1] ? map_coordinate(r, system.r, false) : NaN
	theta_ref = system.active[2] ? map_coordinate(Float64(theta), system.theta, true) : NaN
	phi_ref = system.active[3] ? map_coordinate(Float64(phi), system.phi, true) : NaN
	return (r_ref, theta_ref, phi_ref)
end

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
function map_from_reference_cell(
	coords::Union{NTuple{3, T}, Vector{T}},
	system::KSSphericalCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	if length(coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for spherical coordinates."
			),
		)
	end
	coords = Tuple(coords)
	r_ref, theta_ref, phi_ref = coords
	r = system.active[1] ? map_coordinate_back(r_ref, system.r, false) : NaN
	theta = system.active[2] ? map_coordinate_back(theta_ref, system.theta, true) : NaN
	phi = system.active[3] ? map_coordinate_back(phi_ref, system.phi, true) : NaN
	return (r, theta, phi)
end

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
function map_to_reference_cell(
	coords::Union{Tuple, AbstractVector},
	system::KSCylindricalCoordinates{T},
) where {T}
	check_valid_coordinates(coords)
	coords = Tuple(coords)
	if length(coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for cylindrical coordinates."
			),
		)
	end
	r, theta, z = coords
	r_ref = system.active[1] ? map_coordinate(r, system.r, false) : NaN
	theta_ref = system.active[2] ? map_coordinate(Float64(theta), system.theta, true) : NaN
	z_ref = system.active[3] ? map_coordinate(z, system.z, false) : NaN
	return (r_ref, theta_ref, z_ref)
end

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
function map_from_reference_cell(
	coords::Union{Tuple, AbstractVector},
	system::KSCylindricalCoordinates{T}) where {T}
	check_valid_coordinates(coords)
	coords = Tuple(coords)
	if length(coords) != 3
		throw(
			ArgumentError(
				"Input coordinates must have 3 elements for cylindrical coordinates."
			),
		)
	end
	r_ref, theta_ref, z_ref = coords
	r = system.active[1] ? map_coordinate_back(r_ref, system.r, false) : NaN
	theta = system.active[2] ? map_coordinate_back(theta_ref, system.theta, true) : NaN
	z = system.active[3] ? map_coordinate_back(z_ref, system.z, false) : NaN
	return (r, theta, z)
end

# Normal Vector Computation

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
function compute_normal_vector(
	coord_sys::KSCartesianCoordinates{T, N},
	point::Union{NTuple{N, T}, Vector{T}}) where {T <: Real, N}
	normals = ntuple(_ -> 0.0, N)
	for i in 1:N
		if coord_sys.active[i]
			if abs(point[i] - coord_sys.ranges[i][1]) < eps(T)
				normals = ntuple(j -> j == i ? -1.0 : 0.0, N)
				break
			elseif abs(point[i] - coord_sys.ranges[i][2]) < eps(T)
				normals = ntuple(j -> j == i ? 1.0 : 0.0, N)
				break
			end
		end
	end
	return normals
end

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
function compute_normal_vector(
	coord_sys::KSPolarCoordinates{T},
	point::NTuple{2, T}) where {T <: Real}
	r, theta = point
	theta = mod2pi(theta)
	normals = (0.0, 0.0)

	# Radial normal on radial boundaries
	if coord_sys.active[1]
		if r <= coord_sys.r[1] + eps(T)
			normals = (cos(theta), sin(theta))
		elseif r >= coord_sys.r[2] - eps(T)
			normals = (cos(theta), sin(theta))
		end
	end

	# Azimuthal normal on angular boundaries
	if coord_sys.active[2]
		if theta <= coord_sys.theta[1] + eps(T)
			normals = (-sin(theta), cos(theta))
		elseif theta >= coord_sys.theta[2] - eps(T)
			normals = (-sin(theta), cos(theta))
		end
	end

	return normals
end

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
function compute_normal_vector(
	coord_sys::KSSphericalCoordinates{T},
	point::NTuple{3, T}) where {T <: Real}
	r, theta, phi = point
	theta = mod2pi(theta)
	phi = mod2pi(phi)
	normals = (0.0, 0.0, 0.0)

	# Radial normal on radial boundaries
	if coord_sys.active[1]
		if r <= coord_sys.r[1] + eps(T)
			normals = (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
		elseif r >= coord_sys.r[2] - eps(T)
			normals = (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
		end
	end

	# Polar normal on theta boundaries
	if coord_sys.active[2]
		if theta <= coord_sys.theta[1] + eps(T)
			normals = (-cos(theta) * cos(phi), -cos(theta) * sin(phi), sin(theta))
		elseif theta >= coord_sys.theta[2] - eps(T)
			normals = (-cos(theta) * cos(phi), -cos(theta) * sin(phi), sin(theta))
		end
	end

	# Azimuthal normal on phi boundaries
	if coord_sys.active[3]
		if phi <= coord_sys.phi[1] + eps(T)
			normals = (-sin(phi), cos(phi), 0.0)
		elseif phi >= coord_sys.phi[2] - eps(T)
			normals = (-sin(phi), cos(phi), 0.0)
		end
	end

	return normals
end

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
function compute_normal_vector(
	coord_sys::KSCylindricalCoordinates{T},
	point::NTuple{3, T}) where {T <: Real}
	r, theta, z = point
	theta = mod2pi(theta)
	normals = (0.0, 0.0, 0.0)

	# Radial normal on radial boundaries
	if coord_sys.active[1]
		if r <= coord_sys.r[1] + eps(T)
			normals = (cos(theta), sin(theta), 0.0)
		elseif r >= coord_sys.r[2] - eps(T)
			normals = (cos(theta), sin(theta), 0.0)
		end
	end

	# Azimuthal normal on angular boundaries
	if coord_sys.active[2]
		if theta <= coord_sys.theta[1] + eps(T)
			normals = (-sin(theta), cos(theta), 0.0)
		elseif theta >= coord_sys.theta[2] - eps(T)
			normals = (-sin(theta), cos(theta), 0.0)
		end
	end

	# Z normal on z boundaries
	if coord_sys.active[3]
		if abs(z - coord_sys.z[1]) <= eps(T)
			normals = (0.0, 0.0, -1.0)
		elseif abs(z - coord_sys.z[2]) <= eps(T)
			normals = (0.0, 0.0, 1.0)
		end
	end

	return normals
end

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
function compute_normal_vector(
	coord_sys::AbstractKSCoordinateSystem,
	x::NTuple{1, T}) where {T <: Number}
	if coord_sys isa KSCartesianCoordinates1D
		return compute_normal_vector(coord_sys, x)
	elseif coord_sys isa KSPolarCoordinates1D
		return compute_normal_vector(coord_sys, x)
	elseif coord_sys isa KSSphericalCoordinates1D
		return compute_normal_vector(coord_sys, x)
	else
		error("Unsupported 1D coordinate system type: $(typeof(coord_sys))")
	end
end

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
function compute_normal_vector(
	coord_sys::KSCartesianCoordinates,
	x::NTuple{1, T}) where {T <: Number}
	if abs(x[1] - coord_sys.ranges[1][1]) < eps(T)
		return (-1.0,)
	elseif abs(x[1] - coord_sys.ranges[1][2]) < eps(T)
		return (1.0,)
	else
		return (0.0,)
	end
end

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
function compute_normal_vector(
	coord_sys::KSPolarCoordinates,
	x::NTuple{1, T}) where {T <: Number}
	r = x[1]
	if r <= coord_sys.r[1] + eps(T)
		return (1.0,)
	elseif r >= coord_sys.r[2] - eps(T)
		return (1.0,)
	else
		return (0.0,)
	end
end

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
function compute_normal_vector(
	coord_sys::KSSphericalCoordinates,
	x::NTuple{1, T}) where {T <: Number}
	r = x[1]
	if r <= coord_sys.r[1] + eps(T)
		return (1.0,)
	elseif r >= coord_sys.r[2] - eps(T)
		return (1.0,)
	else
		return (0.0,)
	end
end
end  # module CoordinateSystems
