# using DocStringExtensions

@doc """
    check_empty(arg, name)

Check if an argument is empty.

# Arguments
- `arg`: The argument to check.
- `name::String`: The name of the argument.
"""
function check_empty(arg::SparseMatrixCSC, name::String)
    nnz(arg) == 0 && throw(ArgumentError("$(name) must not be empty"))
end

# Specific method for arrays
function check_empty(arg::AbstractArray, name::String)
    isempty(arg) && throw(ArgumentError("$(name) must not be empty"))
end

# General check_empty function using multiple dispatch
function check_empty(arg, name::String)
    isempty(arg) && throw(ArgumentError("$(name) must not be empty"))
end

@doc """
    check_positive(arg, name)

Check if an argument is positive.

# Arguments
- `arg`: The argument to check.
- `name::String`: The name of the argument.
"""
function check_positive(arg, name::String)
    arg > 0 || throw(ArgumentError("$(name) must be positive"))
end

@doc """
    check_non_negative(arg, name)

Check if an argument is non-negative.

# Arguments
- `arg`: The argument to check.
- `name::String`: The name of the argument.
"""
function check_non_negative(arg, name::String)
    arg >= 0 || throw(ArgumentError("$(name) must be non-negative"))
end

@doc """
    check_type(arg, type, name)

Check if an argument is of a specific type.

# Arguments
- `arg`: The argument to check.
- `type`: The expected type of the argument.
- `name::String`: The name of the argument.
"""
function check_type(arg, type, name::String)
    isa(arg, type) || throw(ArgumentError("$(name) must be of type $(type)"))
end

@doc """
    check_length(arg, len, name)

Check if an argument has a specific length.

# Arguments
- `arg`: The argument to check.
- `len::Int`: The expected length.
- `name::String`: The name of the argument.
"""
function check_length(arg, len::Int, name::String)
    length(arg) == len || throw(ArgumentError("$(name) must have length $(len)"))
end

@doc """
    check_range(arg, min, max, name)

Check if an argument is within a specific range.

# Arguments
- `arg`: The argument to check.
- `min`: The minimum value.
- `max`: The maximum value.
- `name::String`: The name of the argument.
"""
function check_range(arg, min, max, name::String)
    (arg >= min && arg <= max) ||
        throw(ArgumentError("$(name) must be in the range [$(min), $(max)]"))
end

@doc """
    check_non_empty_elements(arg, name)

Check if an argument contains non-empty elements.

# Arguments
- `arg`: The argument to check.
- `name::String`: The name of the argument.
"""


function check_non_empty_elements(arg, name::String)
    isempty(arg) && throw(ArgumentError("$(name) must not be empty"))
    for elem in arg
        isempty(elem) && throw(ArgumentError("Elements of $(name) must not be empty"))
    end
end

@doc """
    check_is_nothing(arg, name::String)

Check if an argument is nothing.

# Arguments
- `arg`: The argument to check.
- `name::String`: The name of the argument.
"""
function check_is_nothing(arg, name::String)
    isnothing(arg) && throw(ArgumentError("$(name) must not be nothing"))
end

@doc """
    check_fields(arg, name::String)

Check if the fields of a struct are valid.

# Arguments
- `arg`: The argument to check.
- `name::String`: The name of the argument.
- `ignore_fields`: Fields to ignore in the check.
"""

function check_fields(arg, name::String, ignore_fields = [])
    for field in fieldnames(typeof(arg))
        if !(string(field) in ignore_fields)
            field_value = getfield(arg, field)
            isnothing(field_value) &&
                throw(ArgumentError("Field '$field' of $name must not be nothing"))
            if (isa(field_value, AbstractArray) || isa(field_value, String)) &&
               isempty(field_value)
                throw(ArgumentError("Field '$field' of $name must not be empty"))
            end
        end
    end
end
