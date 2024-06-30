

"""
    update_basis_functions(element::Element)

Update the basis functions of a given element.

# Arguments
- `element::Element`: The element for which the basis functions need to be updated.

# Returns
- `new_basis_functions::Array{Array{Function,1},1}`: A 2D array containing the updated basis functions.

# Details
This function iterates over the nodes of the element and generates new basis functions for each node.
The basis functions are defined as linear functions that are 1 at the current node and 0 at the neighboring nodes.
The new basis functions are stored in a 2D array and returned as the result.
"""
function update_basis_functions(element::Element)
    new_basis_functions = []

    for i in 1:length(element.nodes)
        n = length(element.nodes[i])
        basis_functions = []

        for j in 1:(n-1)
            basis_function = [
                (x) -> 1 -
                       (x - element.nodes[i][j]) /
                       (element.nodes[i][j+1] - element.nodes[i][j]),
                (x) -> (x - element.nodes[i][j]) /
                       (element.nodes[i][j+1] - element.nodes[i][j])
            ]
            push!(basis_functions, basis_function)
        end

        push!(new_basis_functions, basis_functions)
    end

    return new_basis_functions
end
