module AdaptiveMethods

using LinearAlgebra, SparseArrays
using ..KSTypes, ..SpectralMethods

export h_refine, p_refine, hp_refine, select_subelement_nodes
include("hp_refinement.jl")


export hp_refine_superposition, superconvergence_refinement, adapt_mesh_superposition
include("refinement_by_superpostion.jl")

end

