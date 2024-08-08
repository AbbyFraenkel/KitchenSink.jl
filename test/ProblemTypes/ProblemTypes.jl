
using Test, LinearAlgebra, SparseArrays
using ..KSTypes, ..ProblemTypes, ..SpectralMethods, ..Preprocessing, ..MultiLevelMethods, ..CoordinateSystems

include("boundary_conditions.jl")
include("bvdaes.jl")
include("coupled_problems.jl")
include("ides.jl")
include("odes.jl")
include("pdes.jl")
include("pides.jl")
