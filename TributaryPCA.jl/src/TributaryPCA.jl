"""
Author: Wayne Wang
Email: wayneyw@umich.edu
Last modified: 08/26/2021
"""

module TributaryPCA

using MPI
using LinearAlgebra
using Distributions
using Random
using SparseArrays
using FileIO

include("utils_mpi.jl")
include("oja_mpi.jl")
include("oja.jl")

end 
