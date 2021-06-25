using Test

using HSVGP  
using Random
using Statistics

"""Root mean squared error"""
rmse(x, y) = sqrt.(mean(abs2.(x - y)))

@testset "HSVGP tests" begin
  include("test-gaussian-likelihood.jl")
end
