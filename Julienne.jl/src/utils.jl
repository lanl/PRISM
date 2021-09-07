"""
Helper functions for streaming lm 

Author: Wayne Wang
Email: wayneyw@umich.edu
Last modified: 08/02/2021
"""

function plot_partitions(partitions::AbstractArray, x::AbstractArray, y::AbstractArray)
    """
    Plot the estimated regression curve on top of the point clouds
        for each partition
    """
    plt = scatter(x, y, legend = false)

    for i = 1:length(partitions)
        par = partitions[i]
        rng = par[1]
        coefs = par[2][2:3]
        f(x) = coefs[1] + coefs[2]*x
        plot!(f, rng[1], rng[2], color = :red, legend = false)
    end

    display(plt)
end
