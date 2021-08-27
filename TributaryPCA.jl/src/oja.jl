"""
Oja (non-Distributed) streaming PCA method

Author: Wayne Wang
Email: wayneyw@umich.edu
Last modified: 08/26/2021
"""


struct ADAGRAD end
struct ROBBINS_MONRO end


function oja(Xt::AbstractArray{<:Real}, V::AbstractArray{<:Real},
    α::AbstractArray{<:Real}; scheduling::AbstractString = "adagrad",
    stepsize_dim::Int = 1)
    """
    Oja's method
    """
    # initial setting
    if scheduling == "adagrad"
        scheduling = ADAGRAD()
    elseif scheduling == "robbins-monro"
        scheduling = ROBBINS_MONRO()
    end

    # update eigenvectors & learning rates
    V, α = ojaupdate(scheduling, Xt, V, α, stepsize_dim)
    # retraction
    V .= Array(qr!(V).Q)

    return V, α
end


function ojaupdate(scheduling::ROBBINS_MONRO, Xt::AbstractArray{<:Real}, V::AbstractArray{<:Real},
    α::AbstractArray{<:Real}, stepsize_dim::Int)
    """
    Oja × Robbins-Monro
    """
    V .+= α*Xt*(copy(Xt')*V)
    α = nothing
    return V, α
end


function ojaupdate(scheduling::ADAGRAD, Xt::AbstractArray{<:Real}, V::AbstractArray{<:Real},
    α::AbstractArray{<:Real}, stepsize_dim::Int)
    """
    Oja × Adagrad
    """
    grad = Xt*(copy(Xt')*V)
    if stepsize_dim == 1 # α ∈ k × 1
        α .= α .+ sum(abs2,grad,dims=1)[:]
        V .+= grad ./ reshape(sqrt.(α),(1,size(α,1)))
    elseif stepsize_dim == 2 # α ∈ d × k
        α .= α .+ grad .* grad
        V .+= grad ./ sqrt.(α)
    end
    return V, α
end