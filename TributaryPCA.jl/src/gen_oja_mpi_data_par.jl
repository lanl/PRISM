"""
Helper functions to generate fake simulation data for Distributed AdaOja's algorithm
    with MPI

Author: Wayne Wang
Last modified: 08/02/2021
"""


function gen_spiked_cov_data(Σ::AbstractArray; N::Int = 1)
    """
    Generate fake data from a spiked covariance model s.t.
        X ∈ d × 1 ∼ N(0, Σ) where Σ = A diag(w)^2 A^T + σ^2 I.
    
    Here, A ∈ d × k is a set of k d-dimensional orthonormal vectors. 
    Set w ∈ k × 1 s.t. w_i ∼ Unif(0,1) and w_1 ≥ w_1 ≥ ... ≥ w_k and w_i
    is then scaled by w_1. The σ here is a noise parameter.

    Output: X ∈ d × N 
    """
    d = size(Σ, 1)
    Z = randn((d, N))

    C = cholesky(Σ)
    Z .= C.L * Z #follows N(0,Σ)

    return (N == 1) ? Z[:] : Z
end


function spiked_cov(k::Int, d::Int, σ::Real = 0.5)
    """
    The spiked covariance model: Σ = A diag(w)^2 A^T + σ^2 I.
    
    Here, A ∈ d × k is a set of k d-dimensional orthonormal vectors. 
    Set w ∈ k × 1 s.t. w_i ∼ Unif(0,1) and w_1 ≥ w_1 ≥ ... ≥ w_k and w_i
    is then scaled by w_1. The σ here is a noise parameter.

    Output: A ∈ d × k, Σ ∈ d × d
    """ 
    A = randn((d, k))
    A .= Array(qr!(A).Q)

    w = [rand(Uniform(0,1)) for _ = 1:k]
    sort!(w, rev = true)
    w ./= w[1] 

    Σ = A * (diagm(w).^2) * copy(A') .+ σ^2 * I(d)
    Σ .= (Σ .+ copy(Σ')) ./ 2

    return A, Σ
end


function gen_spiked_cov_data_par(num_par::Int, out_dir::AbstractString;
    N::Int = 1000, d::Int = 128*128, k::Int = 5, σ::Real = 0.5)
    """
    Generate data with spiked covariance and save into partitions
    """
    A, Σ = spiked_cov(k, d, σ)
    X = gen_spiked_cov_data(Σ, N = N)
    d_par = Int(d / num_par)
    X_par = zeros(d_par, N)
    for i = 1:num_par
        X_par .= X[d_par * (i - 1) + 1:d_par * i, :]
        data_dir = string(out_dir, "scaling_N", N, "_d", d, "_k", k, "/scaling_", num_par)
        if !isdir(data_dir)
            mkpath(data_dir)
        end
        save(string(data_dir, "/data_par_", i, ".jld2"), Dict("data_par" => X_par))
    end
    return nothing
end


function gen_data_par(num_par::Int, out_dir::AbstractString;
    N::Int = 1000, d::Int = 512*256, k::Int = 128)
    """
    Generate fake partitioned data
    """
    d_par = Int(d / num_par)
    X_par = zeros(d_par, N)
    for i = 1:num_par
        X_par .= randn((d_par, N))
        data_dir = string(out_dir, "scaling_N", N, "_d", d, "_k", k, "/scaling_", num_par)
        if !isdir(data_dir)
            mkpath(data_dir)
        end
        save(string(data_dir, "/data_par_", i, ".jld2"), Dict("data_par" => X_par))
    end
    return nothing
end


#
# Start generating data
#
using TributaryPCA

num_proc = parse(Int64, ARGS[1]) #take the cml input for number of partitions
out_dir = ARGS[2] #take the cml input for data directory
gen_data_par(num_proc, out_dir)


