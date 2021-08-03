"""
Helper functions for distributed Oja's algorithm using MPI

Author: Wayne Wang
Modified: 08/02/2021
"""


function init(d_par::Int, k::Int)
    """
    Initilize eigenvector partition, its gradient partition, and learning rate
    """
    V_par = randn((d_par, k))
    V_par .= Array(qr!(V_par).Q)
    grad_par = similar(V_par)
    α = 1e-5 * ones(k) 
    return V_par, grad_par, α
end


function data_par_loader(node_id::Int, input_dir::AbstractString)
    """
    Load a data partition
    """
    X_par = load(string(input_dir, "data_par_", node_id + 1, ".jld2"), 
                "data_par")
    return X_par
end
