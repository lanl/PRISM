"""
Distributed AdaOja's algorithm for streaming/online PCA with MPI

Author: Wayne Wang
Last modified: 08/02/2021
"""


function oja_mpi(node_id::Int, num_nodes::Int, comm, input_dir::AbstractString;
    k::Int = 5)
    """
    AdaOja's algorithm with MPI
    """
    # initialization
    Xt = data_par_loader(node_id, input_dir) #data stream of dimension d_b × N_t
    d_par, N_t = size(Xt)
    V_par, grad_par, α = init(d_par, k)
    ## save intial estimate from each partition
    # save(string(input_dir, "Vhat_init_par_", node_id + 1, ".jld2"), Dict("Vhat_par" => V_par))

    # run streaming algorithm on each rank
    Xt = data_par_loader(node_id, input_dir)
    X_par = zeros((d_par, 1))

    # start a timer
    MPI.Barrier(comm) #make sure every process reaches this point
    tstart = MPI.Wtime()

    for i = 1:N_t
        X_par[:, 1] .= Xt[:, i] #making sure the (d_par,1) deos not collapse to (d_par,)
        # compute gradient
        ojagrad_mpi!(grad_par, X_par, V_par, node_id, num_nodes, comm)
        # update learning rates and estimate
        ojaupdate_mpi!(V_par, α, grad_par, node_id, num_nodes, comm)
        # QR orthogonalization
        direct_tsqr!(V_par, node_id, num_nodes, comm; k = k)
    end

    # stop the timer
    MPI.Barrier(comm) #make sure every process reaches this point
    tend = MPI.Wtime() - tstart
    if node_id == 0 #output timing result only on the master node
        # print(string(tend, "\n"))
        open("time_summary_oja_mpi_scaling.txt", "a") do f
            write(f, string(num_nodes, "\t", tend, "\n"))
        end
    end
    
    # save final estimate from each partition
    # save(string(input_dir, "Vhat_par_", node_id + 1, ".jld2"), Dict("Vhat_par" => V_par))

    return nothing
end


function ojagrad_mpi!(grad_par::AbstractArray, X_par::AbstractArray, V_par::AbstractArray, 
    node_id::Int, num_nodes::Int, comm; 
    N::Int = 1, master::Int = 0, 
    tag_from_master::Int = 1,
    tag_from_worker::Int = 2)
    """
    Compute X × (X^T × V) in the Oja's update using MPI 
    """
    # each node (including master) get a partition of
    # X (data) and V (current estimate)
    @assert size(X_par, 1) == size(V_par, 1)

    # compute X^T × V partitions and collect via allreduce, i.e.,
    # X^T_V = X^T_V_par_1 + X^T_V_par_2 + ⋯
    XT_V_par = copy(X_par') * V_par
    XT_V = MPI.Allreduce(XT_V_par, +, comm)

    # compute X × (X^T × V)
    mul!(grad_par, X_par, XT_V, 1.0/N, 0.0)

    return nothing
end


function ojaupdate_mpi!(V_par::AbstractArray, α::AbstractArray, 
    grad_par::AbstractArray, node_id::Int, num_nodes::Int, comm;
    master::Int = 0,
    tag_from_master::Int = 1,
    tag_from_worker::Int = 2)
    """
    Oja × Adagrad learning rate schedule
    """
    # compute local sum of squared grad and reduce
    grad_norm_sq = MPI.Allreduce(sum(abs2, grad_par, dims = 1)[:], +, comm) # ∈ k × 1 
    α .+= grad_norm_sq # α ∈ k × 1

    # update partitions of the current estimate
    V_par .+= grad_par ./ reshape(sqrt.(α), (1, size(α, 1)))

    return nothing
end


function direct_tsqr!(V_par::AbstractArray, node_id::Int, num_nodes::Int, comm;
    k::Int = 5, master::Int = 0, tag_from_master::Int = 1, tag_from_worker::Int = 2)
    """
    Implementation of the DirectTSQR algorithm (Benson, Cleich, and Demmel, 2013)
    """
    # local qr on each partition
    QR1 = qr!(V_par)
    Q1_par = Array(QR1.Q)
    R1_par = Array(QR1.R)

    # allocate intermediate data
    Q2_par = zeros((k, k))

    # master tasks
    if node_id == master
        # gather all R1 partitions on master
        R1 = zeros((k * num_nodes, k))
        R1[1:k, :] .= R1_par

        for i = 1:(num_nodes - 1)
            MPI.Recv!(view(R1, (k * i + 1):k * (i + 1), :), i, tag_from_worker, comm)
        end
        
        # perform QR for R1 on master node
        QR2 = qr!(R1)
        Q2 = Array(QR2.Q)

        # send Q2 partitions to workers
        Q2_par .= Q2[1:k, :]
        for i = 1:(num_nodes - 1)
            MPI.Send(view(Q2, (k * i + 1):k * (i + 1), :), i, tag_from_master, comm)
        end
    end

    # worker tasks
    if node_id != master
        # send R1 partitions to the master
        MPI.Send(R1_par, master, tag_from_worker, comm)

        # receive Q2 partitions from master
        MPI.Recv!(Q2_par, master, tag_from_master, comm)
    end 

    # form final partitions of the Q matrix inplace to V_par
    V_par .= Q1_par * Q2_par

    return nothing
end