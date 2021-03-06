function send_inducing_points_to_global(xi_low::Matrix{Float64}, yi_low::Vector{Float64}, root, comm)

    smsg = MPI.Send(xi_low, root, 0, comm)
    # println("xi_low sent to root")

    smsg = MPI.Send(yi_low, root, 1, comm)
    # println("mean_low sent to root")

end #send_inducing_points_to_global


function receive_inducing_points_in_global(ni_low::Int64, p_low::Int64, recv_id, comm)

    xi_low = zeros(ni_low, p_low)
    rmsg = MPI.Recv!(xi_low, recv_id, 0, comm)
    # println("xi_low received in root\n")

    yi_low = zeros(ni_low)
    rmsg = MPI.Recv!(yi_low, recv_id, 1, comm)
    # println("mean_low received in root\n")

    return xi_low, yi_low

end #receive_inducing_points_in_global


function send_prior_to_local(pred_data::Tuple{Vector{Float64},Vector{Float64}}, recv_id, comm)
    prior_mean, prior_sd = pred_data

    smsg = MPI.Send(prior_mean, recv_id, 2, comm)
    # println("prior_mean sent to local")

    smsg = MPI.Send(prior_sd, recv_id, 3, comm)
    # println("prior_cov sent to local")

end # send_prior_to_local


function receive_prior_in_local(ni_low, root, comm)

    prior_mean = zeros(ni_low)
    prior_sd   = zeros(ni_low)

    rmsg = MPI.Recv!(prior_mean, root, 2, comm)
    # println("prior_mean received in local")
    rmsg = MPI.Recv!(prior_sd,  root, 3, comm)
    # println("prior_cov received in local")

    return prior_mean, prior_sd

end # receive_prior_in_local


function send_trace(trace_l, root ,comm)

    smsg = MPI.Send(trace_l, root, 4, comm)
    # println("trace_l sent to root \n")

end # send_trace


function receive_trace(n_local_iters, recv_id, comm)

    trace_l = zeros(n_local_iters)

    rmsg = MPI.Recv!(trace_l, recv_id, 4, comm)
    # println("trace_l received in root \n")

    return trace_l

end # receive_trace


function send_svgp_params(svgp_params, root ,comm)

    smsg = MPI.Send(svgp_params, root, 5, comm)
    # println("trace_l sent to root \n")

end # send_svgp_params


function receive_svgp_params(recv_id, comm)

    recv_params = SVGP_params(SVGP_data("dummy"), 2)

    rmsg = MPI.Recv!(recv_params, recv_id, 5, comm)
    # println("trace_l received in root \n")

    return recv_params

end # receive_svgp_params

function check_for_query(comm)
    is_message, status  = MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, comm)
    if is_message
        recv_id = status.source
        tag_ind = status.tag
        return true, recv_id, tag_ind
    else
        return false, -1, -1
    end
end

function send_data_to_neighbor(send_x, send_y, send_ind, comm)
    smsg = MPI.Send(send_x, send_ind, 0, comm)
    smsg = MPI.Send(send_y, send_ind, 1, comm)
end

function receive_neighbor_data(n::Int64, p::Int64, recv_id, comm)
    x = zeros(n, p)
    rmsg = MPI.Recv!(x, recv_id, 0, comm)

    y = zeros(n)
    rmsg = MPI.Recv!(y, recv_id, 1, comm)

    return x, y
end

function query_for_neighbor_data(nbor_ind, part_ind, comm)
    smsg = MPI.Send([0], nbor_ind, part_ind, comm)
end


