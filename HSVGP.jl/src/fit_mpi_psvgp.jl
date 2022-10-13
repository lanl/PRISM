"""
    mpifit_psvgp(get_data, n_parts, n_dims, bounds_low, bounds_high;
    ni     = 5,
    batch_size = 20,
    n_iters    = 200,
    frac_local = 0.75)

Fit PSVGP to data for distributed computation
    
    get_data:                function to define how a local partition should read in its data
    ni:                      number of local inducing points
    batch_size:              size of batches for stochastic optimizer
    n_iters:                 number of optimization steps
    frac_local:              what fraction of steps should a cell query neighbors

WARNING: Currently writes out several jld2 files containing local GPs
"""
function mpifit_psvgp(get_data, n_parts, n_dims, bounds_low, bounds_high;
        ni         = 5,               # Number of input dimensions
        batch_size = 5,              # Number of points used in updating local ELBO
        n_iters    = 2000,
        frac_local = 0.2
        )

    #TODO:
    #     Need to clean up saving out of local and global objects at the end.
    #     It's messy now and will get worse with many MPI processes.
    #
    #     Would be nice to write a function for prediction to identify which
    #     partition a new value of X would be in, then direct the prediction to
    #     that local svgp
    #
    #     Need to generalize the interface here. Should have a way to pass in
    #     the structure of data to be worked with. Maybe pass in a get_data
    #     function that returns the x,y data for each partition.
    #
    #     Would like to clean up some of those arguments also. I dont think
    #     we should need to have to pass all that stuff in, but not sure the
    #     best way to give global bounds information without simply specifying
    #     it for cases where all the data cannot be stored on one node.
    #

    MPI.Init()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    mysize = MPI.Comm_size(comm)
    
    # Fixing this for Gaussian case for now. 
    #     Need to generalize this for the future but that will come in part of a large generalization of the code
    n_latent = 1 
    
    # Since Julia is 1-indexed and not 0, making variable for array indexing
    rank_ind = myrank + 1
    
    grad_clip = 1.e-3

    part_split = [[ii] for ii in 1:mysize]
    part_nbors = [[[ii - 1, ii + 1, ii + 20, ii - 20]] for ii in 1:mysize]

    for ii in 1:mysize
        if mod(ii,20) == 1
            filter!(aa -> aa != (ii - 1), part_nbors[ii][1])
        elseif mod(ii,20) == 0
            filter!(aa -> aa != (ii + 1), part_nbors[ii][1])
        end
        if ii <= 20
            filter!(aa -> aa != (ii - 20), part_nbors[ii][1])
        elseif ii + 20 > 400
            filter!(aa -> aa != (ii + 20), part_nbors[ii][1])
        end
    end

    if n_parts > mysize
        for ii = (mysize+1):n_parts
            nbor_inds = [ii - 1, ii + 1, ii + 20, ii - 20]
            if mod(ii,20) == 1
                filter!(aa -> aa != (ii - 1), nbor_inds)
            elseif mod(ii,20) == 0
                filter!(aa -> aa != (ii + 1), nbor_inds)
            end
            if ii <= 20
                filter!(aa -> aa != (ii - 20), nbor_inds)
            elseif ii + 20 > 400
                filter!(aa -> aa != (ii + 20), nbor_inds)
            end
            append!(part_split[1 + ((ii-1) % mysize)], ii)
            append!(part_nbors[1 + ((ii-1) % mysize)], [nbor_inds])            
        end
    end

    ######################## DATA SPLIT ########################################
    # This part need not be done separately. Data will already be stored on nodes

    # Root process is the global node and rest of the processes are local processes
    # datarow goes from 2 to 48601
    # numrowscsv = 48600
    # numrowseachrank = Int(numrowscsv/(mysize-1))


    data_array = [get_data(part) for part in part_split[rank_ind]]
    len        = length(part_split[rank_ind])

    part_n = deepcopy(part_nbors)

    for ii in 1:len
        for kk in 1:length(part_nbors[rank_ind][ii])
            is_request, recv_id, tag_ind = check_for_query(comm)
            while is_request
                dummy   = [0]
                # Receive message to clear out
                rmsg    = MPI.Recv!(dummy, recv_id, tag_ind, comm) #TODO: Do this better. Shouldn't need to send dummy data.
                send_n  = size(data_array[tag_ind][1])[1]
                smsg    = MPI.Send([send_n], recv_id, length(part_split[recv_id+1])+1, comm)
                is_request, recv_id, tag_ind = check_for_query(comm)
            end

            nbor = part_nbors[rank_ind][ii][kk]
            if nbor in part_split[rank_ind]
                part_ind = filter(aa -> part_split[rank_ind][aa] == nbor, 1:size(part_split[rank_ind])[1])[1]
                part_n[rank_ind][ii][kk]  = size(data_array[part_ind][1])[1]
            else
                nbor_ind = filter(aa -> nbor in part_split[aa], 1:size(part_split)[1])[1]
                part_ind = filter(aa -> part_split[nbor_ind][aa] == nbor, 1:size(part_split[nbor_ind])[1])[1]
                # Query neighbor for data
                smsg     = MPI.Send([0], nbor_ind - 1, part_ind, comm)
                got_data = false
                while !got_data
                    is_request, recv_id, tag_ind = check_for_query(comm)
                    if is_request
                        if tag_ind < len+1 # if < len+1 then tag is a partition label
                            dummy   = [0]
                            # Receive message to clear out
                            rmsg    = MPI.Recv!(dummy, recv_id, tag_ind, comm) #TODO: Do this better. Shouldn't need to send dummy data.
                            send_n  = size(data_array[tag_ind][1])[1]
                            smsg    = MPI.Send([send_n], recv_id, length(part_split[recv_id+1])+1, comm)
                        end
                        if tag_ind > len
                            recv_n  = [0]
                            rmsg    = MPI.Recv!(recv_n, recv_id, len+1, comm)
                            got_data = true
                            part_n[rank_ind][ii][kk] = recv_n[1]
                        end # tag_ind > len
                    end # is_request
                end # !got_data
            end
        end
    end

    barrier_req = MPI.Ibarrier(comm)

    all_done = false
    while !all_done
        is_request, recv_id, tag_ind = check_for_query(comm)
        if is_request
            if tag_ind < len+1 # if < len+1 then tag is a partition label
                dummy   = [0]
                # Receive message to clear out
                rmsg    = MPI.Recv!(dummy, recv_id, tag_ind, comm) #TODO: Do this better. Shouldn't need to send dummy data.
                send_n  = size(data_array[tag_ind][1])[1]
                smsg    = MPI.Send([send_n], recv_id, length(part_split[recv_id+1])+1, comm)
            end
            if tag_ind > len
                println("Shouldnt get a message sending data! Something is wrong!")
            end # tag_ind > len
        end # is_request
        all_done, barrier_status = MPI.Test!(barrier_req)
    end # !all_done

    # Initializing "trace" array to store ELBO values
    trace_l = [zeros(n_iters) for ii in 1:len] # To collect the local ELBO value

    # Now that each local node has seen the data, it will send xi_low, cov_low,
    # mean_low to global node.
    # Once the global node gets data from local nodes, it will evaluate xi_high,
    # cov_high, mean_high and theta_high
    # Following that, global node will calculate prior_mean and prior_cov and
    # send it back to local nodes.


    ###################### INITIALIZATION ######################################

    @time begin
        local_svgps = [Inference_obj(data_array[ii][1], float(data_array[ii][2]), ni) for ii in 1:len]
    
    
        # Step 3
        # Initialize local optimizers
        lopt_cmean = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075), Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
        lopt_lrho  = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075), Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
        lopt_lkap  = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075), Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
        lopt_lsig  = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075), Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
        loptm      = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  , Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
        loptS      = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) , Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
        loptx      = [[Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  , Flux.Optimise.InvDecay(0.01)) for jj in 1:n_latent] for ii in 1:len]
    
    
        ###################### END OF INITIALIZATION ###############################
    
    
    
    
        ##################### OUTER LOOP ###########################################
    
        for iter= 1:n_iters
            # if mod(iter, 10) == 0
            #     print(iter, " ")
            # end
            # println("Iteration: ", iter)
            batch_x = 0.
            batch_y = 0.
    
            for ii in 1:len 
                # Step 4
                # Check to see if a neighbor needs data from node
                is_request, recv_id, tag_ind = check_for_query(comm)
                while is_request
                    dummy   = [0]
                    # Receive message to clear out
                    rmsg    = MPI.Recv!(dummy, recv_id, tag_ind, comm) #TODO: Do this better. Shouldn't need to send dummy data. 
    
                    inds    = rand(1:length(local_svgps[tag_ind].data.y), batch_size)
                    send_x  = local_svgps[tag_ind].data.x[inds,:]
                    send_y  = local_svgps[tag_ind].data.y[inds]
                    smsg    = MPI.Send(send_x, recv_id, length(part_split[recv_id+1])+1, comm)
                    smsg    = MPI.Send(send_y, recv_id, length(part_split[recv_id+1])+2, comm)
                    # println(string(rank_ind-1, " sending to ", recv_id, " requested with tag ", tag_ind, " first value of ", send_y[1]))
                    is_request, recv_id, tag_ind = check_for_query(comm)
                end
                
                # Step 5
                # Select either local data or neighbor data
                # TODO: REMOVE - THIS TRANSFORMATION IS ONLY FOR BACKWARD COMPATIBILITY FOR STUDY SCRIPTS
                frac_local_t = (frac_local - 1.0) / (0.2 - 1.0) # REMOVE THE NEED FOR THIS AFTER STUDY
                eff_n        = sum(part_n[rank_ind][ii]) * frac_local_t + local_svgps[ii].data.n
                samp_weights = (part_n[rank_ind][ii] .* frac_local_t) / eff_n
                rands        = rand(batch_size)
                select_rand  = [length(filter(bbb -> cumsum(samp_weights)[bbb] > x, 1:length(samp_weights))) > 0 ? filter(bbb -> cumsum(samp_weights)[bbb] > x, 1:length(samp_weights))[1] : 0 for x in rands]
                batch_x    = zeros(batch_size, size(local_svgps[ii].data.x)[2])
                batch_y    = zeros(batch_size)
                local_size = 0
                if any(select_rand .== 0)
		    local_size = sum(select_rand .== 0)
                    # inds           = rand(1:size(local_svgps[ii].data.y)[1], batch_size)
                    if local_size > local_svgps[ii].data.n
                        inds    = rand(1:size(local_svgps[ii].data.y)[1], local_size)
                    else
                        inds    = randperm(size(local_svgps[ii].data.x)[1])[1:local_size]
                    end
                    batch_x[1:local_size, :] = local_svgps[ii].data.x[inds,:]
                    batch_y[1:local_size]    = local_svgps[ii].data.y[inds]
                end
                if any(select_rand .!= 0)
                    rind = filter(aaa -> aaa > 0, select_rand)
                    nbor = part_nbors[rank_ind][ii][rind]
                    ubor = unique(nbor)
                    nums = [sum(nbor .== uu) for uu in ubor]
                    for parts in 1:length(ubor)
                        ibor = ubor[parts]
                        if ibor in part_split[rank_ind]
                            part_ind = filter(aa -> part_split[rank_ind][aa] == ibor, 1:size(part_split[rank_ind])[1])[1]
                            if nums[parts] > local_svgps[part_ind].data.n
                                inds = rand(1:size(local_svgps[part_ind].data.y)[1], nums[parts])
                            else
                                inds = randperm(size(local_svgps[part_ind].data.x)[1])[1:nums[parts]]
                            end
                            batch_x[(local_size+1):(local_size+nums[parts]),:] = local_svgps[part_ind].data.x[inds,:]
                            batch_y[(local_size+1):(local_size+nums[parts])]   = local_svgps[part_ind].data.y[inds]
                            local_size += nums[parts]
                        else
                            nbor_ind = filter(aa -> ibor in part_split[aa], 1:size(part_split)[1])[1]
                            part_ind = filter(aa -> part_split[nbor_ind][aa] == ibor, 1:size(part_split[nbor_ind])[1])[1]
                            # Query neighbor for data
                            smsg     = MPI.Send([0], nbor_ind - 1, part_ind, comm) 
                            got_data = false
                            # println(string(rank_ind-1," partition ",ii," querying ", nbor_ind - 1, " with tag ", part_ind))
                            while !got_data
                                is_request, recv_id, tag_ind = check_for_query(comm)
                                if is_request
                                    if tag_ind < len+1 # if < len+1 then tag is a partition label
                                        dummy   = [0]
                                        # Receive message to clear out
                                        rmsg    = MPI.Recv!(dummy, recv_id, tag_ind, comm) #TODO: Do this better. Shouldn't need to send dummy data. 
                                        if batch_size > local_svgps[tag_ind].data.n
                                            inds    = rand(1:size(local_svgps[tag_ind].data.y)[1], batch_size)
                                        else
                                            inds    = randperm(size(local_svgps[tag_ind].data.x)[1])[1:batch_size]
                                        end
                                        send_x  = local_svgps[tag_ind].data.x[inds,:]
                                        send_y  = local_svgps[tag_ind].data.y[inds]
                                        smsg    = MPI.Send(send_x, recv_id, length(part_split[recv_id+1])+1, comm)
                                        smsg    = MPI.Send(send_y, recv_id, length(part_split[recv_id+1])+2, comm)
                                        # println(string(rank_ind-1, " sending to ", recv_id, " requested with tag ", tag_ind, " first value of ", send_y[1]))
                                    end
                                    if tag_ind > len
                                        xrecv  = zeros(batch_size, n_dims)
                                        rmsg   = MPI.Recv!(xrecv, recv_id, len+1, comm)
    
                                        yrecv  = zeros(batch_size)
                                        rmsg   = MPI.Recv!(yrecv, recv_id, len+2, comm)
                                        got_data = true

                                        batch_x[(local_size+1):(local_size+nums[parts]),:] = xrecv[1:nums[parts],:]
                                        batch_y[(local_size+1):(local_size+nums[parts])]   = yrecv[1:nums[parts]]
                                        
                                        local_size += nums[parts]
                                        # println(string(rank_ind-1, " got data from ", recv_id, " with first value of ", batch_y[1]))
                                    end # tag_ind > len
                                end # is_request
                            end # !got_data
                        end # nbor in part_split[rank_ind]
                    end # for 
                end # select_local
                
                grads  = gradient(gps -> inference_elbo(
                        batch_x, 
                        batch_y, 
                        eff_n, 
                        gps
                        ), local_svgps[ii])[1]

                for jj in 1:n_latent
                    Flux.Optimise.update!(lopt_cmean[ii][jj], local_svgps[ii].params[jj].const_mean,    -grads.params[jj].const_mean)
                    Flux.Optimise.update!(lopt_lrho[ii][jj],  local_svgps[ii].params[jj].log_rho,       -grads.params[jj].log_rho)
                    Flux.Optimise.update!(lopt_lkap[ii][jj],  local_svgps[ii].params[jj].log_kappa,     -grads.params[jj].log_kappa)
                    Flux.Optimise.update!(lopt_lsig[ii][jj],  local_svgps[ii].params[jj].log_sigma,     -grads.params[jj].log_sigma)
                    Flux.Optimise.update!(loptm[ii][jj],      local_svgps[ii].params[jj].inducing_mean, -grads.params[jj].inducing_mean)
                    Flux.Optimise.update!(loptS[ii][jj],      local_svgps[ii].params[jj].inducing_L,    -LowerTriangular(grads.params[jj].inducing_L) )
                    Flux.Optimise.update!(loptx[ii][jj],      local_svgps[ii].params[jj].inducing_locs, -grads.params[jj].inducing_locs)
                end
                # end
                # Collecting local ELBO in an array ###########################
                trace_l[ii][iter] = inference_elbo(
                    batch_x,
                    batch_y,
                    local_svgps[ii].data.n,
                    local_svgps[ii]
                )
                # println("local ELBO optimized")
            end
    
        end # outeriter
        
        barrier_req = MPI.Ibarrier(comm)
        
        all_done = false
        while !all_done
            is_request, recv_id, tag_ind = check_for_query(comm)
            if is_request
                if tag_ind < len+1 # if < len+1 then tag is a partition label
                    dummy   = [0]
                    # Receive message to clear out
                    rmsg    = MPI.Recv!(dummy, recv_id, tag_ind, comm) #TODO: Do this better. Shouldn't need to send dummy data.
                    inds    = rand(1:size(local_svgps[tag_ind].data.y)[1], batch_size)
                    send_x  = local_svgps[tag_ind].data.x[inds,:]
                    send_y  = local_svgps[tag_ind].data.y[inds]
                    smsg    = MPI.Send(send_x, recv_id, length(part_split[recv_id+1])+1, comm)
                    smsg    = MPI.Send(send_y, recv_id, length(part_split[recv_id+1])+2, comm)
                end
                if tag_ind > len
                    println("Shouldnt get a message sending data! Something is wrong!")
                end # tag_ind > len
            end # is_request
            all_done, barrier_status = MPI.Test!(barrier_req)
        end # !all_done 
    end

    ############################################################################
    
    store_trace = zeros(n_iters, n_parts)

    # Collect all traces to write out at once
    if myrank != 0
        for ii in 1:len
            send_trace(trace_l[ii], 0 ,comm)
        end

    end #myrank != 0

    if myrank == 0
        for ii in 1:len
            store_trace[:, part_split[1][ii] ] = trace_l[ii]
        end

        for proc = 2:mysize
            for ii in 1:size(part_split[proc])[1]
                trace_l = receive_trace(n_iters, proc - 1, comm)
                store_trace[:, part_split[proc][ii] ] = trace_l
            end
        end

        # Write trace to a csv file
        df_return_trace = DataFrame(store_trace, :auto)
        CSV.write(string("./return_trace",mysize,".csv"), df_return_trace)

    end # myrank == 0
    ############################################################################

    ############################################################################
    # Save svgp_params
    # send_svgp_params(local_svgp.params, root ,comm)
    for ii in 1:len
        save(string("local_fitted_parameters",part_split[rank_ind][ii],"_",mysize, ".jld2"), string("local_svgp",part_split[rank_ind][ii]), local_svgps[ii])
        save(string("local_fitted_parameters",part_split[rank_ind][ii],"_",mysize, ".jld2"), string("local_svgp",part_split[rank_ind][ii]), local_svgps[ii])
    end

    ############################################################################

    MPI.Finalize()

end     # mpifit_HSVGP
