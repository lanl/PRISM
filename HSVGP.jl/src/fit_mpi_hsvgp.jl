"""
    mpifit_hsvgp(get_data, n_parts, n_dims, bounds_low, bounds_high;
    ni_low     = 5,
    ni_high    = 50,
    batch_size = 20,
    n_iters    = 200,
    n_subcycle_ELBO_local  = 20,
    n_subcycle_ELBO_global = 20)

Fit HSVGP to data for distributed computation
    
    get_data:                function to define how a local partition should read in its data
    ni_low:                  number of local inducing points
    ni_high:                 number of global inducing points
    batch_size:              size of batches for stochastic optimizer
    n_iters:                 number of optimization steps
    n_subcycle_ELBO_local:   subcycles for local  ELBO before MPI communication
    n_subcycle_ELBO_global:  subcycles for global ELBO before MPI communication
    n_batch:                 number of partitions in a local to global communication batch

WARNING: Currently writes out several jld2 files containing local and global GPss
"""
function mpifit_hsvgp(get_data, n_parts, n_dims, bounds_low, bounds_high;
    ni_low     = 5,               # Number of local inducing points
    ni_high    = 50,              # Number of global inducing points
    batch_size = 20,              # Number of points used in updating local ELBO
    n_iters    = 200,             # Total updates in whole cycle
    n_subcycle_ELBO_local  = 20,  # subcycles for local ELBO before comm.
    n_subcycle_ELBO_global = 20,  # subcycles for global ELBO before comm.
    n_batch    = 5)               # Number of partitions selected at each iteration
    
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
    
    grad_clip = 1.e-3

    MPI.Init()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    mysize = MPI.Comm_size(comm)

    root = 0 # global node

    # Fixing this for Gaussian case for now. 
    #     Need to generalize this for the future but that will come in part of a large generalization of the code
    # n_latent = 1 
    
    total_inducing = n_parts * ni_low
    
    part_split = [[ii] for ii in 1:(mysize-1)]

    if n_parts > mysize - 1
        for ii = mysize:n_parts
            append!(part_split[1 + ((ii-1) % (mysize - 1))], ii)
        end
    end

    ######################## DATA SPLIT ########################################
    # This part need not be done separately. Data will already be stored on nodes

    # Root process is the global node and rest of the processes are local processes
    # datarow goes from 2 to 48601
    # numrowscsv = 48600
    # numrowseachrank = Int(numrowscsv/(mysize-1))

    if myrank != root

        data_array = [get_data(part) for part in part_split[myrank]]
        len        = length(part_split[myrank])


    end # rank != root

    # Initializing "trace" array to store ELBO values
    if myrank == root
        trace_h       = zeros(n_iters)          # To collect the global ELBO value
        store_trace_l = zeros(n_iters,n_parts) # Store trace_l from all proesses
    end
    if myrank != root
        trace_l = [zeros(n_iters) for ii in 1:len] # To collect the local ELBO value
    end

    # Now that each local node has seen the data, it will send xi_low, cov_low,
    # mean_low to global node.
    # Once the global node gets data from local nodes, it will evaluate xi_high,
    # cov_high, mean_high and theta_high
    # Following that, global node will calculate prior_mean and prior_cov and
    # send it back to local nodes.


    ###################### INITIALIZATION ######################################


    if myrank != root
        # Step 1
        # Build local SVGPs ###################
        local_svgps = [SVGP_obj(data_array[ii][1], float(data_array[ii][2]), ni_low) for ii in 1:len]

        for ii in 1:len
            # Step 2
            # Sending mean_factor, xi_low, cov_low, mean_low and N to global #######
            send_inducing_points_to_global(local_svgps[ii].params.inducing_locs, local_svgps[ii].params.inducing_mean, root, comm)
        end

        # Step 3
        # Initialize local optimizers
        lopt_cmean = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:len]
        lopt_lrho  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:len]
        lopt_lkap  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:len]
        lopt_lsig  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:len]
        loptm      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  ) for ii in 1:len]
        loptS      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) ) for ii in 1:len]
        loptx      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  ) for ii in 1:len]

    end # myrank != root

    if myrank == root
        wts = StatsBase.AnalyticWeights(ones(n_parts));

        # Step 3
            # Initialize global svgp and optimizers
            # Root process will receive inducing point data
            # Root process will update global parameters
            # Root process will calculate prior_mean prior_cov for local nodes
            # Root process will send prior_mean prior_cov back to local processes

        # Initialize global optimizers
        opt_cmean = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
        opt_lrho  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
        opt_lkap  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
        opt_lsig  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
        optm      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )
        optS      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) )
        optx      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )

        # Initialize global svgp
        # TODO:
        #     Need bounds on input space and some idea of scale for output
        #         to set initial GP variance parameter and
        #         Latin Hypercube for inducing point locations
        #
        #     Assuming we will work in the case that we can't send all inducing point
        #         data to global. Instead should probably get bounds as arguments to original
        #         function or otherwise.
        fake_data_to_init = SVGP_data(
            vcat(bounds_low', bounds_high'),
            randn(4)*10 # sampling 4 points with something like the variance I was to initialize at
        )
        global_svgp = SVGP_obj(ni_high, SVGP_data("global"), SVGP_params(fake_data_to_init, ni_high), "gaussian")
        
        for part = (root+1):n_parts
            recv_id = 1 + (part - 1) % (mysize - 1)
            # TODO: Would prefer to not do this sequentially by rank number but as communications arrive
            #

            # println("In Process(1): ",recv_id,"\n")

            # Receiving xi_low and yi_low #
            xi_low, yi_low = receive_inducing_points_in_global(ni_low, n_dims, recv_id, comm) ### TODO: IMPORTANT! Need to communicate which partition is being recieved

            # Update global parameters from local data #
            grads  = gradient(gps -> svgp_elbo_global(
                    xi_low, 
                    yi_low, 
                    gps, 
                    global_svgp
                    ), global_svgp.params)[1][]

            Flux.Optimise.update!(opt_cmean, global_svgp.params.const_mean,    -grads.const_mean)
            Flux.Optimise.update!(opt_lrho,  global_svgp.params.log_rho,       -grads.log_rho)
            Flux.Optimise.update!(opt_lkap,  global_svgp.params.log_kappa,     -grads.log_kappa)
            Flux.Optimise.update!(opt_lsig,  global_svgp.params.log_sigma,     -grads.log_sigma)
            Flux.Optimise.update!(optm,      global_svgp.params.inducing_mean, -grads.inducing_mean)
            Flux.Optimise.update!(optS,      global_svgp.params.inducing_C,    -grads.inducing_C)
            Flux.Optimise.update!(optx,      global_svgp.params.inducing_locs, -grads.inducing_locs)
            # Sending prior_mean, prior_cov to local process ###################
            send_prior_to_local( pred_vgp(xi_low, global_svgp.params), recv_id, comm)

        end # for loop
    end
    
    MPI.Barrier(comm)
    if myrank != root 
        #Get prior information from globa 
        prior_mean, prior_sd = receive_prior_in_local(
                                                        ni_low,
                                                        root,
                                                        comm)
    end # myrank == root

    ###################### END OF INITIALIZATION ###############################




    ##################### OUTER LOOP ###########################################

    for iter= 1:n_iters
        if mod(iter, 2000) == 0
            print(iter, " ")
        end

        if myrank != root

            for ii in 1:len ## TODO: Clean this up with Kelin to process non-sequentially
                            ## TODO: ENSURE THE MPI PROPERLY ASSOCIATED DATA COMMUNICATED TO CORRECT PARTITION
                # Step 4
                # Each local process will receive prior_mean and prior_cov

                # Receiving prior_mean, prior_cov to local process #################
                # prior_mean, prior_sd = receive_prior_in_local(
                #                                        ni_low,
                #                                        root,
                #                                        comm)
                # println("prior_mean, prior_cov received in local")

                # Step 5
                # Update local svgp using prior_mean prior_cov from global
                nsteps_curr  = rand(Distributions.Poisson(n_subcycle_ELBO_local)) + 3;
                for sub_iter = 1:nsteps_curr
                    inds   = rand(1:size(local_svgps[ii].data.y)[1], batch_size)

                    # Update global parameters from local data #
                    grads  = gradient(gps -> svgp_elbo_local(
                            local_svgps[ii].data.x[inds,:], 
                            local_svgps[ii].data.y[inds], 
                            gps,
                            local_svgps[ii],
                            (prior_mean, prior_sd)
                            ), local_svgps[ii].params)[1][]

                    Flux.Optimise.update!(lopt_cmean[ii], local_svgps[ii].params.const_mean,    -grads.const_mean)
                    Flux.Optimise.update!(lopt_lrho[ii],  local_svgps[ii].params.log_rho,       -grads.log_rho)
                    Flux.Optimise.update!(lopt_lkap[ii],  local_svgps[ii].params.log_kappa,     -grads.log_kappa)
                    Flux.Optimise.update!(lopt_lsig[ii],  local_svgps[ii].params.log_sigma,     -grads.log_sigma)
                    Flux.Optimise.update!(loptm[ii],      local_svgps[ii].params.inducing_mean, -grads.inducing_mean)
                    Flux.Optimise.update!(loptS[ii],      local_svgps[ii].params.inducing_C,    -grads.inducing_C)
                    Flux.Optimise.update!(loptx[ii],      local_svgps[ii].params.inducing_locs, -grads.inducing_locs)
                    # # Sending prior_mean, prior_cov to local process ###################
                    # send_prior_to_local( pred_vgp(xi_low, global_svgp.params[jj]), recv_id, comm)
                    
                    if sub_iter == nsteps_curr
                        # Collecting local ELBO in an array ###########################
                        trace_l[ii][iter] = svgp_elbo_local(
                            local_svgps[ii].data.x[inds,:],
                            local_svgps[ii].data.y[inds],
                            local_svgps[ii].params,
                            local_svgps[ii],
                            (prior_mean, prior_sd)
                        )
                    end # if sub_iter                  
                end # for sub_iter
                # println("local ELBO optimized")
                
                # THIS IS NOW DONE OUTSIDE THE ii LOOP
                # Step 6
                # Sending inducing point data to global
                # for jj in 1:n_latent
                #     send_inducing_points_to_global(local_svgps[ii].params[jj].inducing_locs, local_svgps[ii].params[jj].inducing_mean, root, comm)
                # end
            end # for ii
            
            # Kelin block for sending inducing point data to global only if you were selected
            is_message, status = MPI.Iprobe(0, 8, comm)                        
            is_message, status = MPI.Iprobe(0, 8, comm) # Kelin: Silly solution, but try this for now
            
            msg  = [1];
            msg2 = ones(1);

            if is_message
                MPI.Recv!(msg, 0, 8, comm);
                global msg = convert(Vector{Int64}, msg)
                global msg2 = convert(Vector{Int64}, ones(msg[1]));
                #println("process ", myrank, "recieved first message ", msg);
                MPI.Recv!(msg2, 0, 10+myrank, comm);
                global msg2 = msg2;
                #println("process ", myrank, "recieved second message ", msg2);
                
                indx = findall(in(part_split[myrank]), msg2)
                for ii in indx
                    send_inducing_points_to_global(local_svgps[ii].params.inducing_locs, local_svgps[ii].params.inducing_mean, root, comm)
                end     
               #println("process ", myrank, "sent inducing points")

               #Get prior information from global 
               prior_mean, prior_sd = receive_prior_in_local(
                                                        ni_low,
                                                        root,
                                                        comm)
            end # if is_message       
        end # if myrank != root

        if myrank == root
            #Select random local partition for batch
            aa = collect(1:n_parts);
            selection = convert(Vector{Int64}, zeros(n_batch));
            StatsBase.efraimidis_ares_wsample_norep!(aa, wts, selection)
            part_split_curr = [intersect(part_split[ii], selection) for ii in 1:(mysize-1)];
            wts[selection] .= 0;
            wts .+= 1;

            for ii in 1:(mysize-1)
               msg_to_send = part_split_curr[ii]
               if length(msg_to_send) > 0
                  MPI.Isend([length(msg_to_send)], ii, 8, comm) 
                  MPI.Isend(msg_to_send, ii, 10+ii, comm); 
                  #println("sending message ", msg_to_send, "to process ", ii)
               end
            end            
            
            step_elbo = 0.
            for part = 1:n_batch
                status  = MPI.Probe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, comm)
                xi_low  = zeros(ni_low, n_dims)
                
                rmsg    = MPI.Recv!(xi_low, status.source, 0, comm)
                yi_low  = zeros(ni_low)
                
                rmsg    = MPI.Recv!(yi_low, status.source, 1, comm)
                recv_id = status.source

                # Update global parameters from local data #
                for sub_iter = 1:n_subcycle_ELBO_global
                    grads  = gradient(gps -> svgp_elbo_global(
                            xi_low, 
                            yi_low, 
                            gps,
                            global_svgp
                            ), global_svgp.params)[1][]

                    Flux.Optimise.update!(opt_cmean, global_svgp.params.const_mean,    -grads.const_mean)
                    Flux.Optimise.update!(opt_lrho,  global_svgp.params.log_rho,       -grads.log_rho)
                    Flux.Optimise.update!(opt_lkap,  global_svgp.params.log_kappa,     -grads.log_kappa)
                    Flux.Optimise.update!(opt_lsig,  global_svgp.params.log_sigma,     -grads.log_sigma)
                    Flux.Optimise.update!(optm,      global_svgp.params.inducing_mean, -grads.inducing_mean)
                    Flux.Optimise.update!(optS,      global_svgp.params.inducing_C,    -grads.inducing_C)
                    Flux.Optimise.update!(optx,      global_svgp.params.inducing_locs, -grads.inducing_locs)
                end # for sub_iter
                # println("global ELBO optimized")

                # Update step_elbo
                step_elbo += svgp_elbo_global(xi_low, yi_low, global_svgp.params, global_svgp)

                # Sending prior_mean, prior_cov to local process ###################
                send_prior_to_local( pred_vgp(xi_low, global_svgp.params), recv_id, comm) # MAKE SURE COMM. INDICATES WHICH PARTITION IN A PROCESS THIS COMM IS FOR
            end # for part 

            trace_h[iter] = step_elbo
        end # myrank == root
        
        barrier_req = MPI.Ibarrier(comm);
        barrier_flag, barrier_status = MPI.Test!(barrier_req);
        while !barrier_flag
            is_message, status = MPI.Iprobe(0, 8, comm)                        
            is_message, status = MPI.Iprobe(0, 8, comm) #Silly solution, but try this for now
            msg  = [1];
            msg2 = ones(1);
            #println("process ", myrank, "message recieved? ", is_message)
            if is_message
                MPI.Recv!(msg, 0, 8, comm);
                global msg = convert(Vector{Int64}, msg);
                global msg2 = convert(Vector{Int64}, ones(msg[1]));
                #println("process ", myrank, "recieved first message ", msg);
                MPI.Recv!(msg2, 0, 10+myrank, comm);
                global msg2 = msg2;
                #println("process ", myrank, "recieved second message ", msg2);
                
                indx = findall(in(part_split[myrank]), msg2)
                #println("process ", myrank, "indx = ", indx);
                for ii in indx
                    send_inducing_points_to_global(local_svgps[ii].params.inducing_locs, local_svgps[ii].params.inducing_mean, root, comm)
                end     
                #println("process ", myrank, "sent inducing points")
                
                #Get prior information from global
                prior_mean, prior_sd = receive_prior_in_local(ni_low, root, comm)
            end 
            barrier_flag, barrier_status = MPI.Test!(barrier_req);
        end # while !barrier_flag
    end # outeriter

    ############################################################################
    # Returning trace_h and trace_l to main function
    if myrank != root

        for ii in 1:len
            send_trace(trace_l[ii], root ,comm)
        end

    end #myrank != root

    if myrank == root

        for part = (root+1):n_parts
            recv_id = 1 + (part - 1) % (mysize - 1)
            trace_l = receive_trace(n_iters, recv_id, comm)
            store_trace_l[:,part] = trace_l
        end # recv_id

        return_trace = hcat(trace_h, store_trace_l)
        # println(return_trace)

        # Write trace to a csv file
        df_return_trace = DataFrame(return_trace)
        CSV.write("./return_trace.csv", df_return_trace)

    end # myrank == root
    ############################################################################

    ############################################################################
    # Save svgp_params
    if myrank != root

        # send_svgp_params(local_svgp.params, root ,comm)
        for ii in 1:len
            save(string("local_fitted_parameters",part_split[myrank][ii],"_",mysize-1, ".jld2"), string("local_svgp",part_split[myrank][ii]), local_svgps[ii])
        end


    end #myrank != root

    if myrank == root

        # local_svgp_params = [receive_svgp_params(recv_id, comm) for recv_id = 1:mysize-1]

        save(string("global_fitted_svgp_",mysize-1,".jld2"), "global_svgp", global_svgp)

    end # myrank == root
    ############################################################################

    MPI.Finalize()

end     # mpifit_HSVGP
