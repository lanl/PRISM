"""
    fit_svgp!(svgp::SVGP_obj; n_iters=10000, batch_size=100, handoff = 1e20, return_param_traces = false)
Fit SVGP to data and update SVGP_obj in place. 
    
    n_iters:                 number of optimization steps
    batch_size:              size of batches for stochastic optimizer
    handoff:                 does not work!! Keep larger than n_iters for now
    return_parameter_traces: flag to indicate whether to return traces for all parameters or not
"""
function fit_svgp!(svgp::SVGP_obj; n_iters=10000, batch_size=100, handoff = 1e20, return_param_traces = false)
    # Initialize optimizers
    opt_lrho = ADAM(1. * 0.0075)     #     opt  = ADAM(0.0075, (0.5, 0.8))
    opt_lkap = ADAM(1. * 0.0075)     #     opt  = ADAM(0.0075, (0.5, 0.8))
    opt_lsig = ADAM(1. * 0.0075)     #     opt  = ADAM(0.0075, (0.5, 0.8))
    optm     = ADAM(1. * 0.01)       #     optm = ADAM(0.02, (0.5, 0.8))
    optS     = ADAM(1. * 0.002)      #     optS = ADAM(0.002, (0.5, 0.8))
    optx     = ADAM(1. * 0.01)       #     optx = ADAM(0.01, (0.5, 0.8))

    # Will optimize Cholesky to ensure positive definite
    trace_elbo = zeros(n_iters)

    if return_param_traces
        trace_xi     = zeros( n_iters, size(svgp.params.inducing_locs)[1], size(svgp.params.inducing_locs)[2] )
        trace_mn     = zeros( n_iters, size(svgp.params.inducing_mean)[1] )
        trace_cov    = zeros( n_iters, size(svgp.params.inducing_L)[1], size(svgp.params.inducing_L)[2] )
        trace_lrho   = zeros( n_iters, size(svgp.params.log_rho)[2] )
        trace_lkappa = zeros( n_iters )
        trace_lsigma = zeros( n_iters )
    end

    for t = 1:n_iters
        if t == handoff
            opt_lrho = Descent(0.05)
            opt_lkap = Descent(0.05)
            opt_lsig = Descent(0.05)
            optm     = Descent(0.05)
            optS     = Descent(0.05)
            optx     = Descent(0.05)
        end
        inds   = rand(1:svgp.data.n, batch_size)
        grads  = gradient(gp -> svgp_elbo(svgp.data.x[inds,:], svgp.data.y[inds], gp, svgp), svgp.params)[1][]

        Flux.Optimise.update!(opt_lrho, svgp.params.log_rho,       -grads.log_rho)
        Flux.Optimise.update!(opt_lkap, svgp.params.log_kappa,     -grads.log_kappa)
        Flux.Optimise.update!(opt_lsig, svgp.params.log_sigma,     -grads.log_sigma)
        Flux.Optimise.update!(optm,     svgp.params.inducing_mean, -grads.inducing_mean)
        Flux.Optimise.update!(optS,     svgp.params.inducing_L,    -LowerTriangular(grads.inducing_L) )
        Flux.Optimise.update!(optx,     svgp.params.inducing_locs, -grads.inducing_locs)

        trace_elbo[t]        = svgp_elbo(svgp.data.x[inds,:], svgp.data.y[inds], svgp.params, svgp)
        if return_param_traces
            trace_xi[t, :, :]  = svgp.params.inducing_locs
            trace_mn[t, :]     = svgp.params.inducing_mean
            trace_cov[t, :, :] = Hermitian(svgp.params.inducing_L * transpose(svgp.params.inducing_L))
        end
    end

    if return_param_traces
        ret_trace = Dict([
                ("inducing_mean",    trace_mn),
                ("inducing_cov",    trace_cov),
                ("inducing_locs",    trace_xi),
                ])
        return trace_elbo, ret_trace
    end

    return trace_elbo
end


"""
    fit_hsvgp!(hsvgp::HSVGP_obj; n_iters=10000, batch_size=100, rep_cycles = 1, handoff = 1e20, return_param_traces = false)
Fit HSVGP to data and update HSVGP_obj in place. 
    
    n_iters:                 number of optimization steps
    batch_size:              size of batches for stochastic optimizer
    rep_cycles:              number of updates to do locally before communicated between layers of hierarchy
    handoff:                 does not work!! Keep larger than n_iters for now
    return_parameter_traces: flag to indicate whether to return traces for all parameters or not
"""
function fit_hsvgp!(hsvgp::HSVGP_obj; n_iters=10000, batch_size=100, rep_cycles = 1, handoff = 1e20, return_param_traces = false)
    n_parts  = hsvgp.n_parts

    # Initialize optimizers
    opt_lrho = ADAM(1. * 0.0075)     #     opt  = ADAM(0.0075, (0.5, 0.8))
    opt_lkap = ADAM(1. * 0.0075)     #     opt  = ADAM(0.0075, (0.5, 0.8))
    opt_lsig = ADAM(1. * 0.0075)     #     opt  = ADAM(0.0075, (0.5, 0.8))
    optm     = ADAM(1. * 0.01)       #     optm = ADAM(0.02, (0.5, 0.8))
    optS     = ADAM(1. * 0.002)      #     optS = ADAM(0.002, (0.5, 0.8))
    optx     = ADAM(1. * 0.01)       #     optx = ADAM(0.01, (0.5, 0.8))

    # Initialize local optimizers
    lopt_lrho = [ADAM(1. * 0.0075)  for ii in 1:n_parts]     #     opt  = ADAM(0.0075, (0.5, 0.8))
    lopt_lkap = [ADAM(1. * 0.0075)  for ii in 1:n_parts]     #     opt  = ADAM(0.0075, (0.5, 0.8))
    lopt_lsig = [ADAM(1. * 0.0075)  for ii in 1:n_parts]     #     opt  = ADAM(0.0075, (0.5, 0.8))
    loptm     = [ADAM(1. * 0.01)    for ii in 1:n_parts]     #     optm = ADAM(0.02, (0.5, 0.8))
    loptS     = [ADAM(1. * 0.002)   for ii in 1:n_parts]     #     optS = ADAM(0.002, (0.5, 0.8))
    loptx     = [ADAM(1. * 0.01)    for ii in 1:n_parts]     #     optx = ADAM(0.01, (0.5, 0.8))

    trace_h = zeros(n_iters)
    trace_l = [zeros(n_iters) for pp in 1:n_parts]

    for t = 1:n_iters
        for rr in 1:rep_cycles
            for pp in 1:n_parts
                inds   = rand(1:hsvgp.local_svgps[pp].data.n, batch_size)
                grads  = gradient(gp -> svgp_elbo_local(
                        hsvgp.local_svgps[pp].data.x[inds,:],
                        hsvgp.local_svgps[pp].data.y[inds],
                        gp,
                        hsvgp.local_svgps[pp],
                        pred_vgp(hsvgp.local_svgps[pp].params.inducing_locs, hsvgp.global_obj)
                        ), hsvgp.local_svgps[pp].params)[1][]

                Flux.Optimise.update!(lopt_lrho[pp], hsvgp.local_svgps[pp].params.log_rho,       -grads.log_rho)
                Flux.Optimise.update!(lopt_lkap[pp], hsvgp.local_svgps[pp].params.log_kappa,     -grads.log_kappa)
                Flux.Optimise.update!(lopt_lsig[pp], hsvgp.local_svgps[pp].params.log_sigma,     -grads.log_sigma)
                Flux.Optimise.update!(loptm[pp],     hsvgp.local_svgps[pp].params.inducing_mean, -grads.inducing_mean)
                Flux.Optimise.update!(loptS[pp],     hsvgp.local_svgps[pp].params.inducing_L,    -LowerTriangular(grads.inducing_L) )
                Flux.Optimise.update!(loptx[pp],     hsvgp.local_svgps[pp].params.inducing_locs, -grads.inducing_locs)
                trace_l[pp][t] = svgp_elbo_local(
                    hsvgp.local_svgps[pp].data.x[inds,:],
                    hsvgp.local_svgps[pp].data.y[inds],
                    hsvgp.local_svgps[pp].params,
                    hsvgp.local_svgps[pp],
                    pred_vgp(hsvgp.local_svgps[pp].params.inducing_locs, hsvgp.global_obj)
                    )
            end
        end

        for rr in 1:rep_cycles
            for pp in 1:n_parts
                grads  = gradient(gp -> svgp_elbo_global(
                        hsvgp.local_svgps[pp].params.inducing_locs,
                        hsvgp.local_svgps[pp].params.inducing_mean,
                        gp,
                        hsvgp.global_obj
                        ), hsvgp.global_obj.params)[1][]

                Flux.Optimise.update!(opt_lrho, hsvgp.global_obj.params.log_rho,       -grads.log_rho)
                Flux.Optimise.update!(opt_lkap, hsvgp.global_obj.params.log_kappa,     -grads.log_kappa)
                Flux.Optimise.update!(opt_lsig, hsvgp.global_obj.params.log_sigma,     -grads.log_sigma)
                Flux.Optimise.update!(optm,     hsvgp.global_obj.params.inducing_mean, -grads.inducing_mean)
                Flux.Optimise.update!(optS,     hsvgp.global_obj.params.inducing_L,    -LowerTriangular(grads.inducing_L) )
                Flux.Optimise.update!(optx,     hsvgp.global_obj.params.inducing_locs, -grads.inducing_locs)
            end
        end
        trace_h[t] = sum([svgp_elbo(
                    hsvgp.local_svgps[pp].params.inducing_locs,
                    hsvgp.local_svgps[pp].params.inducing_mean,
                    hsvgp.global_obj.params,
                    hsvgp.global_obj
                    ) for pp in 1:n_parts])
    end

    return trace_h, trace_l
end
