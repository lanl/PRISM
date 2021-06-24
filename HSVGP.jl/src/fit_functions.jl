"""
    fit_inference!(inf_obj::Inference_obj; n_iters=10000, batch_size=100, handoff = 1e20)
Fit SVGP to data and update SVGP_obj in place. 
    
    n_iters:                 number of optimization steps
    batch_size:              size of batches for stochastic optimizer
    handoff:                 does not work!! Keep larger than n_iters for now
"""
function fit_inference!(inf_obj::Inference_obj; n_iters=10000, batch_size=100, handoff = 1e20)
    n_latent = length(inf_obj.params)
    
    grad_clip = 1.e-3
    # Initialize optimizers
    opt_cmean = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.02)  ) for ii in 1:n_latent]
    opt_lrho  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:n_latent]
    opt_lkap  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:n_latent]
    opt_lsig  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075)) for ii in 1:n_latent]
    optm      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  ) for ii in 1:n_latent]
    optS      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) ) for ii in 1:n_latent]
    optx      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  ) for ii in 1:n_latent]

    # Will optimize Cholesky to ensure positive definite
    trace_elbo = zeros(n_iters)

    for t = 1:n_iters
        if t == handoff
            opt_cmean = [Descent(0.05) for ii in 1:n_latent]
            opt_lrho  = [Descent(0.05) for ii in 1:n_latent]
            opt_lkap  = [Descent(0.05) for ii in 1:n_latent]
            opt_lsig  = [Descent(0.05) for ii in 1:n_latent]
            optm      = [Descent(0.05) for ii in 1:n_latent]
            optS      = [Descent(0.05) for ii in 1:n_latent]
            optx      = [Descent(0.05) for ii in 1:n_latent]
        end
        inds   = rand(1:inf_obj.data.n, batch_size)
        grads  = gradient(gps -> inference_elbo(inf_obj.data.x[inds,:], inf_obj.data.y[inds], inf_obj.data.n, gps), inf_obj)[1]
        
        for ii in 1:n_latent
            Flux.Optimise.update!(opt_cmean[ii], inf_obj.params[ii].const_mean,    -grads.params[ii][].const_mean)
            Flux.Optimise.update!(opt_lrho[ii],  inf_obj.params[ii].log_rho,       -grads.params[ii][].log_rho)
            Flux.Optimise.update!(opt_lkap[ii],  inf_obj.params[ii].log_kappa,     -grads.params[ii][].log_kappa)
            Flux.Optimise.update!(opt_lsig[ii],  inf_obj.params[ii].log_sigma,     -grads.params[ii][].log_sigma)
            Flux.Optimise.update!(optm[ii],      inf_obj.params[ii].inducing_mean, -grads.params[ii][].inducing_mean)
            Flux.Optimise.update!(optS[ii],      inf_obj.params[ii].inducing_L,    -LowerTriangular(grads.params[ii][].inducing_L) )
            Flux.Optimise.update!(optx[ii],      inf_obj.params[ii].inducing_locs, -grads.params[ii][].inducing_locs)
        end

        trace_elbo[t]        = inference_elbo(inf_obj.data.x[inds,:], inf_obj.data.y[inds], inf_obj.data.n, inf_obj)
    end

    return trace_elbo
end


"""
    fit_svgp!(svgp::SVGP_obj; n_iters=10000, batch_size=100, handoff = 1e20, return_param_traces = false)
Fit SVGP to data and update SVGP_obj in place. 
    
    n_iters:                 number of optimization steps
    batch_size:              size of batches for stochastic optimizer
    handoff:                 does not work!! Keep larger than n_iters for now
    return_parameter_traces: flag to indicate whether to return traces for all parameters or not
"""
function fit_svgp!(svgp::SVGP_obj; n_iters=10000, batch_size=100, handoff = 1e20, return_param_traces = false)

    grad_clip = 1.e-2
    
    # Initialize optimizers
    opt_cmean = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.02)  )
    opt_lrho  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    opt_lkap  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    opt_lsig  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    optm      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )
    optS      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) )
    optx      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )

    # Will optimize Cholesky to ensure positive definite
    trace_elbo = zeros(n_iters)

    if return_param_traces
        trace_xi     = zeros( n_iters, size(svgp.params.inducing_locs)[1], size(svgp.params.inducing_locs)[2] )
        trace_mn     = zeros( n_iters, size(svgp.params.inducing_mean)[1] )
        trace_cov    = zeros( n_iters, size(svgp.params.inducing_L)[1], size(svgp.params.inducing_L)[2] )
        trace_lrho   = zeros( n_iters, size(svgp.params.log_rho)[2] )
        trace_lkappa = zeros( n_iters )
        trace_lsigma = zeros( n_iters )
        trace_cmean  = zeros( n_iters )
    end

    for t = 1:n_iters
        if t == handoff
            opt_cmean = Descent(0.05)
            opt_lrho  = Descent(0.05)
            opt_lkap  = Descent(0.05)
            opt_lsig  = Descent(0.05)
            optm      = Descent(0.05)
            optS      = Descent(0.05)
            optx      = Descent(0.05)
        end
        inds   = rand(1:svgp.data.n, batch_size)
        grads  = gradient(gp -> svgp_elbo(svgp.data.x[inds,:], svgp.data.y[inds], gp, svgp), svgp.params)[1][]

        Flux.Optimise.update!(opt_cmean, svgp.params.const_mean,    -grads.const_mean)
        Flux.Optimise.update!(opt_lrho,  svgp.params.log_rho,       -grads.log_rho)
        Flux.Optimise.update!(opt_lkap,  svgp.params.log_kappa,     -grads.log_kappa)
        Flux.Optimise.update!(opt_lsig,  svgp.params.log_sigma,     -grads.log_sigma)
        Flux.Optimise.update!(optm,      svgp.params.inducing_mean, -grads.inducing_mean)
        Flux.Optimise.update!(optS,      svgp.params.inducing_L,    -LowerTriangular(grads.inducing_L) )
        Flux.Optimise.update!(optx,      svgp.params.inducing_locs, -grads.inducing_locs)

        trace_elbo[t]        = svgp_elbo(svgp.data.x[inds,:], svgp.data.y[inds], svgp.params, svgp)
        if return_param_traces
            trace_cmean[t]     = svgp.params.const_mean[1]
            trace_xi[t, :, :]  = svgp.params.inducing_locs
            trace_mn[t, :]     = svgp.params.inducing_mean
            trace_cov[t, :, :] = Hermitian(svgp.params.inducing_L * transpose(svgp.params.inducing_L))
        end
    end

    if return_param_traces
        ret_trace = Dict([
                ("const_mean", trace_cmean),
                ("inducing_mean",     trace_mn),
                ("inducing_cov",     trace_cov),
                ("inducing_locs",     trace_xi),
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
    n_parts   = hsvgp.n_parts
    grad_clip = 1.e-2

    # Initialize optimizers
    opt_cmean = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    opt_lrho  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    opt_lkap  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    opt_lsig  = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))
    optm      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )
    optS      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) )
    optx      = Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )

    # Initialize local optimizers
    lopt_cmean = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))  for ii in 1:n_parts]
    lopt_lrho  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))  for ii in 1:n_parts]
    lopt_lkap  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))  for ii in 1:n_parts]
    lopt_lsig  = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.0075))  for ii in 1:n_parts]
    loptm      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )  for ii in 1:n_parts]
    loptS      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.002) )  for ii in 1:n_parts]
    loptx      = [Flux.Optimiser(Flux.ClipValue(grad_clip), ADAM(1. * 0.01)  )  for ii in 1:n_parts]

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

                Flux.Optimise.update!(lopt_cmean[pp], hsvgp.local_svgps[pp].params.const_mean,    -grads.const_mean)
                Flux.Optimise.update!(lopt_lrho[pp],  hsvgp.local_svgps[pp].params.log_rho,       -grads.log_rho)
                Flux.Optimise.update!(lopt_lkap[pp],  hsvgp.local_svgps[pp].params.log_kappa,     -grads.log_kappa)
                Flux.Optimise.update!(lopt_lsig[pp],  hsvgp.local_svgps[pp].params.log_sigma,     -grads.log_sigma)
                Flux.Optimise.update!(loptm[pp],      hsvgp.local_svgps[pp].params.inducing_mean, -grads.inducing_mean)
                Flux.Optimise.update!(loptS[pp],      hsvgp.local_svgps[pp].params.inducing_L,    -LowerTriangular(grads.inducing_L) )
                Flux.Optimise.update!(loptx[pp],      hsvgp.local_svgps[pp].params.inducing_locs, -grads.inducing_locs)
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

                Flux.Optimise.update!(opt_cmean, hsvgp.global_obj.params.const_mean,    -grads.const_mean)
                Flux.Optimise.update!(opt_lrho,  hsvgp.global_obj.params.log_rho,       -grads.log_rho)
                Flux.Optimise.update!(opt_lkap,  hsvgp.global_obj.params.log_kappa,     -grads.log_kappa)
                Flux.Optimise.update!(opt_lsig,  hsvgp.global_obj.params.log_sigma,     -grads.log_sigma)
                Flux.Optimise.update!(optm,      hsvgp.global_obj.params.inducing_mean, -grads.inducing_mean)
                Flux.Optimise.update!(optS,      hsvgp.global_obj.params.inducing_L,    -LowerTriangular(grads.inducing_L) )
                Flux.Optimise.update!(optx,      hsvgp.global_obj.params.inducing_locs, -grads.inducing_locs)
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
