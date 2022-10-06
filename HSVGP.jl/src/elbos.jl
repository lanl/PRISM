"""
    inference_elbo(x, y, N, likelihood, inference_obj)
Evaluate the elbo for optimization of the inference model with a stochastic variational GP 

Returns scalar real value of the elbo
"""
function inference_elbo(x, y, N, inf_obj::Inference_obj)
    n, p = size(x)

    marg_like = inf_obj.ll_function(x, y, inf_obj.params)

    kl_term = 0
    for gp in inf_obj.params
        nugg_scale = 1.e-6 * exp(gp.log_kappa[1])
        if nugg_scale < 1.e-8
            nugg_scale = 1.e-8
        end
        
        cov_mat_i  = covfn(gp.inducing_locs, gp.inducing_locs, gp)
        cov_mat_i += I * nugg_scale
        
        S          = Hermitian(gp.inducing_L * transpose(gp.inducing_L) ) + nugg_scale * I
        inv_mat_i  = inv(cov_mat_i)
        m          = gp.inducing_mean
        cm         = gp.const_mean[1]
        n_inducing = length(gp.inducing_mean)
        
        kl_term += 0.5 * (tr(inv_mat_i * S) + ((cm .- m)'*inv_mat_i*(cm .- m))[1] - n_inducing + logdet(cov_mat_i) - logdet(S))
        # q || p
    end
    
    # Here as a placeholder to avoid breaking the gradient calculation for now
    #     Should replace later with a separate log-likelihood parameters term
    #     as part of the inference object so that parameters that do not have
    #     spatial GP priors can be point estimated. 
    #     Right now log_sigma really only plays a role in the Gaussian likelihood
    #     but is stuck in the GP objects due for legacy reasons
    #     Remove once likelihood-specific parameters have been created
    unnecessary_lsigma_term_to_remove_later = sum([0. * gp.log_sigma[1] for gp in inf_obj.params])

    return float(N) / float(n) * marg_like - kl_term + unnecessary_lsigma_term_to_remove_later
end

function inference_elbo(x, y, N, inf_obj::Inference_obj, prior_obj::Inference_obj)
    n, p = size(x)

    marg_like = inf_obj.ll_function(x, y, inf_obj.params)

    kl_term = 0
    for ii in 1:size(inf_obj.params)[1]
        gp       = inf_obj.params[ii]
        prior_gp = prior_obj.params[ii]
        nugg_scale = 1.e-6 * exp(gp.log_kappa[1])
        if nugg_scale < 1.e-8
            nugg_scale = 1.e-8
        end
        
        pm, cov_mat_i  = pred_vgp(gp.inducing_locs, prior_gp, full_cov=true)
        cov_mat_i     += I * nugg_scale
        
        S          = Hermitian(gp.inducing_L * transpose(gp.inducing_L) ) + nugg_scale * I
        inv_mat_i  = inv(cov_mat_i)
        m          = gp.inducing_mean .+ gp.const_m
        n_inducing = length(gp.inducing_mean)
        
        kl_term += 0.5 * (tr(inv_mat_i * S) + ((pm .- m)'*inv_mat_i*(pm .- m))[1] - n_inducing + logdet(cov_mat_i) - logdet(S))
        # q || p
    end
    
    # Here as a placeholder to avoid breaking the gradient calculation for now
    #     Should replace later with a separate log-likelihood parameters term
    #     as part of the inference object so that parameters that do not have
    #     spatial GP priors can be point estimated. 
    #     Right now log_sigma really only plays a role in the Gaussian likelihood
    #     but is stuck in the GP objects due for legacy reasons
    #     Remove once likelihood-specific parameters have been created
    unnecessary_lsigma_term_to_remove_later = sum([0. * gp.log_sigma[1] for gp in inf_obj.params])

    return float(N) / float(n) * marg_like - kl_term + unnecessary_lsigma_term_to_remove_later
end

"""
    inference_elbo_local(x, y, N, inference_obj, global_pred)
Evaluate the stochastic variational Gaussian process elbo as part of the hierarchical SVGP 
utilizing the predicted mean and standard deviation from the global SVGP

Warning: Untested, may not work yet. In progress.

Returns scalar real value of the elbo
"""
function inference_elbo_local(x, y, N, inf_obj::Inference_obj, global_pred::Vector{SVGP_params})
    elbo = svgp_elbo(x, y, N, inf_obj)
    
    term5 = 0.0

    for ii in 1:length(global_pred)
        prior_mean, prior_sd = global_pred[ii]
        resid = inf_obj.params[ii].inducing_mean - prior_mean

        term5 -=  0.5 * size(prior_sd)[1] * log(2.0*π)
        term5 -=  sum( log.(prior_sd) )
        term5 -=  0.5 * sum( resid' * (resid ./ prior_sd .^ 2) )
    end

    return elbo + term5
end

"""
    inference_elbo_global(x, y, N, inference_obj)
Evaluate the stochastic variational Gaussian process elbo as part of the hierarchical SVGP 
Is actually just a call to svgp_elbo, but exists here for clarity

Warning: Untested, may not work yet. In progress

Returns scalar real value of the elbo
"""
function inference_elbo_global(x, y, N, inf_obj::Inference_obj)
    return inference_elbo(x, y, N, inf_obj)
end


################################
# Old functions for compatibility

"""
    svgp_elbo(x, y, gp_params, gp_obj)
Evaluate the stochastic variational Gaussian process elbo for optimization

Returns scalar real value of the elbo
"""
function svgp_elbo(x, y, gp_params::SVGP_params, gp_obj::SVGP_obj)
    n, p = size(x)

    sigsq = exp(gp_params.log_sigma[1])^2

    p_mean, p_sd = pred_vgp(x, gp_params)

    residuals  = y - p_mean

    if gp_obj.distrib == "gaussian"
        marg_like  = -0.5 * n * log(2.0*π)
        marg_like -=  n * gp_params.log_sigma[1]
        marg_like -=  0.5 * sum( residuals .^ 2 .+ p_sd .^ 2 ) / sigsq
    end

    if gp_obj.distrib == "poisson"
        marg_like = 0
        for ii in 1:16  # TODO: Hard coded num of sample for marginalization for now. Should make an argument.
            stdn_samps    = randn(n)
            loglam_samps  = p_mean + p_sd .* stdn_samps
            marg_like     = (ii-1.0)/ii * marg_like + 1. / ii * sum( y .* loglam_samps - exp.(loglam_samps) - loggamma.(y .+ 1) )
        end
    end

    # This is an exponential prior on sigsq to try to avoid the "all noise"
    #     fit for the GP.
    sigma_prior = -0.0 * gp_params.log_sigma[1]

    nugg_scale = 1.e-6 * exp(gp_params.log_kappa[1])
    if nugg_scale < 1.e-8
        nugg_scale = 1.e-8
    end


    cov_mat_i  = covfn(gp_params.inducing_locs, gp_params.inducing_locs, gp_params)
    cov_mat_i += I * nugg_scale
    S          = Hermitian(gp_params.inducing_L * transpose(gp_params.inducing_L) ) + nugg_scale * I
    inv_mat_i  = inv(cov_mat_i)
    m          = gp_params.inducing_mean
    cm         = gp_params.const_mean[1]

    kl_term = 0.5 * (tr(inv_mat_i * S) + ((cm .- m)'*inv_mat_i*(cm .- m))[1] - gp_obj.n_inducing + logdet(cov_mat_i) - logdet(S))
    # q || p

    return float(gp_obj.data.n) / float(n) * marg_like + sigma_prior - kl_term
end

function svgp_elbo(x, y, gp_params::SVGP_params, gp_obj::SVGP_obj, prior_gp::SVGP_params)
    n, p = size(x)

    sigsq = exp(gp_params.log_sigma[1])^2

    p_mean, p_sd   = pred_vgp(x, gp_params)
    pr_mean, pr_sd = pred_vgp(gp_params.inducing_locs, prior_gp, full_cov=true)

    residuals  = y - p_mean

    if gp_obj.distrib == "gaussian"
        marg_like  = -0.5 * n * log(2.0*π)
        marg_like -=  n * gp_params.log_sigma[1]
        marg_like -=  0.5 * sum( residuals .^ 2 .+ p_sd .^ 2 ) / sigsq
    end
    
    if gp_obj.distrib == "poisson"
        marg_like = 0
        for ii in 1:16  # TODO: Hard coded num of sample for marginalization for now. Should make an argument.
            stdn_samps    = randn(n)
            loglam_samps  = p_mean + p_sd .* stdn_samps
            marg_like     = (ii-1.0)/ii * marg_like + 1. / ii * sum( y .* loglam_samps - exp.(loglam_samps) - loggamma.(y .+ 1) )
        end
    end

    # This is an exponential prior on sigsq to try to avoid the "all noise"
    #     fit for the GP.
    sigma_prior = -0.0 * gp_params.log_sigma[1]

    nugg_scale = 1.e-6 * exp(gp_params.log_kappa[1])
    if nugg_scale < 1.e-8
        nugg_scale = 1.e-8
    end


    cov_mat_i  = pr_sd
    cov_mat_i += I * nugg_scale
    S          = Hermitian(gp_params.inducing_L * transpose(gp_params.inducing_L) ) + nugg_scale * I
    inv_mat_i  = inv(cov_mat_i)
    m          = gp_params.inducing_mean .+ gp_params.const_mean[1]

    kl_term = 0.5 * (tr(inv_mat_i * S) + ((pr_mean .- m)'*inv_mat_i*(pr_mean .- m))[1] - gp_obj.n_inducing + logdet(cov_mat_i) - logdet(S))
    # q || p

    return float(gp_obj.data.n) / float(n) * marg_like + sigma_prior - kl_term
end

"""
    svgp_elbo_local(x, y, gp_params, gp_local, global_pred)
Evaluate the stochastic variational Gaussian process elbo as part of the hierarchical SVGP 
utilizing the predicted mean and standard deviation from the global SVGP

Returns scalar real value of the elbo
"""
function svgp_elbo_local(x, y, gp_params::SVGP_params, gp_local::SVGP_obj, global_pred)
    prior_mean, prior_sd = global_pred

    elbo = svgp_elbo(x, y, gp_params, gp_local)

    resid = gp_params.inducing_mean - prior_mean

    term5  = -0.5 * size(prior_sd)[1] * log(2.0*π)
    term5 -=  sum( log.(prior_sd) )
    term5 -=  0.5 * sum( resid' * (resid ./ prior_sd .^ 2) )

    return elbo + term5
end

"""
    svgp_elbo_global(x, y, gp_params, gp_global)
Evaluate the stochastic variational Gaussian process elbo as part of the hierarchical SVGP 
Is actually just a call to svgp_elbo, but exists here for clarity

Returns scalar real value of the elbo
"""
function svgp_elbo_global(x, y, gp_params::SVGP_params, gp_global::SVGP_obj)
    return svgp_elbo(x, y, gp_params, gp_global)
end
