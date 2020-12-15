"""
    svgp_elbo(x, y, gp_params, gp_obj)
Evaluate the stochastic variational Gaussian process elbo for optimization

Returns scalar real value of the elbo
"""
function svgp_elbo(x, y, gp_params::SVGP_params, gp_obj::SVGP_obj)
    n, p = size(x)

    sigsq = exp(gp_params.log_sigma[1])^2

    p_mean, p_sd = pred_vgp(x, gp_obj)

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
    sigma_prior = -3.0 * gp_params.log_sigma[1]

    nugg_scale = 1.e-6 * ( (2.0 * gp_obj.data.mean_y)^2 + gp_obj.data.var_y)
    if nugg_scale < 1.e-8
        nugg_scale = 1.e-8
    end


    cov_mat_i  = covfn(gp_params.inducing_locs, gp_params.inducing_locs, gp_obj)
    cov_mat_i += I * nugg_scale
    S          = Hermitian(gp_params.inducing_L * transpose(gp_params.inducing_L) )
    inv_mat_i  = inv(cov_mat_i)
    m          = gp_params.inducing_mean

    kl_term = 0.5 * (tr(inv_mat_i * S) + ((-m)'*inv_mat_i*(-m))[1] - gp_obj.n_inducing + logdet(cov_mat_i) - logdet(S))
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
    term5 -=  0.5 * sum( resid' * (resid ./ prior_sd) )

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
