"""
    pred_vgp(x_pred, gp_params)
Obtain predictions from variational Gaussian process at locations x_pred. 
Returns tuple with predicted means and predicted standard deviations
"""
function pred_vgp(x_pred, gp_params::SVGP_params; full_cov=false, include_noise=false)
    opt_xi    = gp_params.inducing_locs
    opt_m     = gp_params.inducing_mean
    opt_cov   = Hermitian(gp_params.inducing_L * transpose(gp_params.inducing_L))

    ni, nd    = size(opt_xi)

    rho        = exp.(gp_params.log_rho)
    kappa      = exp(gp_params.log_kappa[1])
    sigsq      = exp(gp_params.log_sigma[1])^2

    nugg_scale = 1.e-6 * kappa
    if nugg_scale < 1.e-8
        nugg_scale = 1.e-8
    end

    cov_pred   = covfn(x_pred, x_pred, gp_params) + I * nugg_scale
    cov_mat_i  = covfn(gp_params.inducing_locs, gp_params.inducing_locs, gp_params) + I * nugg_scale
    cross_mat  = covfn(gp_params.inducing_locs, x_pred, gp_params)
    cross_inv  = cov_mat_i \ cross_mat
    
    p_mean = cross_inv' * (opt_m .- gp_params.const_mean[1]) .+ gp_params.const_mean[1] 
    p_cov  = cov_pred + cross_inv' * ( opt_cov - cov_mat_i) * cross_inv;

    if include_noise
        p_cov = p_cov + sigsq * I
    end
    
    if full_cov
        return p_mean, p_cov
    else
        return p_mean, sqrt.(diag(p_cov))
    end
end

"""
    pred_vgp(x_pred, gp_obj)
Obtain predictions from variational Gaussian process at locations x_pred. 
Returns tuple with predicted means and predicted standard deviations.
Included for backward compatibility, but only part of gp_obj used is the params
"""

function pred_vgp(x_pred, gp_obj::SVGP_obj; full_cov=false, include_noise=false)
    gp_params = gp_obj.params
    opt_xi    = gp_params.inducing_locs
    opt_m     = gp_params.inducing_mean
    opt_cov   = Hermitian(gp_params.inducing_L * transpose(gp_params.inducing_L))

    ni, nd    = size(opt_xi)

    rho        = exp.(gp_params.log_rho)
    kappa      = exp(gp_params.log_kappa[1])
    sigsq      = exp(gp_params.log_sigma[1])^2

    nugg_scale = 1.e-6 * kappa
    if nugg_scale < 1.e-8
        nugg_scale = 1.e-8
    end

    cov_pred   = covfn(x_pred, x_pred, gp_params) + I * nugg_scale
    cov_mat_i  = covfn(gp_params.inducing_locs, gp_params.inducing_locs, gp_params) + I * nugg_scale
    cross_mat  = covfn(gp_params.inducing_locs, x_pred, gp_params)
    cross_inv  = cov_mat_i \ cross_mat
    
    p_mean = cross_inv' * (opt_m .- gp_params.const_mean[1]) .+ gp_params.const_mean[1]
    p_cov  = cov_pred + cross_inv' * ( opt_cov - cov_mat_i) * cross_inv;

    if include_noise
        p_cov = p_cov + sigsq * I
    end

    if full_cov
        return p_mean, p_cov
    else
        return p_mean, sqrt.(diag(p_cov))
    end
end
