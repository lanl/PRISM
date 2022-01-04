"""
    pred_vgp(x_pred, gp_params)
Obtain predictions from variational Gaussian process at locations x_pred. 
Returns tuple with predicted means and predicted standard deviations
"""
function pred_vgp(x_pred, gp_params::SVGP_params)
    opt_xi    = gp_params.inducing_locs
    opt_m     = gp_params.inducing_mean
    opt_cov   = diagm(gp_params.inducing_C)

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
    p_cov  = diag( cov_pred + cross_inv' * ( opt_cov - cov_mat_i) * cross_inv);

    return p_mean, sqrt.(p_cov)
end

"""
    pred_vgp(x_pred, gp_obj)
Obtain predictions from variational Gaussian process at locations x_pred. 
Returns tuple with predicted means and predicted standard deviations.
Included for backward compatibility, but only part of gp_obj used is the params
"""

function pred_vgp(x_pred, gp_obj::SVGP_obj)
    gp_params = gp_obj.params
    opt_xi    = gp_params.inducing_locs
    opt_m     = gp_params.inducing_mean
    opt_cov   = gp_params.inducing_C)

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
    p_cov  = diag( cov_pred + cross_inv' * ( opt_cov - cov_mat_i) * cross_inv);

    return p_mean, sqrt.(p_cov)
end
