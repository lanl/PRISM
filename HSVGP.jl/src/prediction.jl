"""
    pred_vgp(x_pred, gp_obj)
Obtain predictions from variational Gaussian process at locations x_pred. 
Returns tuple with predicted means and predicted standard deviations
"""
function pred_vgp(x_pred, gp_obj::SVGP_obj)
    opt_xi    = gp_obj.params.inducing_locs
    opt_m     = gp_obj.params.inducing_mean
    opt_cov   = Hermitian(gp_obj.params.inducing_L * transpose(gp_obj.params.inducing_L))
    mean_fac  = gp_obj.data.mean_y

    ni, nd    = size(opt_xi)

    rho        = exp.(gp_obj.params.log_rho)
    kappa      = exp(gp_obj.params.log_kappa[1])
    sigsq      = exp(gp_obj.params.log_sigma[1])^2

    nugg_scale = 1.e-6 * ( (2.0 * gp_obj.data.mean_y)^2 + gp_obj.data.var_y)
    if nugg_scale < 1.e-8
        nugg_scale = 1.e-8
    end

    cov_pred   = covfn(x_pred, x_pred, gp_obj) + I * nugg_scale
    cov_mat_i  = covfn(gp_obj.params.inducing_locs, gp_obj.params.inducing_locs, gp_obj) + I * nugg_scale
    cross_mat  = covfn(x_pred, gp_obj.params.inducing_locs, gp_obj)
    inv_mat_i  = inv(cov_mat_i)
    cross_inv  = cross_mat * inv_mat_i

    p_mean = cross_mat * inv_mat_i * opt_m
    p_cov  = diag( cov_pred + cross_inv * ( opt_cov - cov_mat_i) * cross_inv');

    return p_mean, sqrt.(p_cov)
end
