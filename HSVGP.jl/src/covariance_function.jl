"""
    covfn(x1, x2, gp)
Calculate GP covariance between x1 and x2

Returns GP covariance matrix
"""
function covfn(x1, x2, gp::SVGP_obj)
    rho = exp.(gp.params.log_rho)
    R   = exp.(-pairwise(SqEuclidean(), (x1 ./ rho)', (x2 ./ rho)', dims=2))

    return exp(gp.params.log_kappa[1]) * R .+ (2.0 * gp.data.mean_y)^2
end

# function covfn(x1, x2, params::SVGP_params)
#     rho = exp.(params.log_rho)
#     x2    = x2 ./ rho
#     R     = exp.(-pairwise(haversine(), x1', x2')) # WORK ON BEING ABLE TO USE THIS KERNEL FOR EARTH SURFACE
#     return exp(params.log_kappa[1]) * R .+ (2.0 * params.mean_factor)^2
# end
