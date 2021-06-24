"""
    covfn(x1, x2, gp_params)
Calculate GP covariance between x1 and x2 based on a squared exponential covariance function

Returns GP covariance matrix
"""
function covfn(x1, x2, gp_params::SVGP_params)
    rho = exp.(gp_params.log_rho)
    R   = exp.(-Distances.pairwise(SqEuclidean(), x1 ./ rho, x2 ./ rho, dims=1))

    return exp(gp_params.log_kappa[1]) * R
end

# function covfn(x1, x2, params::SVGP_params)
#     rho = exp.(params.log_rho)
#     x2    = x2 ./ rho
#     R     = exp.(-pairwise(haversine(), x1', x2')) # WORK ON BEING ABLE TO USE THIS KERNEL FOR EARTH SURFACE
#     return exp(params.log_kappa[1]) * R .+ (2.0 * params.mean_factor)^2
# end

"""
    covfn_lonlat(x1, x2, gp_params)
Calculate GP covariance between x1 and x2 based on the Haversine distance on a sphere

Warning: Currently does not work due to lack of auto-differentiation through Haversine distance call

Returns GP covariance matrix
"""
function covfn_lonlat(x1, x2, gp_params::SVGP_params)
    rho = exp.(gp_params.log_rho)
    R   = exp.(-Distances.pairwise(Haversine(), x1 ./ rho, x2 ./ rho, dims=1))

    return exp(gp_params.log_kappa[1]) * R
end
