"""
    gaussian_likelihood(x, y, gp_params)
Evaluate Gaussian likelihood for SVGP

Returns scalar real value of the marginal likelihood
"""
function gaussian_likelihood(x, y, gp_params::Vector{SVGP_params})
    n, p  = size(x)
    sigsq = exp(gp_params[1].log_sigma[1])^2

    p_mean, p_sd = pred_vgp(x, gp_params[1])

    residuals  = y - p_mean
    marg_like  = -0.5 * n * log(2.0*π)
    marg_like -=  n * gp_params[1].log_sigma[1]
    marg_like -=  0.5 * sum( residuals .^ 2 .+ p_sd .^ 2 ) / sigsq
    
    # This is an exponential prior on sigsq to try to avoid the "all noise"
    #     fit for the GP.
    # TODO: Determine if this can/should be dropped or not
    marg_like += -3.0 * gp_params[1].log_sigma[1]
    
    return marg_like
end

"""
    poisson_likelihood(x, y, gp_params)
Monte Carlo approximation to marginal likelihood, sampling from variational distribution

Returns scalar real value of the marginal likelihood
"""
function poisson_likelihood(x, y, gp_params::Vector{SVGP_params})
    n, p         = size(x)
    p_mean, p_sd = pred_vgp(x, gp_params[1])
    
    marg_like = 0
    for ii in 1:16  # TODO: Hard coded num of sample for marginalization for now. Should make an argument.
        stdn_samps    = randn(n)
        loglam_samps  = p_mean + p_sd .* stdn_samps
        marg_like    += sum( y .* loglam_samps - exp.(loglam_samps) - loggamma.(y .+ 1) )
    end
    
    marg_like /= 16.
    
    return marg_like
end

"""
    gev_likelihood(x, y, gp_params)
Monte Carlo approximation to marginal likelihood, sampling from variational distribution

WARNING: Likely has some serious support issues right now. Need checking on bounds when
         param_samps[3] is not 0. Lower bound for > 0, Upper bound for < 0

Returns scalar real value of the marginal likelihood
"""
function gev_likelihood(x, y, gp_params::Vector{SVGP_params})
    n, p     = size(x)
    p_params = [pred_vgp(x, svgp) for svgp in gp_params]
    
    marg_like = 0
    for ii in 1:16  # TODO: Hard coded num of sample for marginalization for now. Should make an argument.
        param_samps   = [p_par[1] + p_par[2] .* randn(n) for p_par in p_params]
        
        t_vector      = (1. .+ param_samps[3] .* ( y - param_samps[1]) ./ exp.(param_samps[2]) ) .^ ( -1.0 ./ param_samps[3])
        new_like_term = sum( -param_samps[2] + (param_samps[3] .+ 1.) .* log.(t_vector) - t_vector )
        
        marg_like    += new_like_term
    end

    marg_like /= 16.
    
    return marg_like
end

"""
    gumbel_likelihood(x, y, gp_params)
Monte Carlo approximation to marginal likelihood, sampling from variational distribution

GEV with shape parameter = 0. Does not have the support challenges that the GEV does

Returns scalar real value of the marginal likelihood
"""
function gumbel_likelihood(x, y, gp_params::Vector{SVGP_params})
    n, p     = size(x)
    p_params = [pred_vgp(x, svgp) for svgp in gp_params]
    
    marg_like = 0
    for ii in 1:16  # TODO: Hard coded num of sample for marginalization for now. Should make an argument.
        param_samps   = [p_par[1] + p_par[2] .* randn(n) for p_par in p_params]
        
        z_vector      = ( y - param_samps[1]) ./ exp.(param_samps[2]) 
        new_like_term = sum( -param_samps[2] - z_vector - exp.(-z_vector) )
        
        marg_like    += new_like_term
    end

    marg_like /= 16.

    return marg_like
end

"""
    create_custom_likelihood(ll_func)
Monte Carlo approximation to a custom marginal likelihood, sampling from variational distribution.
ll_func should be a function whose arguments are (x, y, p)
where p an array of arrays. 
The outer array is the size of the number of parameters modeled with SVGPs.
The size of the inner array is the number of observations 


WARNING: Likely has some serious support issues right now. Need checking on bounds when
         param_samps[3] is not 0. Lower bound for > 0, Upper bound for < 0

Returns scalar real value of the marginal likelihood
"""
function create_custom_likelihood(ll_func)
    function cust_like(x, y, gp_params::Vector{SVGP_params})
        n, p     = size(x)
        p_params = [pred_vgp(x, svgp) for svgp in gp_params]
    
        marg_like = 0
        for ii in 1:16  # TODO: Hard coded num of sample for marginalization for now. Should make an argument.
            param_samps   = [p_par[1] + p_par[2] .* randn(n) for p_par in p_params]
        
            new_like_term = sum( ll_func(x,y,param_samps) )

            marg_like     = (ii-1.0)/ii * marg_like + 1. / ii * new_like_term
        end
        
        marg_like /= 16.
        
        return marg_like
    end

    return cust_like
end
