# Custom structure for holding data for sparse, variational Gaussian process
struct SVGP_data
    x::Array{Float64,2}   # x values for observed data
    y::Array{Float64,1}   # y values for observed data
    n::Int64              # Number of observations
    p::Int64              # Number of input dimensions
    mean_y::Float64       # mean of data for scaling constant mean prior
    var_y::Float64        # variance of data for setting nugget
end

# Constructor for building by feeding X and Y data. 
#    This should be the most common constructor
SVGP_data(x::Array{Float64,2}, y::Array{Float64,1}) = SVGP_data(x, y, size(x)[1], size(x)[2], mean(y), var(y))

# Constructor for dummy data object
#    Required for building SVGP object with full data not accessible
SVGP_data(dummy::String) = SVGP_data(zeros(1,2), [0], 1, 2, 0., 1.)

# Custom structure for holding parameters for sparse, variational Gaussian process
mutable struct SVGP_params
    log_rho::Array{Float64,2}                              # log correlation length for GP
    log_kappa::Array{Float64,1}                            # log GP variance
    log_sigma::Array{Float64,1}                            # log GP error standard deviation
    inducing_L::LowerTriangular{Float64,Array{Float64,2}}  # Cholesky of variational covariance
    inducing_mean::Array{Float64,1}                        # Inducing point mean parameters
    inducing_locs::Array{Float64,2}                        # Inducing point location parameters
    function SVGP_params(inp_data::SVGP_data, n_inducing::Int64)
        ni   = n_inducing
        np   = size(inp_data.x)[2]
        # Initialize the correlation lengths to 1/4 of the domain in each coordinate dimension
        #     This requires the data to exist, hence the dummy data structure for now
        # TODO: More elegent way to do this. Maybe just an argument but somewhat want something better
        lrho = reshape([log( (maximum(inp_data.x[:,ii]) - minimum(inp_data.x[:,ii]))/4. ) for ii in 1:np], (1,np))
        # Initialize the variance based on the variance of the data
        #     This requires the data to exist, hence the dummy data structure for now
        # TODO: More elegent way to do this. 
        lkap = [log(var(inp_data.y))]
        lssq = [-4.0]
        icov = 0.1 * I + zeros(ni, ni);

        # Initialize inducing point locations to be a Latin hypercube across the domain
        x_range = [(minimum(inp_data.x[:,ii]), maximum(inp_data.x[:,ii])) for ii in 1:np]
        xi      = scaleLHC(LHCoptim(ni, np, 1000)[1], x_range);

        # Initialize the inducing point means using the nearest neighbor observations
        kdtree      = KDTree(transpose(inp_data.x))
        idxs, dists = knn(kdtree, transpose(xi), 1);
        init_mean   = [inp_data.y[ii[1]] for ii in idxs];

        return new(lrho, lkap, lssq, cholesky(icov).L, init_mean, xi)
    end
end

# Custom structure for holding sparse, variational Gaussian process object
struct SVGP_obj
    n_inducing::Int64      # Number of inducing points
    data::SVGP_data
    params::SVGP_params
    distrib::String        # Label for which likelihood to use. "gaussian" or "poisson" for now
    function SVGP_obj(n_inducing::Int64, data::SVGP_data, params::SVGP_params, distrib::String)
        return new(n_inducing, data, params, distrib)
    end
end

# Constructor from data defaulting to Gaussian likelihood
SVGP_obj(x::Array{Float64,2}, y::Array{Float64,1}, ni::Int64) = SVGP_obj(ni, SVGP_data(x,y), SVGP_params(SVGP_data(x,y), ni), "gaussian")

# Constructor from data for general likelihoods
#     Should often be the 
SVGP_obj(x::Array{Float64,2}, y::Array{Float64,1}, ni::Int64, distrib::String) = SVGP_obj(ni, SVGP_data(x,y), SVGP_params(SVGP_data(x,y), ni), distrib)

# Hierarchical SVGP structure. Holds global and local SVGPs
#     Not for distributed calculations as all data is held within the object
struct HSVGP_obj
    ni_global::Int64
    global_obj::SVGP_obj
    local_svgps::Array{SVGP_obj, 1}
    n_parts::Int64
    function HSVGP_obj(x::Array{Float64,2}, y::Array{Float64,1}, ni_local::Int64, ni_global::Int64, part_labels::Array{Int64,1}, distrib::String)
        n_parts     = maximum(part_labels)
        local_svgps = [SVGP_obj(x[part_labels .== ii, :], y[part_labels .== ii], ni_local, distrib) for ii in 1:n_parts]
        data        = SVGP_data(x,y)
        obj         = SVGP_obj(ni_global, SVGP_data("global"), SVGP_params(data, ni_global), "gaussian")
        return new(ni_global, obj, local_svgps, n_parts)
    end
end

# Constructor from data defaulting to Gaussian likelihood
HSVGP_obj(
    x::Array{Float64,2},
    y::Array{Float64,1},
    ni_local::Int64,
    ni_global::Int64,
    part_labels::Array{Int64,1}) = HSVGP_obj(x, y, ni_local, ni_global, part_labels, "gaussian")


# UNUSED SO FAR
# Possible struct for carrying optimization info to container inside the full structs
# mutable struct SVGP_opt
#     trace_elbo::Array{Float64,1} # Optimization trace for fit
#     function SVGP_opt(x)
#         return new(x)
#     end
# end
