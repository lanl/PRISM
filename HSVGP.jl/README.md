HSVGP.jl
=========

HSVGP.jl is a sparse variational Gaussian process and hierarchical sparse Gaussian process implementation in Julia. 
These models have both been implemented for Gaussian and Poisson likelihood data with further likelihoods planned down
the road. 

## WORK IN PROGRESS

Installation
------------

Current requirement is for local use as shown in examples below. Installation instructions directly from github coming soon.

Example
------------

Two example notebooks in the HSVGP/examples. One for the Gaussian likelihood and one for the Poisson likelihood.

Example code below:

```julia
include("../src/HSVGP.jl")
using .HSVGP  
using Random, PyPlot, LinearAlgebra, Statistics  

rng = MersenneTwister(123);

N   = 50000
X   = 2 .* rand(rng, N)
X   = reshape(X,(N,1)); # Ensure X has two dimensions as required
Y   = 3*exp.(-0.5 .* X[:,1]) .* sin.(6*X[:,1]) + 0.1 .* randn(rng, N);# .+ 10.;

test_model = HSVGP.SVGP_obj(X,Y,5)    
inds       = rand(1:N, 12);  

opt_trace, p_traces = HSVGP.fit_svgp!(test_model, n_iters=20000, batch_size=20);  

pX            = reshape([-0.:0.05:2.;],(41,1)); 
predY, predSD = HSVGP.pred_vgp(pX, test_model);
err_sigma     = exp(test_model.params.log_sigma[1]);  

PyPlot.scatter(X, Y, alpha=0.2)
PyPlot.scatter(test_model.params.inducing_locs, test_model.params.inducing_mean, alpha=0.8)
PyPlot.plot(pX, predY, alpha=0.8)
PyPlot.fill_between(pX[:,1], predY + 2. .* sqrt.(predSD.^2 .+ err_sigma^2), predY - 2. .* sqrt.(predSD.^2 .+ err_sigma^2),alpha=.5)
PyPlot.xlabel("X")
PyPlot.ylabel("Y")
PyPlot.title("Original data Y ")
```

License
-------

HSVGP.jl is licensed under the INSERT LICENSE HERE.
