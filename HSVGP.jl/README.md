HSVGP.jl
=========
[![Build Status][ci-img]](https://github.com/lanl/PRISM/actions)
[![Codecov][codecov-img]](https://codecov.io/gh/lanl/PRISM)

HSVGP.jl is a sparse variational Gaussian process and hierarchical sparse Gaussian process implementation in Julia. 
These models have both been implemented for Gaussian and Poisson likelihood data with further likelihoods planned down
the road. 

## WORK IN PROGRESS

Installation
------------

```julia
] add https://github.com/lanl/PRISM:HSVGP.jl
```
Example
------------

Three example notebooks in the HSVGP/examples. One for the Gaussian likelihood, one for the Poisson likelihood, and one for implementing a custom likelihood.

Example code for fitting a simple SVGP below:

```julia
using HSVGP  
using Random, PyPlot, LinearAlgebra, Statistics  

rng = MersenneTwister(123);

N   = 50000
X   = 2 .* rand(rng, N)
X   = reshape(X,(N,1)); # Ensure X has two dimensions as required
Y   = 3*exp.(-0.5 .* X[:,1]) .* sin.(6*X[:,1]) + 0.1 .* randn(rng, N);# .+ 10.;

test_model = HSVGP.SVGP_obj(X,Y,5)    

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

Example of fitting a SVGP for the mean of Poisson count data using the Inference object syntax:

```julia
using HSVGP  
using Random, PyPlot, LinearAlgebra, Statistics  

rng = MersenneTwister(123);
N   = 50000
X   = 2 .* rand(rng, N)
X   = reshape(X,(N,1)); # Ensure X has two dimensions as required
lam = 6. * exp.(-2.0 .* X[:,1]) .* sin.(6*X[:,1]) .+ 10.;
Y   = [rand(Poisson(l),1)[1] for l in lam];

test_inf   = HSVGP.Inference_obj(X, float(Y), 6, "poisson");

opt_trace_inf = HSVGP.fit_inference!(test_inf, n_iters=20000, batch_size=10);

pX            = reshape([-0.:0.05:2.;],(41,1)); 
predY, predSD = HSVGP.pred_vgp(pX, test_inf.params[1]);

PyPlot.scatter(X, Y, alpha=0.2)
PyPlot.scatter(test_inf.params[1].inducing_locs, exp.(test_inf.params[1].inducing_mean), alpha=0.8)
PyPlot.plot(pX, exp.(predY), alpha=0.8)
PyPlot.fill_between(pX[:,1], exp.(predY + 2. .* sqrt.(predSD.^2)), exp.(predY - 2. .* sqrt.(predSD.^2)),alpha=.5)
PyPlot.xlabel("X")
PyPlot.ylabel("Y")
PyPlot.title("Original data Y ")
```

Example of fitting a SVGP for the mean of Exponentially distributed data using the Inference object syntax:

```julia
using HSVGP  
using Random, PyPlot, LinearAlgebra, Statistics  

N   = 50000
X   = 2 .* rand(N)
X   = reshape(X,(N,1)); # Ensure X has two dimensions as required

lam = 6. * exp.(-2.0 .* X[:,1]) .* sin.(6*X[:,1]) .+ 10.;
Y   = [rand(Exponential(l),1)[1] for l in lam];

# Exponential Log-Likelihood
function exp_ll(x,y,param_arr)
    return -param_arr[1] -  y ./ exp.(param_arr[1]) 
end

cust_ll = HSVGP.create_custom_likelihood(exp_ll)

test_inf   = HSVGP.Inference_obj(X,Y,10,"exponential", cust_ll, 1);

# Log scale inducing mean initialization. Current initialization assumes
#     GP is being fit to untransformed data
test_inf.params[1].inducing_mean = log.(test_inf.params[1].inducing_mean)

opt_trace_inf64 = HSVGP.fit_inference!(test_inf, n_iters=10000, batch_size=20);

pX = reshape([-0.:0.05:2.;],(41,1));
predY_inf, predSD_inf = HSVGP.pred_vgp(pX, test_inf.params[1]);
err_sigma_inf = exp(test_inf.params[1].log_sigma[1]);

PyPlot.scatter(X, Y, alpha=0.2)
PyPlot.scatter(test_inf.params[1].inducing_locs, test_inf.params[1].inducing_mean, alpha=0.8)
PyPlot.plot(pX, predY_inf, alpha=0.8)
PyPlot.fill_between(pX[:,1], predY_inf + 2. .* sqrt.(predSD_inf.^2 .+ err_sigma_inf^2), predY_inf - 2. .* sqrt.(predSD_inf.^2 .+ err_sigma_inf^2),alpha=.5)
PyPlot.xlabel("X")
PyPlot.ylabel("Y")
PyPlot.title("Original data Y ")
```

[ci-img]: https://github.com/lanl/PRISM/workflows/HSVGP-CI/badge.svg
[codecov-img]: https://img.shields.io/codecov/c/github/lanl/PRISM/master.svg?label=codecov
