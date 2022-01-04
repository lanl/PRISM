# PRISM
[![HSVGP Build Status][hsvgp-ci-img]](https://github.com/lanl/PRISM/actions)
[![Codecov][codecov-img]](https://codecov.io/gh/lanl/PRISM)

Programming Repository for In Situ Modeling

![PRISM](prism.png)

The Programming Repository for In Situ Modeling (PRISM) is a set of tools for fitting statistics and machine learning models to simulation data inside the simulations as they are running. By fitting models inside running simulations, PRISM can be used to analyze simulation data that is otherwise inaccessible because of I/O and storage bottlenecks associated with exascale and other future high performance computing architectures. The tools are designed to implement a wide variety of data analyses with an emphasis on spatiotemporal hierarchical Bayesian models. PRISM is efficient, scalable, and streaming with estimation based on variational inference, advanced Monte Carlo techniques, and fast optimization methods. The core modeling components aid this goal by imposing sparsity and approximate inference wherever possible. These components are written in Julia, a high-level programming language designed for high performance. PRISM also contains tools for interfacing with large-scale scientific simulations written in Fortran and C/C++. This layer of abstraction allows the data scientist to construct analysis models in Julia without concern for the implementation details of the simulation capability. With these components, PRISM can be used to unlock the full scientific potential of next-generation HPC simulations.

[hsvgp-ci-img]: https://github.com/lanl/PRISM/workflows/HSVGP-CI/badge.svg
[codecov-img]: https://img.shields.io/codecov/c/github/lanl/PRISM/master.svg?label=codecov

## Description of tools

### HSVGP.jl
Package for fitting stochastic, variational, sparse Gaussian process regression for in-situ statistical modeling. This contains implementations of the Hensman 2013 / Hensman 2015 sparse Gaussian process models for general likelihoods with mini-batch optimization, the structure for fitting in a distributed environment, and an implementation of a hierarchical, distributed Gaussian process model.

### TributaryPCA.jl
TributaryPCA.jl is a Julia implementation of the AdaOja streaming PCA algorithm in a distributed computing setting with MPI. This contains implementations of online principal components analysis with distributed linear algebra via two routines (partitioned Cholesky decomposition and direct tall-skinny QR decomposition).

### Julienne.jl
Julienne.jl is a Julia implementation of streaming linear regression with a modified F-test for online change-point detection. This contains an implementation of Myers et al (2016) “Partitioning a Large Simulation as It Runs” which tests, in a streaming fashion, whether an existing linear fit continues to describe new data or whether a new linear fit would provide a better description. 

### FastGPEstimation
FastGP estimation is a set of coding tools towards the identification of Gaussian Process hyperparameters using convolutional neural networks and the verification of these parameters using maximum likelihood estimation. 

#### Helper Functions included in this package

##### GPCNNRegressor_julia.ipynb
GPCNNRegressor_julia.ipynb is a Julia implementation of a convolutional neural network regression model to identify hyper parameters for a Gaussian distribution.

##### render_CAM_netCDF.ipynb
render_CAM_netCDF.ipynb is a Python script that takes as input the NetCDF results of an E3SM simulation and converts it into processor-based segments for processing.

##### grid_global_data.ipynb
Grid_global_data.ipynb is a Python script that takes the E3SM output and grids it to a 200 x 100 grid for processing. 

##### FitGP.ipynb
FitGP.ipynb identifies the three hyperparameters, sigma and two correlation lengths for the maximum likelihood estimation given 2D gridded data. Results are saved in param_ard_data2.npz.

##### generate_GP_for_ML.ipynb
generate_GP_for_ML.ipynb is a helper module to the hyperparameters for a GP.
