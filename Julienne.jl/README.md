Julienne.jl
=========

Julienne.jl is a Julia implementation of streaming linear regression and a modified F-test for online change-point detection.

Paper: Myers et al (2016). Partitioning a Large Simulation as It Runs.

Example
------------

Here, an example of running the algorithm on simulated one-dimensional data is provided.

```julia
using Julienne 

# generate random data around a sinusoidal curve
sims_tot = 200
x = range(1, stop = 4pi, length = sims_tot)
y = 2.0 .* sin.(x) .+ 0.05 .* randn(sims_tot)
scatter(x, y, legend = false)

# run the streaming lm & partitioning method
partitions = stream_lm()

# plot the data together with estimated lines & partitions
plot_partitions(partitions, x, y)
```
