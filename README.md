# PRISM
[![HSVGP Build Status][hsvgp-ci-img]](https://github.com/lanl/PRISM/actions)
[![Codecov][codecov-img]](https://codecov.io/gh/lanl/PRISM)

Programming Repository for In Situ Modeling

![PRISM](prism.png)

The Programming Repository for In Situ Modeling (PRISM) is a set of tools for fitting statistics and machine learning models
to simulation data inside the simulations as they are running. By fitting models inside running simulations, PRISM can be
used to analyze simulation data that is otherwise inaccessible because of I/O and storage bottlenecks associated with
exascale and other future high performance computing architectures. The tools are designed to implement a wide variety of
data analyses with an emphasis on spatiotemporal hierarchical Bayesian models. PRISM is efficient, scalable, and
streaming with estimation based on variational inference, advanced Monte Carlo techniques, and fast optimization
methods. The core modeling components aid this goal by imposing sparsity and approximate inference wherever possible.
These components are written in Julia, a high-level programming language designed for high performance. PRISM also
contains tools for interfacing with large-scale scientific simulations written in Fortran and C/C++. This layer of abstraction
allows the data scientist to construct analysis models in Julia without concern for the implementation details of the
simulation capability. With these components, PRISM can be used to unlock the full scientific potential of next-generation
HPC simulations.

[hsvgp-ci-img]: https://github.com/lanl/PRISM/workflows/HSVGP-CI/badge.svg
[codecov-img]: https://img.shields.io/codecov/c/github/lanl/PRISM/master.svg?label=codecov
