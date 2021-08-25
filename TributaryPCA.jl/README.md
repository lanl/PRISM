TributaryPCA.jl
=========

TributaryPCA.jl is a Julia implementation of the AdaOja straming PCA algorithm in a distributed computing setting with MPI. 

![TributaryPCA](trib_pca_logo.png)

Example
------------

Here, example command line code for testing the package is given below. Two testing scripts are including in the source folder.

First, generate fake data partitions, where the first cml input indicates the number of partitions and the second cml input indicates a folder where the data partitions can be saved (if the path does not exist already, it will automatically create one):

```shell
julia gen_oja_mpi_data_par.jl 4 mpi_data
```

And, given that a proper MPI version has been installed and configured, run the testing script with appropriate number of partitions and input directories (where the data partitions are stored):

```shell
mpiexec -n 4 julia --project run_oja_mpi.jl 4 mpi_data
```


