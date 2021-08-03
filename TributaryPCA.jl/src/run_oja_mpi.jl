"""
Script to run the distributed AdaOja algorithm for streaming PCA

Author: Wayne Wang
Last modified: 08/02/2021
"""


using TributaryPCA


MPI.Init()
comm = MPI.COMM_WORLD

# run steaming PCA
num_proc = parse(Int64, ARGS[1]) #take cml input for number of MPi processors
input_dir = ARGS[2]
TributaryPCA.oja_mpi(MPI.Comm_rank(comm), MPI.Comm_size(comm), comm, input_dir)

# terminates the MPI environment
MPI.Finalize()