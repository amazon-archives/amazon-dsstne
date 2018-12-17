'''
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
'''

# Import MPI before calling dsstnecuda.Startup to avoid opal_init(), orte_init(), and ompi_rte_init() failures.
from mpi4py import MPI
import dsstne as dn
import numpy as np
# import ctypes
import math
import time
import sys

startTime = time.time()

# Initialize GPU network
dn.Startup(sys.argv) # The command-line arguments don't get used because 'import MPI' previously called MPI_Init().
dn.SetRandomSeed(6436215)

# Define some constants
batch = 8192
total_epochs = 48
training_epochs = 8
epochs = 0
alpha = 0.001
lmbda = 0.0
lmbda1 = 0.0
mu = 0.9
mu1 = 0.999

# Load data
DataSetList = dn.LoadNetCDF("train_small.nc")

dn.SetStreaming(DataSetList[0], True)
dn.SetStreaming(DataSetList[1], True)
dn.SetStreaming(DataSetList[2], True)

# Dump memory usage
gpuMemoryUsage, cpuMemoryUsage = dn.GetMemoryUsage()
print "**** GPU memory usage:", gpuMemoryUsage, " KB"
print "**** CPU memory usage:", cpuMemoryUsage, " KB"

# Create neural network
if len(sys.argv) < 2:
    Network = dn.LoadNeuralNetworkJSON("search.json", batch, DataSetList)
else:
    Network = dn.LoadNeuralNetworkNetCDF(sys.argv[1], batch)

# Dump memory usage
gpuMemoryUsage, cpuMemoryUsage = dn.GetMemoryUsage()
print "**** GPU memory usage:", gpuMemoryUsage, " KB"
print "**** CPU memory usage:", cpuMemoryUsage, " KB"

# Load the data sets
dn.LoadDataSets(Network, DataSetList)
dn.SetCheckpoint(Network, "checkpoint/check", 1)
CheckpointFileName, CheckpointInterval = dn.GetCheckpoint(Network)
print "**** checkpoint filename =", CheckpointFileName
print "**** checkpoint interval =", CheckpointInterval

# Train
dn.SetTrainingMode(Network, "Adam")
dn.SetDebugLevel(Network, True)
while (epochs < total_epochs):
    dn.Train(Network, training_epochs, alpha, lmbda, lmbda1, mu, mu1)
    alpha = alpha * 0.5
    epochs = epochs + training_epochs

# Save the final neural network
dn.SaveNetCDF(Network, "results/search_network.nc")

# Dump memory usage
gpuMemoryUsage, cpuMemoryUsage = dn.GetMemoryUsage()
print "**** GPU memory usage:", gpuMemoryUsage, " KB"
print "**** CPU memory usage:", cpuMemoryUsage, " KB"

# Delete the network and data sets then shut down the GPU network
for i in range(0, len(DataSetList)):
    dn.DeleteDataSet(DataSetList[i])
dn.DeleteNNNetwork(Network)
dn.Shutdown()

endTime = time.time()
print "**** Time for cuda:  ", endTime-startTime
