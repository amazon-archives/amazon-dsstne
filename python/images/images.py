'''
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
'''

'''
This Python program is intended for use with the CIFAR-10 image recognition test data that may be installed by
following the instructions explained in the comments in the "amazon-dsstne/samples/cifar-10/dparse.cpp" file.
'''

# Import MPI before calling dsstnecuda.Startup to avoid opal_init(), orte_init(), and ompi_rte_init() failures.
from mpi4py import MPI
import dsstne as dn
import numpy as np
# import ctypes
# import math
import time
import sys

startTime = time.time()

# Initialize GPU network
dn.Startup(sys.argv) # The command-line arguments don't get used because 'import MPI' previously called MPI_Init().
if (len(sys.argv) == 2):
    try:
        CDL = dn.CreateCDLFromJSON(sys.argv[1])
    except:
        print "**** error,", sys.argv[0], "could not parse CDL file", sys.argv[1]
        sys.exit()
else:
    print "**** error: you must provide a CDL file to", sys.argv[0], "(typically either train.cdl or predict.cdl)"
    sys.exit()

print "**** random seed =", dn.GetCDLRandomSeed(CDL)
dn.SetRandomSeed(dn.GetCDLRandomSeed(CDL))

# Load training data
print "*** data filename =", dn.GetCDLDataFileName(CDL)
DataSetList = dn.LoadNetCDF(dn.GetCDLDataFileName(CDL))

# Create neural network
print "**** network filename =", dn.GetCDLNetworkFileName(CDL)
if (dn.GetCDLMode(CDL) == "Prediction"):
    Network = dn.LoadNeuralNetworkNetCDF(dn.GetCDLNetworkFileName(CDL), dn.GetCDLBatch(CDL))
else:
    Network = dn.LoadNeuralNetworkJSON(dn.GetCDLNetworkFileName(CDL), dn.GetCDLBatch(CDL), DataSetList)

# Dump memory usage
gpuMemoryUsage, cpuMemoryUsage = dn.GetMemoryUsage()
print "**** GPU memory usage:", gpuMemoryUsage, " KB"
print "**** CPU memory usage:", cpuMemoryUsage, " KB"

# Load the data sets
dn.LoadDataSets(Network, DataSetList)

if (dn.GetCDLMode(CDL) == "Training"):
    dn.SetCheckpoint(Network, dn.GetCDLCheckpointFileName(CDL), dn.GetCDLCheckpointInterval(CDL));
    CheckpointFileName, CheckpointInterval = dn.GetCheckpoint(Network)
    print "**** checkpoint filename =", CheckpointFileName
    print "**** checkpoint interval =", CheckpointInterval

# Train or predict based on operating mode
if (dn.GetCDLMode(CDL) == "Training"):
    dn.SetTrainingMode(Network, dn.GetCDLOptimizer(CDL))
    alpha = dn.GetCDLAlpha(CDL)
    lambda1 = 0.0
    mu1 = 0.0
    epochs = 0
    while (epochs < dn.GetCDLEpochs(CDL)):
        dn.Train(Network, dn.GetCDLAlphaInterval(CDL), alpha, dn.GetCDLLambda(CDL), lambda1, dn.GetCDLMu(CDL), mu1)
        alpha = alpha * dn.GetCDLAlphaMultiplier(CDL)
        epochs = epochs + dn.GetCDLAlphaInterval(CDL)
    print "**** saving training results to file: ", dn.GetCDLResultsFileName(CDL)
    dn.SaveNetCDF(Network, dn.GetCDLResultsFileName(CDL))
else:
    K = 10
    topK = dn.PredictTopK(Network, DataSetList, CDL, K)
    print "**** top", K, "results follow:"
    # Formatted printing is cumbersome
    for index, value in enumerate(topK, 1):
        print format(index, '3d'),
        for i, x in enumerate(value): print format(x, '.3f'),
        print

# Dump memory usage
gpuMemoryUsage, cpuMemoryUsage = dn.GetMemoryUsage()
print "**** GPU memory usage:", gpuMemoryUsage, " KB"
print "**** CPU memory usage:", cpuMemoryUsage, " KB"

# Report the network name and some other details
print "**** network name =", dn.GetName(Network)
Layers = dn.GetLayers(Network)
print "**** network layers =", Layers

# Delete the CDL instance, the training data, and the neural network and then shut down the GPU network
dn.DeleteCDL(CDL)
for i in range(0, len(DataSetList)):
    dn.DeleteDataSet(DataSetList[i]);
dn.DeleteNNNetwork(Network)
dn.Shutdown()

endTime = time.time()
print "**** time for cuda:  ", endTime-startTime
