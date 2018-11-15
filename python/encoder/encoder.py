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
dn.SetRandomSeed(12345)

# Create neural network
batch = 256

print "**** Extracting input layer weights."

# Load data
DataSetList = dn.LoadNetCDF("eval.nc")

# Load trained network
if len(sys.argv) < 2:
    TrainedNetwork = dn.LoadNeuralNetworkNetCDF("search_network.nc", batch)
else:
    TrainedNetwork = dn.LoadNeuralNetworkJSON(sys.argv[1])

# Get the weights and biases and then delete the trained network
EncodingWeight = dn.GetWeight(TrainedNetwork, "Input_query", "Hidden_query")
EncodingArray = dn.GetWeights(EncodingWeight)
print "**** EncodingArray shape:", EncodingArray.shape
EncodingBiasArray = dn.GetBiases(EncodingWeight)
print "**** EncodingBiasArray shape:", EncodingBiasArray.shape
dn.DeleteNNNetwork(TrainedNetwork)

# Create the embedding network and load the data sets, weights, and biases from the training network
pEmbeddingNetwork = dn.LoadNeuralNetworkJSON("embedding.json", batch, DataSetList)
dn.LoadDataSets(pEmbeddingNetwork, DataSetList)
InputWeight = dn.GetWeight(pEmbeddingNetwork, "Input_asin", "Hidden_asin")
bEncoding = dn.SetWeights(InputWeight, EncodingArray)
bEncodingBias = dn.SetBiases(InputWeight, EncodingBiasArray)

# Generate and save embeddings piecemeal and advance by STRIDE between successive generations
HiddenLayer = dn.GetLayer(pEmbeddingNetwork, "Hidden_asin")
NxHidden, Ny, Nz, Nw = dn.GetLocalDimensions(HiddenLayer)
print "**** NxHidden, Ny Nz, Nw =", NxHidden, Ny, Nz, Nw
STRIDE = NxHidden * Ny * Nz * Nw
EmbeddingExamples = dn.GetExamples(pEmbeddingNetwork)
EmbeddingBatch = dn.GetBatch(pEmbeddingNetwork)
print "**** EmbeddingExamples =", EmbeddingExamples
EmbeddingArray = np.empty(shape=(EmbeddingExamples + batch, STRIDE), dtype=np.float32)
ZeroArray = np.zeros(EmbeddingExamples, dtype=np.float32)
print "**** EmbeddingArray shape =", EmbeddingArray.shape, "ZeroArray shape =", ZeroArray.shape
for pos in range(0, EmbeddingExamples, EmbeddingBatch):
    dn.SetPosition(pEmbeddingNetwork, pos)
    dn.PredictBatch(pEmbeddingNetwork, 0) # It is not possible to omit the second argument and let it default to 0
    innerBatch = EmbeddingBatch
    if (pos + innerBatch > EmbeddingExamples):
        innerBatch = EmbeddingExamples - pos
    dn.GetUnits(HiddenLayer, EmbeddingArray, pos * STRIDE)

# Normalize ASIN embeddings so we can just dot-product them with query embeddings (dirty trick but fair)
print "**** normalizing ASIN embeddings"
norms = np.linalg.norm(EmbeddingArray, axis=1, keepdims=True)
# Don't normalize the last row of the matrix because apparently the normalizer is zero and produces a warning;
# otherwise, the transpose could be expressed as 'EmbeddingArray /= norms'. The last row is ignored by transpose,
# so normalizing that row isn't necessary.
for i in range(0, EmbeddingExamples):
    if (norms[i] != 0.0):
        EmbeddingArray[i] /= norms[i]

# Transpose ASIN embeddings
print "**** transposing embeddings"
ASINWeightArray = np.empty(shape=(STRIDE, EmbeddingExamples), dtype=np.float32)
# EmbeddingArray.transpose() creates a non-contiguous 2D array that is incompatible with dn.SetWeights()
# so the slow Python transpose loops (below) may be used instead but dn.Transpose() is much faster.
# for i in range(0, EmbeddingExamples):
#     for j in range(0, STRIDE):
#         ASINWeightArray[j][i] = EmbeddingArray[i][j]
dn.Transpose(ASINWeightArray, EmbeddingArray)

# Delete network and data sets
print "**** deleting embedding network"
dn.DeleteNNNetwork(pEmbeddingNetwork)
print "**** deleting data sets"
for i in range(0, len(DataSetList)):
    dn.DeleteDataSet(DataSetList[i])

# Load final network
batch = 256
print "**** Calculating MRR"
DataSetList = dn.LoadNetCDF("eval.nc")
Network = dn.LoadNeuralNetworkJSON("inference.json", batch, DataSetList)
InputWeight = dn.GetWeight(Network, "Input_query", "Hidden_query")
InputArray = dn.GetWeights(InputWeight)
print "**** InputArray shape:", InputArray.shape
InputBiasArray = dn.GetBiases(InputWeight)
print "**** InputBiasArray shape:", InputBiasArray.shape
dn.SetWeights(InputWeight, EncodingArray)
dn.SetBiases(InputWeight, EncodingBiasArray)
OutputWeight = dn.GetWeight(Network, "Hidden_query", "Output_asin")
dn.SetWeights(OutputWeight, ASINWeightArray)
dn.SetBiases(OutputWeight, ZeroArray)
dn.LoadDataSets(Network, DataSetList)

# Copy embeddings weights and ASIN embeddings
InputLayer = dn.GetLayer(Network, "Input_query")
HiddenLayer = dn.GetLayer(Network, "Hidden_query")
OutputLayer = dn.GetLayer(Network, "Output_asin")
NxInput, Ny, Nz, Nw = dn.GetDimensions(InputLayer)
NxHidden, Ny, Nz, Nw = dn.GetDimensions(HiddenLayer)
NxOutput, Ny, Nz, Nw = dn.GetDimensions(OutputLayer)
print "**** NxOutput =", NxOutput

# Find output dataset
outputIndex = 0
print "**** DataSetList size =", len(DataSetList)
while (outputIndex < len(DataSetList) and dn.GetDataSetName(DataSetList[outputIndex]) != "target"):
    outputIndex = outputIndex + 1
print "**** outputIndex =", outputIndex

# Calculate MRR
MRR = dn.CalculateMRR(Network, DataSetList, OutputLayer, outputIndex, NxOutput)
print "**** MRR is", MRR

# Delete the network and data sets then shut down the GPU network
for i in range(0, len(DataSetList)):
    dn.DeleteDataSet(DataSetList[i])
dn.DeleteNNNetwork(Network)
dn.Shutdown()

endTime = time.time()
print "**** Time for cuda:  ", endTime-startTime
