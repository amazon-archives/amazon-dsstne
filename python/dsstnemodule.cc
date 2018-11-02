/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

// The following .h files contain function definitions in order that those functions are compiled in the same compilation unit
// as this dsstnemodule.c file. A single compilation unit is necessary so that NumPy functions such as PyArray_SimpleNew() appear
// in the same compilation unit as the import_matrix() function that is called from the initdsstne() function below; otherwise,
// a segmentation fault will occur.
#include "dsstnemodule.h"
#include "dsstnecalculate.h"
#include "CDLAccessors.h"
#include "utilities.h"
#include "NNNetworkFunctions.h"
#include "NNNetworkAccessors.h"
#include "NNLayerAccessors.h"
#include "NNWeightAccessors.h"
#include "NNDataSetAccessors.h"
#include "utilities.h"

// See section 1.4 of https://docs.python.org/3/extending/extending.html
static PyMethodDef dsstneMethods[] = {
    {"GetCDLRandomSeed", CDLAccessors::GetRandomSeed, METH_VARARGS,
     "Get random seed from the source CDL"},
    
    {"SetCDLRandomSeed", CDLAccessors::SetRandomSeed, METH_VARARGS,
     "Set random seed in the destination CDL"},
    
    {"GetCDLEpochs", CDLAccessors::GetEpochs, METH_VARARGS,
     "Get epochs from the source CDL"},
    
    {"SetCDLEpochs", CDLAccessors::SetEpochs, METH_VARARGS,
     "Set epochs in the destination CDL"},
    
    {"GetCDLBatch", CDLAccessors::GetBatch, METH_VARARGS,
     "Get batch from the source CDL"},
    
    {"SetCDLBatch", CDLAccessors::SetBatch, METH_VARARGS,
     "Set batch in the destination CDL"},
    
    {"GetCDLCheckpointInterval", CDLAccessors::GetCheckpointInterval, METH_VARARGS,
     "Get checkpoint interval from the source CDL"},
    
    {"SetCDLCheckpointInterval", CDLAccessors::SetCheckpointInterval, METH_VARARGS,
     "Set checkpoint interval in the destination CDL"},
    
    {"GetCDLAlphaInterval", CDLAccessors::GetAlphaInterval, METH_VARARGS,
     "Get alpha interval from the source CDL"},
    
    {"SetCDLAlphaInterval", CDLAccessors::SetAlphaInterval, METH_VARARGS,
     "Set alpha interval in the destination CDL"},
    
    {"GetCDLShuffleIndexes", CDLAccessors::GetShuffleIndexes, METH_VARARGS,
     "Get shuffle indexes from the source CDL"},
    
    {"SetCDLShuffleIndexes", CDLAccessors::SetShuffleIndexes, METH_VARARGS,
     "Set shuffle indexes in the destination CDL"},
    
    {"GetCDLAlpha", CDLAccessors::GetAlpha, METH_VARARGS,
     "Get alpha from the source CDL"},
    
    {"SetCDLAlpha", CDLAccessors::SetAlpha, METH_VARARGS,
     "Set alpha in the destination CDL"},
    
    {"GetCDLLambda", CDLAccessors::GetLambda, METH_VARARGS,
     "Get lambda from the source CDL"},
    
    {"SetCDLLambda", CDLAccessors::SetLambda, METH_VARARGS,
     "Set lambda in the destination CDL"},
    
    {"GetCDLMu", CDLAccessors::GetMu, METH_VARARGS,
     "Get mu from the source CDL"},
    
    {"SetCDLMu", CDLAccessors::SetMu, METH_VARARGS,
     "Set mu in the destination CDL"},
    
    {"GetCDLAlphaMultiplier", CDLAccessors::GetAlphaMultiplier, METH_VARARGS,
     "Get alpha multiplier from the source CDL"},
    
    {"SetCDLAlphaMultiplier", CDLAccessors::SetAlphaMultiplier, METH_VARARGS,
     "Set alpha multiplier in the destination CDL"},
    
    {"GetCDLMode", CDLAccessors::GetMode, METH_VARARGS,
     "Get the mode enumerator from the source CDL"},
    
    {"SetCDLMode", CDLAccessors::SetMode, METH_VARARGS,
     "Set the mode enumerator in the destination CDL"},
    
    {"GetCDLOptimizer", CDLAccessors::GetOptimizer, METH_VARARGS,
     "Get the training mode enumerator from the source CDL"},
    
    {"SetCDLOptimizer", CDLAccessors::SetOptimizer, METH_VARARGS,
     "Set the training mode enumerator in the destination CDL"},
    
    {"GetCDLNetworkFileName", CDLAccessors::GetNetworkFileName, METH_VARARGS,
     "Get network filename from the source CDL"},
    
    {"SetCDLNetworkFileName", CDLAccessors::SetNetworkFileName, METH_VARARGS,
     "Set network filename in the destination CDL"},
    
    {"GetCDLDataFileName", CDLAccessors::GetDataFileName, METH_VARARGS,
     "Get data filename from the source CDL"},
    
    {"SetCDLDataFileName", CDLAccessors::SetDataFileName, METH_VARARGS,
     "Set data filename in the destination CDL"},
    
    {"GetCDLCheckpointFileName", CDLAccessors::GetCheckpointFileName, METH_VARARGS,
     "Get checkpoint filename from the source CDL"},
    
    {"SetCDLCheckpointFileName", CDLAccessors::SetCheckpointFileName, METH_VARARGS,
     "Set checkpoint filename in the destination CDL"},
    
    {"GetCDLResultsFileName", CDLAccessors::GetResultsFileName, METH_VARARGS,
     "Get results filename from the source CDL"},
    
    {"SetCDLResultsFileName", CDLAccessors::SetResultsFileName, METH_VARARGS,
     "Set results filename in the destination CDL"},
    
    {"Startup", Utilities::Startup, METH_VARARGS,
     "Initialize the GPUs and MPI"},
    
    {"Shutdown", Utilities::Shutdown, METH_VARARGS,
     "Shutdown the GPUs"},
    
    {"CreateCDLFromJSON", Utilities::CreateCDLFromJSON, METH_VARARGS,
     "Create a CDL instance and initialize it from a JSON file"},
    
    {"CreateCDLFromDefaults", Utilities::CreateCDLFromDefaults, METH_VARARGS,
     "Create a CDL instance and initialize it with default values"},
    
    {"DeleteCDL", Utilities::DeleteCDL, METH_VARARGS,
     "Delete a CDL instance"},
    
    {"LoadNetCDF", Utilities::LoadDataSetFromNetCDF, METH_VARARGS,
     "Load a Python array (i.e. a list) of data sets from a CDF file"},
    
    {"DeleteDataSet", Utilities::DeleteDataSet, METH_VARARGS,
     "Delete a data set"},
    
    {"LoadNeuralNetworkNetCDF", Utilities::LoadNeuralNetworkFromNetCDF, METH_VARARGS,
     "Load a neural network from a CDF file"},
    
    {"LoadNeuralNetworkJSON", Utilities::LoadNeuralNetworkFromJSON, METH_VARARGS,
     "Load a neural network from a JSON config file, a batch number, and a Python list of data sets"},
    
    {"DeleteNNNetwork", Utilities::DeleteNNNetwork, METH_VARARGS,
     "Delete a neural network"},
    
    {"OpenFile", Utilities::OpenFile, METH_VARARGS,
     "Open a FILE* stream"},
    
    {"CloseFile", Utilities::CloseFile, METH_VARARGS,
     "Close a FILE* stream"},
    
    {"SetRandomSeed", Utilities::SetRandomSeed, METH_VARARGS,
     "Set random seed in the GPU"},
    
    {"GetMemoryUsage", Utilities::GetMemoryUsage, METH_VARARGS,
     "Get the GPU and CPU memory usage"},
    
    {"Transpose", Utilities::Transpose, METH_VARARGS,
     "Transpose a NumPy 2D matrix to create a contiguous matrix"},
    
    {"CreateFloatGpuBuffer", Utilities::CreateFloatGpuBuffer, METH_VARARGS,
     "Create a GPU Buffer of type NNFloat and of the specified size"},
    
    {"DeleteFloatGpuBuffer", Utilities::DeleteFloatGpuBuffer, METH_VARARGS,
     "Delete a GPU Buffer of type NNFloat"},
    
    {"CreateUnsignedGpuBuffer", Utilities::CreateUnsignedGpuBuffer, METH_VARARGS,
     "Create a GPU Buffer of type uint32_t and of the specified size"},
    
    {"DeleteUnsignedGpuBuffer", Utilities::DeleteUnsignedGpuBuffer, METH_VARARGS,
     "Delete a GPU Buffer of type uint32_te"},
    
    {"ClearDataSets", NNNetworkFunctions::ClearDataSets, METH_VARARGS,
     "Clear the data sets from the neural network"},
    
    {"LoadDataSets", NNNetworkFunctions::LoadDataSets, METH_VARARGS,
     "Load a Python list of data sets into the neural network"},
    
    {"Randomize", NNNetworkFunctions::Randomize, METH_VARARGS,
     "Randomize a neural network"},
    
    {"Validate", NNNetworkFunctions::Validate, METH_VARARGS,
     "Validate the network gradients numerically for the neural network"},
    
    {"Train", NNNetworkFunctions::Train, METH_VARARGS,
     "Train the neural network"},
    
    {"PredictBatch", NNNetworkFunctions::PredictBatch, METH_VARARGS,
     "Predict batch for the neural network"},
    
    {"CalculateTopK", NNNetworkFunctions::CalculateTopK, METH_VARARGS,
     "Calculate the top K results for the neural network"},
    
    {"PredictTopK", NNNetworkFunctions::PredictTopK, METH_VARARGS,
     "Do a prediction calculation and return the top K results for a neural network"},
    
    {"CalculateMRR", NNNetworkFunctions::CalculateMRR, METH_VARARGS,
     "Calculate the MRR for a neural network"},
    
    {"SaveBatch", NNNetworkFunctions::SaveBatch, METH_VARARGS,
     "Save the batch to a file for the neural network"},
    
    {"DumpBatch", NNNetworkFunctions::DumpBatch, METH_VARARGS,
     "Dump the batch to a FILE for the neural network"},
    
    {"SaveLayer", NNNetworkFunctions::SaveLayer, METH_VARARGS,
     "Save the specified layer to a file for the neural network"},
    
    {"DumpLayer", NNNetworkFunctions::DumpLayer, METH_VARARGS,
     "Dump the specified layer to a FILE for the neural network"},
    
    {"SaveWeights", NNNetworkFunctions::SaveWeights, METH_VARARGS,
     "Save the weights connecting two layers to a file for the neural network"},
    
    {"LockWeights", NNNetworkFunctions::LockWeights, METH_VARARGS,
     "Lock the weights connecting two layers for the neural network"},
    
    {"UnlockWeights", NNNetworkFunctions::UnlockWeights, METH_VARARGS,
     "Unlock the weights connecting two layers for the neural network"},
    
    {"SaveNetCDF", NNNetworkFunctions::SaveNetCDF, METH_VARARGS,
     "Save training results to a CDF file for the neural network"},
    
    {"P2P_Bcast", NNNetworkFunctions::P2P_Bcast, METH_VARARGS,
     "Broadcast data from process 0 to all other processes for the neural network"},
    
    {"P2P_Allreduce", NNNetworkFunctions::P2P_Allreduce, METH_VARARGS,
     "Reduce a buffer across all processes for the neural network"},
    
    {"GetBatch", NNNetworkAccessors::GetBatch, METH_VARARGS,
     "Get the batch from the source neural network"},
    
    {"SetBatch", NNNetworkAccessors::SetBatch, METH_VARARGS,
     "Set the batch in the destination neural network"},
    
    {"GetPosition", NNNetworkAccessors::GetPosition, METH_VARARGS,
     "Get the position from the source neural network"},
    
    {"SetPosition", NNNetworkAccessors::SetPosition, METH_VARARGS,
     "Set the position in the destination neural network"},
    
    {"GetShuffleIndices", NNNetworkAccessors::GetShuffleIndices, METH_VARARGS,
     "Get the shuffle indices from the source neural network; unwrap the unnecessary tuple"},
    
    {"SetShuffleIndices", NNNetworkAccessors::SetShuffleIndices, METH_VARARGS,
     "Set the shuffle indices boolean in the destination neural network"},
    
    {"GetSparsenessPenalty", NNNetworkAccessors::GetSparsenessPenalty, METH_VARARGS,
     "Get the sparseness penalty p and beta from the source neural network"},
    
    {"SetSparsenessPenalty", NNNetworkAccessors::SetSparsenessPenalty, METH_VARARGS,
     "Set the sparseness penalty p and beta in the destination neural network"},
    
    {"GetDenoising", NNNetworkAccessors::GetDenoising, METH_VARARGS,
     "Get the denoising p from the source neural network and unwrap the unnecessary tuple"},
    
    {"SetDenoising", NNNetworkAccessors::SetDenoising, METH_VARARGS,
     "Set the denoising p in the destination neural network"},
    
    {"GetDeltaBoost", NNNetworkAccessors::GetDeltaBoost, METH_VARARGS,
     "Get the delta boost one and zero from the source neural network"},
    
    {"SetDeltaBoost", NNNetworkAccessors::SetDeltaBoost, METH_VARARGS,
     "Set the delta boost one and zero in the destination neural network"},
    
    {"GetDebugLevel", NNNetworkAccessors::GetDebugLevel, METH_VARARGS,
     "Get the debug level from the source neural network"},
    
    {"SetDebugLevel", NNNetworkAccessors::SetDebugLevel, METH_VARARGS,
     "Set the debug level in the destination neural network"},
    
    {"GetCheckpoint", NNNetworkAccessors::GetCheckpoint, METH_VARARGS,
     "Get the checkpoint file name and interval from the source neural network"},
    
    {"SetCheckpoint", NNNetworkAccessors::SetCheckpoint, METH_VARARGS,
     "Set the checkpoint filename and interval in the destination neural network"},
    
    {"GetLRN", NNNetworkAccessors::GetLRN, METH_VARARGS,
     "Get the local response network (LRN) k, n, alpha, and beta from the source neural network"},
    
    {"SetLRN", NNNetworkAccessors::SetLRN, METH_VARARGS,
     "Set the local response network (LRN) k, n, alpha, and beta in the destination neural network"},
    
    {"GetSMCE", NNNetworkAccessors::GetSMCE, METH_VARARGS,
     "Get the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale from the source neural network"},
    
    {"SetSMCE", NNNetworkAccessors::SetSMCE, METH_VARARGS,
     "Set the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale in the destination neural network"},
    
    {"GetMaxout", NNNetworkAccessors::GetMaxout, METH_VARARGS,
     "Get the maxout k from the source neural network and unwrap the unnecessary tuple"},
    
    {"SetMaxout", NNNetworkAccessors::SetMaxout, METH_VARARGS,
     "Set the maxout k in the destination neural network"},
    
    {"GetExamples", NNNetworkAccessors::GetExamples, METH_VARARGS,
     "Get the examples from the source neural network"},
    
    {"GetWeight", NNNetworkAccessors::GetWeight, METH_VARARGS,
     "Get the set of weights connecting the specified input and output layers from the source neural network"},
    
    {"GetBufferSize", NNNetworkAccessors::GetBufferSize, METH_VARARGS,
     "Get the buffer size of the specified layer for the source neural network"},
    
    {"GetLayer", NNNetworkAccessors::GetLayer, METH_VARARGS,
     "Get the specified layer from the source neural network"},
    
    {"GetLayers", NNNetworkAccessors::GetLayers, METH_VARARGS,
     "Get the list of layer names from the source neural network"},
    
    {"GetName", NNNetworkAccessors::GetName, METH_VARARGS,
     "Get the name of the source neural network"},
    
    {"GetUnitBuffer", NNNetworkAccessors::GetUnitBuffer, METH_VARARGS,
     "Get the unit buffer for the specified layer from the source neural network"},
    
    {"GetDeltaBuffer", NNNetworkAccessors::GetDeltaBuffer, METH_VARARGS,
     "Get the delta buffer for the specified layer from the source neural network"},
    
    {"GetWeightBuffer", NNNetworkAccessors::GetWeightBuffer, METH_VARARGS,
     "Get the weight buffer for the specified input and output layers from the source neural network"},
    
    {"GetScratchBuffer", NNNetworkAccessors::GetScratchBuffer, METH_VARARGS,
     "Get the current scratch buffer from the source neural network and resize it if necessary"},
    
    {"GetP2PSendBuffer", NNNetworkAccessors::GetP2PSendBuffer, METH_VARARGS,
     "Get the current local send buffer from the source neural network"},
    
    {"GetP2PReceiveBuffer", NNNetworkAccessors::GetP2PReceiveBuffer, METH_VARARGS,
     "Get the current local receive buffer from the source neural network"},
    
    {"GetP2PCPUBuffer", NNNetworkAccessors::GetP2PCPUBuffer, METH_VARARGS,
     "Get the system memory work buffer from the source neural network"},
    
    {"GetPeerBuffer", NNNetworkAccessors::GetPeerBuffer, METH_VARARGS,
     "Get the current adjacent peer receive buffer from the source neural network"},
    
    {"GetPeerBackBuffer", NNNetworkAccessors::GetPeerBackBuffer, METH_VARARGS,
     "Get the current adjacent peer send buffer from the source neural network"},
    
    {"SetClearVelocity", NNNetworkAccessors::SetClearVelocity, METH_VARARGS,
     "Set the clear velocity flag in the destination neural network"},
    
    {"SetTrainingMode", NNNetworkAccessors::SetTrainingMode, METH_VARARGS,
     "Set the training mode enumerator in the destination neural network"},
    
    {"GetLayerName", NNLayerAccessors::GetName, METH_VARARGS,
     "Get the name from the source layer"},
    
    {"GetKind", NNLayerAccessors::GetKind, METH_VARARGS,
     "Get the kind enumerator from the source layer"},
    
    {"GetType", NNLayerAccessors::GetType, METH_VARARGS,
     "Get the type enumerator from the source layer"},
    
    {"GetAttributes", NNLayerAccessors::GetAttributes, METH_VARARGS,
     "Get the attributes from the source layer"},
    
    {"GetDataSetBase", NNLayerAccessors::GetDataSet, METH_VARARGS,
     "Get the data set from the source layer"},
    
    {"GetNumDimensions", NNLayerAccessors::GetNumDimensions, METH_VARARGS,
     "Get the number of dimensions from the source layer"},
    
    {"GetDimensions", NNLayerAccessors::GetDimensions, METH_VARARGS,
     "Get dimensions from the source layer"},
    
    {"GetLocalDimensions", NNLayerAccessors::GetLocalDimensions, METH_VARARGS,
     "Get local dimensions from the source layer"},
    
    {"GetKernelDimensions", NNLayerAccessors::GetKernelDimensions, METH_VARARGS,
     "Get kernel dimensions from the source layer"},
    
    {"GetKernelStride", NNLayerAccessors::GetKernelStride, METH_VARARGS,
     "Get kernel stride from the source layer"},
    
    {"GetUnits", NNLayerAccessors::GetUnits, METH_VARARGS,
     "Modify the destination float32 NumPy array beginning at a specified index by copying the units from the source layer"},
    
    {"SetUnits", NNLayerAccessors::SetUnits, METH_VARARGS,
     "Set the units of the destination layer by copying the units from a source float32 Numpy array"},
    
    {"GetDeltas", NNLayerAccessors::GetDeltas, METH_VARARGS,
     "Modify the destination float32 NumPy array beginning at a specified index by copying the deltas from the source layer"},
    
    {"SetDeltas", NNLayerAccessors::SetDeltas, METH_VARARGS,
     "Set the deltas of the destination layer by copying the deltas from a source float32 Numpy array"},
    
    {"CopyWeights", NNWeightAccessors::CopyWeights, METH_VARARGS,
     "Copy the weights from the specified source weight to the destination weight"},
    
    {"SetWeights", NNWeightAccessors::SetWeights, METH_VARARGS,
     "Set the weights in the destination weight from a source NumPy array of weights"},
    
    {"SetBiases", NNWeightAccessors::SetBiases, METH_VARARGS,
     "Set the biases in the destination weight from a source NumPy array of biases"},
    
    {"GetWeights", NNWeightAccessors::GetWeights, METH_VARARGS,
     "Get the weights from the source weight"},
    
    {"GetBiases", NNWeightAccessors::GetBiases, METH_VARARGS,
     "Get the biases from the source weight"},
    
    {"SetNorm", NNWeightAccessors::SetNorm, METH_VARARGS,
     "Set the norm for the destination weight"},
    
    {"GetDataSetName", NNDataSetAccessors::GetDataSetName, METH_VARARGS,
     "Get the name from the source NNDataSetBase*"},
    
    {"CreateDenseDataSet", NNDataSetAccessors::CreateDenseDataSet, METH_VARARGS,
     "Create an encapsulated NNDataSetBase* from a dense NumPy array"},
    
    {"CreateSparseDataSet", NNDataSetAccessors::CreateSparseDataSet, METH_VARARGS,
     "Create an encapsulated NNDataSetBase* from a compressed sparse row (CSR) SciPy two-dimensional matrix"},
    
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// See https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
PyMODINIT_FUNC initdsstne(void) {
    (void) Py_InitModule("dsstne", dsstneMethods);
    import_array();
    // Add initialization code here
}
