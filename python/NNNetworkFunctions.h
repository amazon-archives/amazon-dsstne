/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __NNNETWORKFUNCTIONS_H__
#define __NNNETWORKFUNCTIONS_H__

class NNNetworkFunctions {
    public:
        static PyObject* ClearDataSets(PyObject* self, PyObject* args);
        static PyObject* LoadDataSets(PyObject* self, PyObject* args);
        static PyObject* Randomize(PyObject* self, PyObject* args);
        static PyObject* Validate(PyObject* self, PyObject* args);
        static PyObject* Train(PyObject* self, PyObject* args);
        static PyObject* PredictBatch(PyObject* self, PyObject* args);
        static PyObject* CalculateTopK(PyObject* self, PyObject* args);
        static PyObject* PredictTopK(PyObject* self, PyObject* args);
        static PyObject* CalculateMRR(PyObject* self, PyObject* args);
        static PyObject* SaveBatch(PyObject* self, PyObject* args);
        static PyObject* DumpBatch(PyObject* self, PyObject* args);
        static PyObject* SaveLayer(PyObject* self, PyObject* args);
        static PyObject* DumpLayer(PyObject* self, PyObject* args);
        static PyObject* SaveWeights(PyObject* self, PyObject* args);
        static PyObject* LockWeights(PyObject* self, PyObject* args);
        static PyObject* UnlockWeights(PyObject* self, PyObject* args);
        static PyObject* SaveNetCDF(PyObject* self, PyObject* args);
        static PyObject* P2P_Bcast(PyObject* self, PyObject* args);
        static PyObject* P2P_Allreduce(PyObject* self, PyObject* args);
};

/**
 * Clear the data sets from the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkFunctions::ClearDataSets(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    pNetwork->ClearDataSets();
    Py_RETURN_NONE;
}
  
/**
 * Load a Python list of data sets into the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param pDataSetBaseList - a list of encapsulated NNDataSetBase*
 * @throws exception upon failure to parse arguments or to construct a vector from the list
 */
PyObject* NNNetworkFunctions::LoadDataSets(PyObject* self, PyObject* args) {
    PyObject *pDataSetBaseList = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, PyObject*>(args, pDataSetBaseList, "neural network", "OO");
    if (pNetwork == NULL) return NULL;
    vector<NNDataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNNetworkFunctions::LoadDataSets received empty vDataSetBase vector");
        return NULL;
    }
    pNetwork->LoadDataSets(vDataSetBase);
    Py_RETURN_NONE;
}
  
/**
 * Randomize the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkFunctions::Randomize(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    pNetwork->Randomize();
    Py_RETURN_NONE;
}
  
/**
 * Validate the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkFunctions::Validate(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->Validate());
}
  
/**
 * Train the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param epochs - the epochs unsigned integer
 * @param alpha - the alpha float
 * @param lambda - the lambda float
 * @param lambda1 - the lambda1 float
 * @param mu - the mu float
 * @param mu1 - the mu1 float
 * @return a PyObject* that references a float that reports the average error
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkFunctions::Train(PyObject* self, PyObject* args) {
    uint32_t epochs = 0;
    NNFloat alpha = 0.0, lambda = 0.0, lambda1 = 0.0, mu = 0.0, mu1 = 0.0;
    NNNetwork* pNetwork = parsePtrAndSixValues<NNNetwork*, uint32_t, NNFloat, NNFloat, NNFloat, NNFloat, NNFloat>(args,
                                                                                                                  epochs,
                                                                                                                  alpha,
                                                                                                                  lambda,
                                                                                                                  lambda1,
                                                                                                                  mu,
                                                                                                                  mu1,
                                                                                                                  "neural network",
                                                                                                                  "OIfffff");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("f", pNetwork->Train(epochs, alpha, lambda, lambda1, mu, mu1));
}
  
/**
 * Predict batch for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param layers - an unsigned integer that specifies the number of layers
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkFunctions::PredictBatch(PyObject* self, PyObject* args) {
    uint32_t layers = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, uint32_t>(args, layers, "neural network", "OI");
    if (pNetwork == NULL) return NULL;
    pNetwork->PredictBatch(layers);
    Py_RETURN_NONE;
}

/**
 * Calculate the top K results for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param layer - a string that specifies the layer name
 * @param k - an unsigned integer that specifies K
 * @param pbKey - an encapsulated float GpuBuffer
 * @param pbValue - an encapsulates unsigned integer GpuBuffer
 * @throws exception upon failure to parse arguments or to open the GpuBuffer capsules
 */
PyObject* NNNetworkFunctions::CalculateTopK(PyObject* self, PyObject* args) {
    char const* layer = NULL;
    uint32_t k = 0;
    PyObject *pbKeyCapsule = NULL, *pbValueCapsule = NULL;
    NNNetwork* pNetwork = parsePtrAndFourValues<NNNetwork*, char const*, uint32_t, PyObject*, PyObject*>(args,
                                                                                                         layer,
                                                                                                         k,
                                                                                                         pbKeyCapsule,
                                                                                                         pbValueCapsule,
                                                                                                         "neural network",
                                                                                                         "OsIOO");
    if (pNetwork == NULL) return NULL;
    GpuBuffer<NNFloat>* pbKey = reinterpret_cast<GpuBuffer<NNFloat>*>(PyCapsule_GetPointer(pbKeyCapsule, "float gpu buffer"));
    GpuBuffer<uint32_t>* pbValue = reinterpret_cast<GpuBuffer<uint32_t>*>(PyCapsule_GetPointer(pbValueCapsule, "unsigned gpu buffer"));
    if (pbKey == NULL || pbValue == NULL) return NULL;
    pNetwork->CalculateTopK(string(layer), k, pbKey, pbValue);
    Py_RETURN_NONE;
}

/**
 * Do a prediction calculation and return the top K results for a neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param pDataSetBaseList - a list of encapsulated NNDataSetBase*
 * @param pCDL - an encapsulated CDL*
 * @param k - an unsigned integer that specifies the number K of results
 * @return a PyObject* that references a list of lists wherein the inner lists contains three floats
 * @throws exception upon failure to parse arguments, to get the CDL* from its capsule, or to construct a vector<NNDataSetBase*> from the list
 */
PyObject* NNNetworkFunctions::PredictTopK(PyObject* self, PyObject* args) {
    PyObject *pDataSetBaseList = NULL, *pCdlCapsule = NULL;
    uint32_t k = 0;
    NNNetwork* pNetwork = parsePtrAndThreeValues<NNNetwork*, PyObject*, PyObject*, uint32_t>(args, pDataSetBaseList, pCdlCapsule, k, "neural network", "OOOI");
    if (pNetwork == NULL) return NULL;
    CDL* pCDL = reinterpret_cast<CDL*>(PyCapsule_GetPointer(pCdlCapsule, "cdl"));
    if (pCDL == NULL) return NULL;
    vector<NNDataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNNetworkFunctions::PredictTopK received empty vDataSetBase vector");
        return NULL;
    }
    return dsstnecalculate_PredictTopK(pNetwork, vDataSetBase, *pCDL, k);
}
  
/**
 * Calculate the MRR for a neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param pDataSetBaseList - a list of encapsulated NNDataSetBase*
 * @param pOutputLayer - an encapsulated NNLayer* that references the output layer
 * @param outputIndex - an unsigned integer that represents the index to the output data set
 * @param NxOutput - an unsigned integer that represents the leading dimension of the output layer
 * @return a PyObject* that references the MRR float
 * @throws exception upon failure to parse arguments, to get the NNLayer* from its capsule, or to construct a vector<NNDataSetBase*> from the list
 */
PyObject* NNNetworkFunctions::CalculateMRR(PyObject* self, PyObject* args) {
    PyObject *pDataSetBaseList = NULL, *pOutputLayerCapsule=NULL;
    uint32_t outputIndex = 0, NxOutput = 0;
    NNNetwork* pNetwork = parsePtrAndFourValues<NNNetwork*, PyObject*, PyObject*, uint32_t, uint32_t>(args,
                                                                                                      pDataSetBaseList,
                                                                                                      pOutputLayerCapsule,
                                                                                                      outputIndex,
                                                                                                      NxOutput,
                                                                                                      "neural network",
                                                                                                      "OOOII");
    if (pNetwork == NULL) return NULL;
    NNLayer* pOutputLayer = reinterpret_cast<NNLayer*>(PyCapsule_GetPointer(pOutputLayerCapsule, "layer"));
    if (pOutputLayer == NULL) return NULL;
    vector<NNDataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNNetworkFunctions::CalculateMRR received empty vDataSetBase vector");
        return NULL;
    }
    return dsstnecalculate_CalculateMRR(pNetwork, vDataSetBase, pOutputLayer, outputIndex, NxOutput);
}
  
/**
 * Save the batch to a file for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param fname - a string that specifies the filename
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkFunctions::SaveBatch(PyObject* self, PyObject* args) {
    char const* fname = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, fname, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    pNetwork->SaveBatch(string(fname));
    Py_RETURN_NONE;
}

/**
 * Dump the batch to a FILE for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param fp - an encapsulated FILE*
 * @throws exception upon failure to parse arguments or to open the FILE pointer capsule
 */
PyObject* NNNetworkFunctions::DumpBatch(PyObject* self, PyObject* args) {
    PyObject* fpCapsule = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, PyObject*>(args, fpCapsule, "neural network", "OO");
    if (pNetwork == NULL) return NULL;
    FILE* fp = reinterpret_cast<FILE*>(PyCapsule_GetPointer(fpCapsule, "file pointer"));
    if (fp == NULL) return NULL;
    pNetwork->DumpBatch(fp);
    Py_RETURN_NONE;
}

/**
 * Save the layer to a file for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param fname - a string that specifies the filename
 * @param layer - a string that specifies the layer name
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkFunctions::SaveLayer(PyObject* self, PyObject* args) {
    char const *fname = NULL, *layer = NULL;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, char const*, char const*>(args, fname, layer, "neural network", "Oss");
    if (pNetwork == NULL) return NULL;
    pNetwork->SaveLayer(string(fname), string(layer));
    Py_RETURN_NONE;
}

/**
 * Dump the layer to a FILE for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param fp - an encapsulated FILE*
 * @param layer - a string that specifies the layer name
 * @throws exception upon failure to parse arguments or to open the FILE pointer capsule
 */
PyObject* NNNetworkFunctions::DumpLayer(PyObject* self, PyObject* args) {
    PyObject* fpCapsule = NULL;
    char const* layer = NULL;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, PyObject*, char const*>(args, fpCapsule, layer, "neural network", "OOs");
    if (pNetwork == NULL) return NULL;
    FILE* fp = reinterpret_cast<FILE*>(PyCapsule_GetPointer(fpCapsule, "file pointer")); // Need a function that opens a FILE and returns a Python capsule
    if (fp == NULL) return NULL;
    pNetwork->DumpLayer(fp, string(layer));
    Py_RETURN_NONE;
}

/**
 * Save the weights connecting two layers to a file for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param fname - a string that specifies the filename
 * @param inputLayer - a string that specifies the input layer name
 * @param outputLayer - a string that specifies the output layer name
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkFunctions::SaveWeights(PyObject* self, PyObject* args) {
    char const *fname = NULL, *inputLayer = NULL, *outputLayer = NULL;
    NNNetwork* pNetwork =
        parsePtrAndThreeValues<NNNetwork*, char const*, char const*, char const*>(args, fname, inputLayer, outputLayer, "neural network", "Osss");
    if (pNetwork == NULL) return NULL;
    pNetwork->SaveWeights(string(fname), string(inputLayer), string(outputLayer));
    Py_RETURN_NONE;
}

/**
 * Lock the weights connecting two layers for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param inputLayer - a string that specifies the input layer name
 * @param outputLayer - a string that specifies the output layer name
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkFunctions::LockWeights(PyObject* self, PyObject* args) {
    char const *inputLayer = NULL, *outputLayer = NULL;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->LockWeights(string(inputLayer), string(outputLayer)));
}

/**
 *Unlock the weights connecting two layers for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param inputLayer - a string that specifies the input layer name
 * @param outputLayer - a string that specifies the output layer name
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkFunctions::UnlockWeights(PyObject* self, PyObject* args) {
    char const *inputLayer = NULL, *outputLayer = NULL;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->UnlockWeights(string(inputLayer), string(outputLayer)));
}

/**
 * Save the training results to a CDF file for the neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @param fname - a string that specifies the CDF filename
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkFunctions::SaveNetCDF(PyObject* self, PyObject* args) {
    char const* fname = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, fname, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SaveNetCDF(string(fname)));
}

/**
 * Broadcast data from process 0 to all other processes for the neural network.
 * @param pNetwork - the encapsulated NNNetwork*
 * @param pBuffer - an encapsulated pointer that references the broadcast buffer
 * @param size - an unsigned integer that specifies the size of the broadcast buffer
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments, or to open or rename the buffer capsule, or to build the return value
 */
PyObject* NNNetworkFunctions::P2P_Bcast(PyObject* self, PyObject* args) {
    PyObject* capsule = NULL;
    size_t size = 0;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, PyObject*, size_t>(args, capsule, size, "neural network", "OOI");
    if (pNetwork == NULL) return NULL;
    // Save the capsule name, override it with NULL, get the pointer using NULL for the name, and restore the capsule name.
    // Ugly, but a way to ignore the capsule name and hence treat the encapsulated pointer as typeless, i.e., void*
    char const* name = PyCapsule_GetName(capsule);
    // Use PyErr_Occurred() to check for an error instead of checking name == NULL, because the name could in principle be NULL.
    if (PyErr_Occurred != NULL) return NULL;
    if (PyCapsule_SetName(capsule, NULL) != 0) return NULL;
    void* pBuffer = PyCapsule_GetPointer(capsule, NULL);
    if (PyCapsule_SetName(capsule, name) != 0) return NULL;
    if (pBuffer == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->P2P_Bcast(pBuffer, size));
}

/**
 * Reduce a buffer across all processes for the neural network.
 * @param pNetwork - the encapsulated NNNetwork*
 * @param pBuffer - an encapsulated pointer that references the buffer
 * @param size - an unsigned integer that specifies the size of the buffer
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments, or to open the buffer capsule, or to build the return value
 */
PyObject* NNNetworkFunctions::P2P_Allreduce(PyObject* self, PyObject* args) {
    PyObject* capsule = NULL;
    size_t size = 0;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, PyObject*, size_t>(args, capsule, size, "neural network", "OOI");
    if (pNetwork == NULL) return NULL;
    NNFloat* pBuffer = reinterpret_cast<NNFloat*>(PyCapsule_GetPointer(capsule, "float"));
    if (pBuffer == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->P2P_Allreduce(pBuffer, size));
}

#endif
