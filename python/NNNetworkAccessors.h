
/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __NNNETWORKACCESSORS_H__
#define __NNNETWORKACCESSORS_H__

class NNNetworkAccessors {
    public:
        static PyObject* GetBatch(PyObject* self, PyObject* args);
        static PyObject* SetBatch(PyObject* self, PyObject* args);
        static PyObject* GetPosition(PyObject* self, PyObject* args);
        static PyObject* SetPosition(PyObject* self, PyObject* args);
        static PyObject* GetShuffleIndices(PyObject* self, PyObject* args);
        static PyObject* SetShuffleIndices(PyObject* self, PyObject* args);
        static PyObject* GetSparsenessPenalty(PyObject* self, PyObject* args);
        static PyObject* SetSparsenessPenalty(PyObject* self, PyObject* args);
        static PyObject* GetDenoising(PyObject* self, PyObject* args);
        static PyObject* SetDenoising(PyObject* self, PyObject* args);
        static PyObject* GetDeltaBoost(PyObject* self, PyObject* args);
        static PyObject* SetDeltaBoost(PyObject* self, PyObject* args);
        static PyObject* GetDebugLevel(PyObject* self, PyObject* args);
        static PyObject* SetDebugLevel(PyObject* self, PyObject* args);
        static PyObject* GetCheckpoint(PyObject* self, PyObject* args);
        static PyObject* SetCheckpoint(PyObject* self, PyObject* args);
        static PyObject* GetLRN(PyObject* self, PyObject* args);
        static PyObject* SetLRN(PyObject* self, PyObject* args);
        static PyObject* GetSMCE(PyObject* self, PyObject* args);
        static PyObject* SetSMCE(PyObject* self, PyObject* args);
        static PyObject* GetMaxout(PyObject* self, PyObject* args);
        static PyObject* SetMaxout(PyObject* self, PyObject* args);
        static PyObject* GetExamples(PyObject* self, PyObject* args);
        static PyObject* GetWeight(PyObject* self, PyObject* args);
        static PyObject* GetBufferSize(PyObject* self, PyObject* args);
        static PyObject* GetLayer(PyObject* self, PyObject* args);
        static PyObject* GetLayers(PyObject* self, PyObject* args);
        static PyObject* GetName(PyObject* self, PyObject* args);
        static PyObject* GetUnitBuffer(PyObject* self, PyObject* args);
        static PyObject* GetDeltaBuffer(PyObject* self, PyObject* args);
        static PyObject* GetWeightBuffer(PyObject* self, PyObject* args);
        static PyObject* GetScratchBuffer(PyObject* self, PyObject* args);
        static PyObject* GetP2PSendBuffer(PyObject* self, PyObject* args);
        static PyObject* GetP2PReceiveBuffer(PyObject* self, PyObject* args);
        static PyObject* GetP2PCPUBuffer(PyObject* self, PyObject* args);
        static PyObject* GetPeerBuffer(PyObject* self, PyObject* args);
        static PyObject* GetPeerBackBuffer(PyObject* self, PyObject* args);
        static PyObject* SetClearVelocity(PyObject* self, PyObject* args);
        static PyObject* SetTrainingMode(PyObject* self, PyObject* args);
};

/**
 * Get the batch from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the batch unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetBatch(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("I", pNetwork->GetBatch());
}

/**
 * Set the batch in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param batch - the batch unsigned integer
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetBatch(PyObject* self, PyObject* args) {
    uint32_t batch = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, uint32_t>(args, batch, "neural network", "OI");
    if (pNetwork == NULL) return NULL;
    pNetwork->SetBatch(batch);
    Py_RETURN_NONE;
}

/**
 * Get the position from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the position unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetPosition(PyObject* self, PyObject* args) {
  NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("I", pNetwork->GetPosition());
}

/**
 * Set the position in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param position - the position unsigned integer
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetPosition(PyObject* self, PyObject* args) {
    uint32_t position = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, uint32_t>(args, position, "neural network", "OI");
    if (pNetwork == NULL) return NULL;
    pNetwork->SetPosition(position);
    Py_RETURN_NONE;
}

/**
 * Get the shuffle indices flag from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the shuffle indices integer flag
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetShuffleIndices(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    int bShuffleIndices;
    tie(bShuffleIndices) = pNetwork->GetShuffleIndices();
    return Py_BuildValue("i", bShuffleIndices);
}

/**
 * Set the shuffle indices flag in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param bShuffleIndices - the shuffle indices integer flag
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetShuffleIndices(PyObject* self, PyObject* args) {
    int bShuffleIndices = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, int>(args, bShuffleIndices, "neural network", "Oi");
    if (pNetwork == NULL) return NULL;
    pNetwork->SetShuffleIndices(bShuffleIndices);
    Py_RETURN_NONE;
}

/**
 * Get the sparseness penalty p and beta from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references a list containing the p and beta NNFloats
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNNetworkAccessors::GetSparsenessPenalty(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    NNFloat p, beta;
    tie(p, beta) = pNetwork->GetSparsenessPenalty();
    return Py_BuildValue("[ff]", p, beta); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Set the sparseness penalty p and beta in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param p - the sparseness penalty p float
 * @param beta - the sparseness penalty beta float
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetSparsenessPenalty(PyObject* self, PyObject* args) {
    NNFloat p = 0.0, beta = 0.0;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, NNFloat, NNFloat>(args, p, beta, "neural network", "Off");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetSparsenessPenalty(p, beta));
}

/**
 * Get the denoising p from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the denoising p float
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetDenoising(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    NNFloat denoisingP;
    tie(denoisingP) = pNetwork->GetDenoising();
    return Py_BuildValue("f", denoisingP);
}

/**
 * Set the denoising p in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param denoisingP - the denoising p float
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetDenoising(PyObject* self, PyObject* args) {
    NNFloat denoisingP = 0.0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, NNFloat>(args, denoisingP, "neural network", "Of");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetDenoising(denoisingP));
}

/**
 * Get the delta boost one and zero floats from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references a list of the one and zero floats
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNNetworkAccessors::GetDeltaBoost(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    NNFloat one, zero;
    tie(one, zero) = pNetwork->GetDeltaBoost();
    return Py_BuildValue("[ff]", one, zero); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Set the delta boost one and zero in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param one - the delta boost one float
 * @param zero - the delta boost zero float
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetDeltaBoost(PyObject* self, PyObject* args) {
    NNFloat one = 0.0, zero = 0.0;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, NNFloat, NNFloat>(args, one, zero, "neural network", "Off");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetDeltaBoost(one, zero));
}

/**
 * Get the debug level flag from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the debug level integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetDebugLevel(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->GetDebugLevel());
}

/**
 * Set the debug level in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param bDebugLevel - the debug level flag integer
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetDebugLevel(PyObject* self, PyObject* args) {
    int bDebugLevel = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, int>(args, bDebugLevel, "neural network", "Oi");
    if (pNetwork == NULL) return NULL;
    pNetwork->SetDebugLevel(bDebugLevel);
    Py_RETURN_NONE;
}

/**
 * Get the checkpoint filename and interval from the source neural network. Note the capitalization difference between GetCheckPoint
 * and SetCheckpoint that is not present in NNNetworkAccessors::GetCheckpoint.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references a list that contains the checkpoint name string and interval integer
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNNetworkAccessors::GetCheckpoint(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    string filename;
    int32_t interval;
    tie(filename, interval) = pNetwork->GetCheckPoint();
    return Py_BuildValue("[si]", filename.c_str(), interval); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Set the checkpoint filename and interval in the destination neural network. Note the capitalization difference between SetCheckpoint
 * and GetCheckPoint that is not present in NNNetworkAccessors::GetCheckpoint.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param filename - the checkpoint filename string
 * @param interval - the checkpoint interval integer
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetCheckpoint(PyObject* self, PyObject* args) {
    char const* filename = NULL;
    int32_t interval = 0;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, char const*, int32_t>(args, filename, interval, "neural network", "Osi");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetCheckpoint(string(filename), interval));
}

/**
 * Get the local response network (LRN) k, n, alpha, and beta from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references a list of k (float), n (unsigned integer), alpha (float), and beta (float)
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNNetworkAccessors::GetLRN(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    NNFloat k, alpha, beta;
    uint32_t n;
    tie(k, n, alpha, beta) = pNetwork->GetLRN();
    return Py_BuildValue("[fIff]", k, n, alpha, beta); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Set the local response network (LRN) k, n, alpha, and beta in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param k - the LRN k float
 * @param n - the LRN n unsigned integer
 * @param alpha - the LRN alpha float
 * @param beta - the LRN beta float
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetLRN(PyObject* self, PyObject* args) {
    uint32_t n = 0;
    NNFloat k = 0.0, alpha = 0.0, beta = 0.0;
    NNNetwork* pNetwork = parsePtrAndFourValues<NNNetwork*, NNFloat, uint32_t, NNFloat, NNFloat>(args, k, n, alpha, beta, "neural network", "OfIff");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetLRN(k, n, alpha, beta));
}

/**
 * Get the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references a list of the oneTarget, zeroTarget, oneScale, and zeroScale floats
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNNetworkAccessors::GetSMCE(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    NNFloat oneTarget, zeroTarget, oneScale, zeroScale;
    tie(oneTarget, zeroTarget, oneScale, zeroScale) = pNetwork->GetSMCE();
    return Py_BuildValue("[ffff]", oneTarget, zeroTarget, oneScale, zeroScale); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Set the scaled marginal cross entropy (SMCE) oneTarget, zeroTarget, oneScale, and zeroScale in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param oneTarget - the SMCE oneTarget float
 * @param zeroTarget - the SMCE zeroTarget float
 * @param oneScale - the SMCE oneScale float
 * @param zeroScale - the SMCE zeroScale float
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetSMCE(PyObject* self, PyObject* args) {
    NNFloat oneTarget = 0.0, zeroTarget = 0.0, oneScale = 0.0, zeroScale = 0.0;
    NNNetwork* pNetwork = parsePtrAndFourValues<NNNetwork*, NNFloat, NNFloat, NNFloat, NNFloat>(args, oneTarget, zeroTarget, oneScale, zeroScale,
                                                                                                "neural network", "Offff");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetSMCE(oneTarget, zeroTarget, oneScale, zeroScale));
}

/**
 * Get the maxout k from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the maxout k unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetMaxout(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    uint32_t k;
    tie(k) = pNetwork->GetMaxout();
    return Py_BuildValue("I", k);
}

/**
 * Set the maxout k in the destination neural network.
 * @param pNetwork - the encapsulated destination NNNetwork*
 * @param maxoutK - the maxout k unsigned integer
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::SetMaxout(PyObject* self, PyObject* args) {
    uint32_t maxoutK = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, uint32_t>(args, maxoutK, "neural network", "OI");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("i", pNetwork->SetMaxout(maxoutK));
}

/**
 * Get the total examples from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the total examples unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetExamples(PyObject* self, PyObject* args) {
  NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("I", pNetwork->GetExamples());
}

/**
 * Get the weight connecting the specified input and output layers for the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param inputLayer - a string that specifies the input layer
 * @param outputLayer - a string that specifies the output layer
 * @return an encapsulated NNWeight* that references the weight connecting the two layers
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetWeight(PyObject* self, PyObject* args) {
    char const *inputLayer = NULL, *outputLayer = NULL;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(const_cast<NNWeight*>(pNetwork->GetWeight(string(inputLayer), string(outputLayer)))), "weight", NULL);
}

/**
 * Get the buffer size of the specified layer for the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param layer - a string that specifies the layer
 * @return an PyObject* that references the buffer size unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetBufferSize(PyObject* self, PyObject* args) {
    char const* layer = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, layer, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("K", pNetwork->GetBufferSize(string(layer))); // "K" format converts C unsigned long long (uint64_t) to a Python 3.0 object.
}

/**
 * Get the specified layer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param layer - a string that specifies the layer
 * @return an encapsulated NNLayer* that references the layer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetLayer(PyObject* self, PyObject* args) {
    char const* layer = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, layer, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(const_cast<NNLayer*>(pNetwork->GetLayer(string(layer)))), "layer", NULL);
}

/**
 * Get the list of layer names from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return an list of the layer name strings
 * @throws exception upon failure to parse arguments, or to create a new Python list, or to add a name to the list
 */
PyObject* NNNetworkAccessors::GetLayers(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    vector<string> layers = pNetwork->GetLayers();
    if (layers.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNNetworkAccessors::GetLayers received empty layers vector");
        return NULL;
    }
    // Convert the vector to a Python list.
    size_t n = layers.size();
    PyObject* list = PyList_New(n);
    if (list == NULL) {
        Py_XDECREF(list); // list is NULL, so Py_XDECREF is appropriate.
        // Failure of PyList_New appears not to raise an exception, so raise one.
        std::string message = "NNNetworkAccessors::GetLayers failed in PyList_New(" + std::to_string(n) + ")";
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
    for (unsigned int i = 0; i < n; i++) {
        if (PyList_SetItem(list, i, Py_BuildValue("s", layers[i].c_str())) < 0) {
            // Failure of PyList_SetItem appears not to raise an exception, so raise one.
            std::string message = "NNNetworkAccessors::GetLayers failed in PyList_SetItem for index = " + std::to_string(i);
            PyErr_SetString(PyExc_RuntimeError, message.c_str());
            goto fail;
        }
    }
    return list;
 fail:
    Py_DECREF(list); // list is not NULL, so Py_DECREF is appropriate.
    return NULL;
}

/**
 * Get the name of the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references the name string
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNNetworkAccessors::GetName(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return Py_BuildValue("s", pNetwork->GetName().c_str());
}

/**
 * Get the unit buffer for the specified layer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param layer - a string that specifies the layer
 * @return a PyObject* that references an NNFloat* that references the unit buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetUnitBuffer(PyObject* self, PyObject* args) {
    char const* layer = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, layer, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetUnitBuffer(string(layer))), "float", NULL);
}

/**
 * Get the delta buffer for the specified layer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param layer - a string that specifies the layer
 * @return a PyObject* that references an NNFloat* that references the delta buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetDeltaBuffer(PyObject* self, PyObject* args) {
    char const* layer = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, layer, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetDeltaBuffer(string(layer))), "float", NULL);
}

/**
 * Get the weight buffer for the specified input and output layers from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param inputLayer - a string that specifies the input layer
 * @param outputLayer - a string that specifies the output layer
 * @return a PyObject* that references an NNFloat* that references the weight buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetWeightBuffer(PyObject* self, PyObject* args) {
    char const *inputLayer = NULL, *outputLayer = NULL;
    NNNetwork* pNetwork = parsePtrAndTwoValues<NNNetwork*, char const*, char const*>(args, inputLayer, outputLayer, "neural network", "Oss");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetWeightBuffer(string(inputLayer), string(outputLayer))), "float", NULL);
}

/**
 * Get the current scratch buffer from the source neural network and resize it if necessary.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param size - an unsigned integer that specifies the current buffer size
 * @return a PyObject* that references an NNFloat* that references the possibly resized scratch buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetScratchBuffer(PyObject* self, PyObject* args) {
    size_t size = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, size_t>(args, size, "neural network", "OI");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetScratchBuffer(size)), "float", NULL);
}

/**
 * Get the current local send buffer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references an NNFloat* that references the current local send buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetP2PSendBuffer(PyObject* self, PyObject* args) {
  NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetP2PSendBuffer()), "float", NULL);
}

/**
 * Get the current local receive buffer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references an NNFloat* that references the current local receive buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetP2PReceiveBuffer(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetP2PReceiveBuffer()), "float", NULL);
}

/**
 * Get the system memory work buffer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references an NNFloat* that references the system memory work buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetP2PCPUBuffer(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetP2PCPUBuffer()), "float", NULL);
}

/**
 * Get the current adjacent peer receive buffer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references an NNFloat* that references the current adjacent peer receive buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetPeerBuffer(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetPeerBuffer()), "float", NULL);
}

/**
 * Get the current adjacent peer send buffer from the source neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @return a PyObject* that references an NNFloat* that references the current adjacent peer send buffer
 * @throws exception upon failure to parse arguments or to create a capsule
 */
PyObject* NNNetworkAccessors::GetPeerBackBuffer(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork->GetPeerBackBuffer()), "float", NULL);
}

/**
 * Set the clear velocity flag in the destination neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param bClearVelocity - the clear velocity flag integer
 * @throws exception upon failure to parse arguments
 */
PyObject* NNNetworkAccessors::SetClearVelocity(PyObject* self, PyObject* args) {
    int bClearVelocity = 0;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, int>(args, bClearVelocity, "neural network", "Oi");
    if (pNetwork == NULL) return NULL;
    pNetwork->SetClearVelocity(bClearVelocity);
    Py_RETURN_NONE;
}

/**
 * Set the training mode enumerator in the destination neural network.
 * @param pNetwork - the encapsulated source NNNetwork*
 * @param trainingMode - the training mode enumerator string
 * @throws exception upon failure to parse arguments or if the training mode enumerator string is unsupported
 */
PyObject* NNNetworkAccessors::SetTrainingMode(PyObject* self, PyObject* args) {
    char const* trainingMode = NULL;
    NNNetwork* pNetwork = parsePtrAndOneValue<NNNetwork*, char const*>(args, trainingMode, "neural network", "Os");
    if (pNetwork == NULL) return NULL;
    map<string, TrainingMode>::iterator it = stringToIntTrainingModeMap.find(string(trainingMode));
    if (it == stringToIntTrainingModeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "SetTrainingMode received unsupported training mode enumerator string");
        return NULL;
    }
    pNetwork->SetTrainingMode(it->second);
    Py_RETURN_NONE;
}

#endif
