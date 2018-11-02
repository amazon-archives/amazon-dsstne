/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __NNWEIGHTACCESSORS_H__
#define __NNWEIGHTACCESSORS_H__

class NNWeightAccessors {
    public:
        static PyObject* CopyWeights(PyObject* self, PyObject* args);
        static PyObject* SetWeights(PyObject* self, PyObject* args);
        static PyObject* SetBiases(PyObject* self, PyObject* args);
        static PyObject* GetWeights(PyObject* self, PyObject* args);
        static PyObject* GetBiases(PyObject* self, PyObject* args);
        static PyObject* SetNorm(PyObject* self, PyObject* args);
};

/**
 * Copy the weights from the specified source weight to the destination weight.
 * @param pWeight - an encapsulated pointer to the destination weight
 * @param pSrcWeight - an encapsulated pointer to the source weight
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse args or to open the capsules
 */
PyObject* NNWeightAccessors::CopyWeights(PyObject* self, PyObject* args) {
    PyObject* capsule = NULL;
    NNWeight* pWeight = parsePtrAndOneValue<NNWeight*, PyObject*>(args, capsule, "weight", "OO");
    if (pWeight == NULL) return NULL;
    NNWeight* pSrcWeight = reinterpret_cast<NNWeight*>(PyCapsule_GetPointer(capsule, "weight"));
    if (pSrcWeight == NULL) return NULL;
    return Py_BuildValue("i", pWeight->CopyWeights(pSrcWeight));
}

/**
 * Set the weights in the destination weight from a source NumPy array of weights.
 * @param pWeight - an encapsulated pointer to the destination weight
 * @param numpyArray - the source NumPy array of weights
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse args or in the case of an incompatible NumPy array
 */
PyObject* NNWeightAccessors::SetWeights(PyObject* self, PyObject* args) {
    PyArrayObject* numpyArray = NULL;
    NNWeight* pWeight = parsePtrAndOneValue<NNWeight*, PyArrayObject*>(args, numpyArray, "weight", "OO");
    if (pWeight == NULL) return NULL;
    if (CheckNumPyArray(numpyArray) == NULL) return NULL;
    std::vector<float> weights = NumPyArrayToVector(numpyArray);
    if (weights.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNWeightAccessors::SetWeights received empty weights vector");
        return NULL;
    }
    return Py_BuildValue("i", pWeight->SetWeights(weights));
}

/**
 * Set the biases in the destination weight from a source NumPy array of biases.
 * @param pWeight - an encapsulated pointer to the destination weight
 * @param numpyArray - the source NumPy array of biases
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse args or in the case of an incompatible NumPy array
 */
PyObject* NNWeightAccessors::SetBiases(PyObject* self, PyObject* args) {
    PyArrayObject* numpyArray = NULL;
    NNWeight* pWeight = parsePtrAndOneValue<NNWeight*, PyArrayObject*>(args, numpyArray, "weight", "OO");
    if (pWeight == NULL) return NULL;
    if (CheckNumPyArray(numpyArray) == NULL) return NULL;
    std::vector<float> biases = NumPyArrayToVector(numpyArray);
    if (biases.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNWeightAccessors::SetBiases received empty biases vector");
        return NULL;
    }
    return Py_BuildValue("i", pWeight->SetBiases(biases));
}

/**
 * Get the weights from the source weight.
 * @param pWeight - an encapsulated pointer to the source weight
 * @return a PyArrayObject* that references a NumPy array that contains the weights
 * @throws exception upon failure to parse args or if the dimensions of the weights are unobtainable
 */
PyObject* NNWeightAccessors::GetWeights(PyObject* self, PyObject* args) {
    std::vector<float> weights;
    NNWeight* pWeight = parsePtr<NNWeight*>(args, "weight");
    if (pWeight == NULL) return NULL;
    pWeight->GetWeights(weights); // The weights vector is passed by reference and modified
    std::vector<uint64_t> dimensions; // The dimensions vector is passed by reference and modified
    if (pWeight->GetDimensions(dimensions) == false) {
        PyErr_SetString(PyExc_RuntimeError, "GetWeights failed in NNWeight::GetDimensions");
        return NULL;
    }
    int nd = dimensions.size();
    npy_intp dims[nd];
    for (int i = 0; i < nd; i++) {
        dims[i] = dimensions.at(i);
    }
    PyArrayObject* numpyArray = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(nd, dims, NPY_FLOAT32));
    if (numpyArray == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NNWeightAccessors::GetWeights failed in PyArray_SimpleNew");
        // PyArray_SimpleNew creates a new reference; see https://partiallattice.wordpress.com/2013/03/09/a-stroll-down-optimization-lane/
        Py_XDECREF(numpyArray); // numpyArray is NULL, so Py_XDECREF is appropriate.
        return NULL;
    }
    memcpy(PyArray_DATA(numpyArray), weights.data(), weights.size() * sizeof(float));
    return reinterpret_cast<PyObject*>(numpyArray);
}

/**
 * Get the biases from the source weight.
 * @param pWeight - an encapsulated pointer to the source weight
 * @return a PyArrayObject* that references a NumPy array that contains the biases
 * @throws exception upon failure to parse args or if the dimensions of the biases are unobtainable
 */
PyObject* NNWeightAccessors::GetBiases(PyObject* self, PyObject* args) {
    std::vector<float> biases;
    NNWeight* pWeight = parsePtr<NNWeight*>(args, "weight");
    if (pWeight == NULL) return NULL;
    pWeight->GetBiases(biases); // The biases vector is passed by reference and modified
    std::vector<uint64_t> dimensions; // The dimensions vector is passed by reference and modified
    if (pWeight->GetDimensions(dimensions) == false) {
        PyErr_SetString(PyExc_RuntimeError, "GetBiases failed in NNWeight::GetDimensions");
        return NULL;
    }
    int nd = dimensions.size();
    npy_intp dims[nd];
    for (int i = 0; i < nd; i++) {
        dims[i] = dimensions.at(i);
    }
    PyArrayObject* numpyArray = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(nd, dims, NPY_FLOAT32));
    if (numpyArray == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NNWeightAccessors::GetBiases failed in PyArray_SimpleNew");
        // PyArray_SimpleNew creates a new reference; see https://partiallattice.wordpress.com/2013/03/09/a-stroll-down-optimization-lane/
        Py_XDECREF(numpyArray); // numpyArray is NULL, so Py_XDECREF is appropriate.
        return NULL;
    }
    memcpy(PyArray_DATA(numpyArray), biases.data(), biases.size() * sizeof(float));
    return reinterpret_cast<PyObject*>(numpyArray);
}

/**
 * Set norm in the destination weight.
 * @param pWeight - an encapsulated pointer to the destination weight
 * @param norm - a float that specifies the norm
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse args
 */
PyObject* NNWeightAccessors::SetNorm(PyObject* self, PyObject* args) {
    float norm = 0.0;
    NNWeight* pWeight = parsePtrAndOneValue<NNWeight*, float>(args, norm, "weight", "Of");
    if (pWeight == NULL) return NULL;
    return Py_BuildValue("i", pWeight->SetNorm(norm));
}

#endif
