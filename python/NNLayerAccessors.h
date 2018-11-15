/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __NNLAYERACCESSORS_H__
#define __NNLAYERACCESSORS_H__

class NNLayerAccessors {
    public:
        static PyObject* GetName(PyObject* self, PyObject* args);
        static PyObject* GetKind(PyObject* self, PyObject* args);
        static PyObject* GetType(PyObject* self, PyObject* args);
        static PyObject* GetAttributes(PyObject* self, PyObject* args);
        static PyObject* GetDataSet(PyObject* self, PyObject* args);
        static PyObject* GetNumDimensions(PyObject* self, PyObject* args);
        static PyObject* GetDimensions(PyObject* self, PyObject* args);
        static PyObject* GetLocalDimensions(PyObject* self, PyObject* args);
        static PyObject* GetKernelDimensions(PyObject* self, PyObject* args);
        static PyObject* GetKernelStride(PyObject* self, PyObject* args);
        static PyObject* GetUnits(PyObject* self, PyObject* args);
        static PyObject* SetUnits(PyObject* self, PyObject* args);
        static PyObject* GetDeltas(PyObject* self, PyObject* args);
        static PyObject* SetDeltas(PyObject* self, PyObject* args);
};

/**
 * Map from an NNLayer::Kind enumerator to a string.
 */
static map<NNLayer::Kind, string> intToStringKindMap = {
    {NNLayer::Kind::Input,  "Input"},
    {NNLayer::Kind::Hidden, "Hidden"},
    {NNLayer::Kind::Output, "Output"},
    {NNLayer::Kind::Target, "Target"}
};

/**
 * Map from an NNLayer::Type enumerator to a string.
 */
static map<NNLayer::Type, string> intToStringTypeMap = {
    {NNLayer::Type::FullyConnected, "FullyConnected"},
    {NNLayer::Type::Convolutional, "Convolutional"},
    {NNLayer::Type::Pooling,       "Pooling"},
};

/**
 * Get the name from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references the name string
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNLayerAccessors::GetName(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    return Py_BuildValue("s", pLayer->GetName().c_str());
}

/**
 * Get the kind enumerator from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references the kind enumerator string
 * @throws exception upon failure to parse arguments or to build the return value or if the kind enumerator is unsupported
 */
PyObject* NNLayerAccessors::GetKind(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    map<NNLayer::Kind, string>::iterator it = intToStringKindMap.find(pLayer->GetKind());
    if (it == intToStringKindMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "NNLayerAccessors::GetKind received unsupported kind enumerator");
        return NULL;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * Get the type enumerator from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references the type enumerator string
 * @throws exception upon failure to parse arguments or to build the return value if the type enumerator is unsupported
 */
PyObject* NNLayerAccessors::GetType(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    map<NNLayer::Type, string>::iterator it = intToStringTypeMap.find(pLayer->GetType());
    if (it == intToStringTypeMap.end()) {
        PyErr_SetString(PyExc_RuntimeError, "NNLayerAccessors::GetType received illegal type value");
        return NULL;
    }
    return Py_BuildValue("s", it->second.c_str());
}

/**
 * Get the attributes from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references the attributes unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNLayerAccessors::GetAttributes(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    return Py_BuildValue("I", pLayer->GetAttributes());
}

/**
 * Get the data set from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references an encapsulated NNDataSetBase* that references the data set
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNLayerAccessors::GetDataSet(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    return PyCapsule_New(reinterpret_cast<void*>(const_cast<NNDataSetBase*>(pLayer->GetDataSet())), "data set", NULL);
}

/**
 * Get the number of dimensions from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references the number of dimensions unsigned integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNLayerAccessors::GetNumDimensions(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    return Py_BuildValue("I", pLayer->GetNumDimensions());
}

/**
 * Get the dimensions from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references a list of four unsigned integers that represent the dimensions
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNLayerAccessors::GetDimensions(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    uint32_t Nx = 0, Ny = 0, Nz = 0, Nw = 0;
    tie(Nx, Ny, Nz, Nw) = pLayer->GetDimensions();
    return Py_BuildValue("[IIII]", Nx, Ny, Nz, Nw); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Get the local dimensions from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references a list of four unsigned integers that represent the local dimensions
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNLayerAccessors::GetLocalDimensions(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    uint32_t Nx = 0, Ny = 0, Nz = 0, Nw = 0;
    tie(Nx, Ny, Nz, Nw) = pLayer->GetLocalDimensions();
    return Py_BuildValue("[IIII]", Nx, Ny, Nz, Nw); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Get the kernel dimensions from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references a list of three unsigned integers that represent the kernel dimensions
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNLayerAccessors::GetKernelDimensions(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    uint32_t kernelX = 0, kernelY = 0, kernelZ = 0;
    tie(kernelX, kernelY, kernelZ) = pLayer->GetKernelDimensions();
    return Py_BuildValue("[III]", kernelX, kernelY, kernelZ); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Get kernel stride from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references a list of three unsigned integers that represent the kernel stride
 * @throws exception upon failure to parse arguments or to build the return list
 */
PyObject* NNLayerAccessors::GetKernelStride(PyObject* self, PyObject* args) {
    NNLayer* pLayer = parsePtr<NNLayer*>(args, "layer");
    if (pLayer == NULL) return NULL;
    uint32_t kernelStrideX = 0, kernelStrideY = 0, kernelStrideZ = 0;
    tie(kernelStrideX, kernelStrideY, kernelStrideZ) = pLayer->GetKernelStride();
    return Py_BuildValue("[III]", kernelStrideX, kernelStrideY, kernelStrideZ); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Modify the destination float32 NumPy array beginning at the specified offset by copying the units from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @param unitsArray - the destination float32 NumPy array
 * @param offset - the beginning offset
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value or in the case of an incompatible NumPy array or NULL array data
 */
PyObject* NNLayerAccessors::GetUnits(PyObject* self, PyObject* args) {
    PyArrayObject* unitsArray = NULL;
    uint32_t offset = 0;
    NNLayer* pLayer = parsePtrAndTwoValues<NNLayer*, PyArrayObject*, uint32_t>(args, unitsArray, offset, "layer", "OOI");
    if (pLayer == NULL) return NULL;
    if (CheckNumPyArray(unitsArray) == NULL) return NULL;
    // Get a pointer to the beginning of the data for the unitsArray NumPy array then add the offset and call pLayer->GetUnits() to modify the data.
    float* pUnits = static_cast<float*>(PyArray_DATA(unitsArray));
    if (pUnits  == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NNLayerAccessors::GetUnits received NULL NumPy array data pointer");
        return NULL;
    }
    pUnits += offset;
    return Py_BuildValue("i", pLayer->GetUnits(pUnits));
}

/**
 * Set the units of the destination layer by copying the units from a source float32 Numpy array.
 * @param pLayer - the encapsulated source NNLayer*
 * @param numpyArray - the source float32 NumPy array
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value or in the case of an incompatible NumPy array
 */
PyObject* NNLayerAccessors::SetUnits(PyObject* self, PyObject* args) {
    PyArrayObject* unitsArray = NULL;
    NNLayer* pLayer = parsePtrAndOneValue<NNLayer*, PyArrayObject*>(args, unitsArray, "layer", "OO");
    if (pLayer == NULL) return NULL;
    if (CheckNumPyArray(unitsArray) == NULL) return NULL;
    std::vector<float> units = NumPyArrayToVector(unitsArray);
    if (units.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNLayerAccessors::SetUnits received empty units vector");
        return NULL;
    }
    return Py_BuildValue("i", pLayer->SetUnits(units));
}

/**
 * Modify the destination float32 NumPy array beginning at a specified offset by copying the deltas from the source layer.
 * @param pLayer - the encapsulated source NNLayer*
 * @param deltasArray - the destination float32 NumPy array
 * @param offset - the beginning offset
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value or in the case of an incompatible NumPy array or NULL array data
 */
PyObject* NNLayerAccessors::GetDeltas(PyObject* self, PyObject* args) {
    PyArrayObject* deltasArray = NULL;
    uint32_t offset = 0;
    NNLayer* pLayer = parsePtrAndTwoValues<NNLayer*, PyArrayObject*, uint32_t>(args, deltasArray, offset, "layer", "OOI");
    if (pLayer == NULL) return NULL;
    if (CheckNumPyArray(deltasArray) == NULL) return NULL;
    // Get a pointer to the beginning of the data for the deltasArray NumPy array then add the offset and call pLayer->GetDeltas() to modify the data.
    float* pDeltas = static_cast<float*>(PyArray_DATA(deltasArray));
    if (pDeltas == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NNLayerAccessors::GetDeltas received NULL NumPy array data pointer");
        return NULL;
    }
    pDeltas += offset;
    return Py_BuildValue("i", pLayer->GetDeltas(pDeltas));
}

/**
 * Set the deltas of the destination layer by copying the deltas from a source float32 Numpy array.
 * @param pLayer - the encapsulated source NNLayer*
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value or in the case of an incompatible NumPy array
 */
PyObject* NNLayerAccessors::SetDeltas(PyObject* self, PyObject* args) {
    PyArrayObject* deltasArray = NULL;
    NNLayer* pLayer = parsePtrAndOneValue<NNLayer*, PyArrayObject*>(args, deltasArray, "layer", "OO");
    if (pLayer == NULL) return NULL;
    if (CheckNumPyArray(deltasArray) == NULL) return NULL;
    std::vector<float> deltas = NumPyArrayToVector(deltasArray);
    if (deltas.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "NNLayerAccessors::SetDeltas received empty deltas vector");
        return NULL;
    }
    return Py_BuildValue("i", pLayer->SetDeltas(deltas));
}

#endif
