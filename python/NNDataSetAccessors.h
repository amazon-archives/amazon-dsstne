/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __NNDATASETACCESSORS_H__
#define __NNDATASETACCESSORS_H__

class NNDataSetAccessors {
    public:
        static PyObject* GetDataSetName(PyObject* self, PyObject* args);
        static PyObject* CreateDenseDataSet(PyObject* self, PyObject* args);
        static PyObject* CreateSparseDataSet(PyObject* self, PyObject* args);
        static PyObject* SetStreaming(PyObject* self, PyObject* args);
        static PyObject* GetStreaming(PyObject* self, PyObject* args);
};

/**
 * Get the name of the source data set.
 * @param pDataSetBase - the encapsulated source NNDataSetBase*
 * @return a PyObject* that references the name string
 * @throws exception upon failure to parse arguments
 */
PyObject* NNDataSetAccessors::GetDataSetName(PyObject* self, PyObject* args) {
    NNDataSetBase* pDataSetBase = parsePtr<NNDataSetBase*>(args, "data set");
    if (pDataSetBase == NULL) return NULL;
    return Py_BuildValue("s", pDataSetBase->_name.c_str());
}

/**
 * Convert the data type of a NumPy array to an NNDataSetEnums::DataType.
 * @param numpyArray - a NumPy array
 * @return upon success, an NNDataSetEnums::DataType; upon failure, NNDataSetEnums::DataType::RGB8
 */
static NNDataSetEnums::DataType getDataType(PyArrayObject* numpyArray) {
    if (PyArray_ISUNSIGNED(numpyArray) && PyArray_ITEMSIZE(numpyArray) == 4) {
        return NNDataSetEnums::DataType::UInt;
    }
    if (PyArray_ISSIGNED(numpyArray) && PyArray_ITEMSIZE(numpyArray) == 4) {
        return NNDataSetEnums::DataType::Int;
    }
    if (PyArray_ISFLOAT(numpyArray) && PyArray_ITEMSIZE(numpyArray) == 4) {
        return NNDataSetEnums::DataType::Float;
    }
    if (PyArray_ISFLOAT(numpyArray) && PyArray_ITEMSIZE(numpyArray) == 8) {
        return NNDataSetEnums::DataType::Double;
    }
    if (PyArray_ISUNSIGNED(numpyArray) && PyArray_ITEMSIZE(numpyArray) == 1) {
        return NNDataSetEnums::DataType::UChar;
    }
    if (PyArray_ISSIGNED(numpyArray) && PyArray_ITEMSIZE(numpyArray) == 1) {
        return NNDataSetEnums::DataType::Char;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unsupported NumPy data type");
    return NNDataSetEnums::RGB8; // RGB8 is synonymous with UChar, so return it as an error code.
}

/**
 * Create an encapsulated data set from a a name string and a dense NumPy array.
 * @param name - the name string
 * @param numpyArray - the dense NumPy array
 * @return a PyObject* that references an encapsulated NNDataSetBase*
 * @throws exception upon failure to parse arguments or if the dense NumPy data type is unsupported by dsstne
 */
PyObject* NNDataSetAccessors::CreateDenseDataSet(PyObject* self, PyObject* args) {
    char const* name = NULL;
    PyArrayObject* numpyArray = NULL;
    if (!PyArg_ParseTuple(args, "sO", &name, &numpyArray)) return NULL;
    NNDataSetEnums::DataType dataType = getDataType(numpyArray);
    if (dataType == NNDataSetEnums::DataType::RGB8) { // As stated above in getDataType(), DataType::RGB8 is used as an error code.
        PyErr_SetString(PyExc_RuntimeError, "NNDataSetAccessors::CreateDenseDataSet received unsupported data type");
        return NULL;
    }
    // Create an instance of NNDataSetDimensions from the dimensions of the NumPy array; examples represents the trailing dimension.
    uint32_t width=1, height=1, length=1, examples=1;
    int ndim = PyArray_NDIM(numpyArray);
    if (ndim >= 1) {
        examples = PyArray_DIM(numpyArray, 0);
    }
    if (ndim >= 2) {
        width = PyArray_DIM(numpyArray, 0);
        examples = PyArray_DIM(numpyArray, 1);
    }
    if (ndim >= 3) {
        height = PyArray_DIM(numpyArray, 1);
        examples = PyArray_DIM(numpyArray, 2);
    }
    if (ndim >= 4) {
        length = PyArray_DIM(numpyArray, 2);
        examples = PyArray_DIM(numpyArray, 3);
    }
    NNDataSetDimensions dimensions = NNDataSetDimensions(width, height, length);
    // Create an instance of NNDataSetDescriptor.
    NNDataSetDescriptor descriptor;
    descriptor._name = std::string(name);
    descriptor._dataType = dataType;
    descriptor._attributes = 0; // NNDataSetEnums::Attributes value for a dense matrix
    descriptor._dim = dimensions;
    descriptor._examples = examples;
    descriptor._sparseDensity = NNFloat(1.0);
    // Create an instance of NNDataSet, copy the dense NumPy data into it, and return it.
    NNDataSetBase* pDataSet = createNNDataSet(descriptor);
    pDataSet->CopyDenseData(numpyArray);
    return PyCapsule_New(reinterpret_cast<void*>(pDataSet), "data set", NULL);
}

/**
 * Create an encapsulated data set from a name string and a compressed sparse row (CSR) SciPy two-dimensional matrix. See
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html for a description of the CSR matrix
 * attributes that are used as calling parameters to this CreateSparseDataSet function.
 * @param name - the name string
 * @param shape - a two-element tuple that represents the CSR shape attribute
 * @param data - a one-dimensional NumPy array that represents the CSR data attribute
 * @param indices - a one-dimensional NumPy array that represents the CSR indices attribute
 * @param indptr - a one-dimensional NumPy array that represents the CSR indptr attribute
 * @return a PyObject* that references an encapsulated NNDataSetBase*
 * @throws exception upon failure to parse arguments or the shape tuple of if the sparse NumPy data type is unsupported by dsstne
 */
PyObject* NNDataSetAccessors::CreateSparseDataSet(PyObject* self, PyObject* args) {
    char const* name = NULL;
    PyObject* shape = NULL;
    PyArrayObject *data = NULL, *indices = NULL, *indptr = NULL;
    if (!PyArg_ParseTuple(args, "sOOOO", &name, &shape, &data, &indices, &indptr)) return NULL;
    NNDataSetEnums::DataType dataType = getDataType(data);
    if (dataType == NNDataSetEnums::DataType::RGB8) { // As stated above in getDataType(), DataType::RGB8 is used as an error code.
        PyErr_SetString(PyExc_RuntimeError, "NNDataSetAccessors::CreateSparseDataSet received unsupported data type");
        return NULL;
    }
    // shape is a Python tuple, so it is parsed to obtain the number of rows and columns of the sparse matrix
    int rows=-1, cols=-1;
    if (!PyArg_ParseTuple(shape, "ii", &rows, &cols)) return NULL;
    // Whether a CSR or CSC sparse NumPy matrix, create an instance of NNDataSetDimension with cols as the width.
    NNDataSetDimensions dimensions = NNDataSetDimensions(cols, 1, 1);
    // Whether a CSR or CSC sparse NumPy matrix, create an instance of NNDataSetDescriptor with rows as _examples..
    NNDataSetDescriptor descriptor;
    descriptor._name = std::string(name);
    descriptor._dataType = dataType;
    descriptor._attributes = NNDataSetEnums::Attributes::Sparse;
    descriptor._dim = dimensions;
    descriptor._examples = rows;
    descriptor._sparseDensity = NNFloat(PyArray_SIZE(data)) / NNFloat(rows * cols);
    // Copy the indices array to make a uint32_t sparseIndex array.
    int* indicesData = static_cast<int*>(PyArray_DATA(indices));
    npy_intp indicesSize = PyArray_SIZE(indices);
    uint32_t sparseIndex[indicesSize];
    for (int i = 0; i < indicesSize; i++) {
        sparseIndex[i] = indicesData[i];
    }
    // Copy the indptr array to create uint64_t sparseStart and sparseEnd arrays.
    int* indptrData = static_cast<int*>(PyArray_DATA(indptr));
    npy_intp indptrSize = PyArray_SIZE(indptr);
    uint64_t sparseStart[indptrSize - 1];
    uint64_t sparseEnd[indptrSize - 1];
    for (int i = 0; i < indptrSize - 1; i++) {
        sparseStart[i] = indptrData[i]; // sparseStart comprises all but the last element of the indptr array
    }
    for (int i = 1; i < indptrSize; i++) {
        sparseEnd[i - 1] = indptrData[i]; // sparseEnd comprises all but the first element of the indptr array
    }
    // Create an instance of NNDataSet, copy the dense NumPy data into it, and return it.
    NNDataSetBase* pDataSet = createNNDataSet(descriptor);
    pDataSet->CopySparseData(sparseStart, sparseEnd, data, sparseIndex);
    return PyCapsule_New(reinterpret_cast<void*>(pDataSet), "data set", NULL);
}

/**
 * Set the streaming flag in the encapsulated data set.
 * @param pDataSet - the encapsulated destination NNDataSetBase*
 * @param streaming - the streaming integer
 * @return a PyObject* that references a boolean to indicate success
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNDataSetAccessors::SetStreaming(PyObject* self, PyObject* args) {
    int streaming = 0;
    NNDataSetBase* pDataSet = parsePtrAndOneValue<NNDataSetBase*, int>(args, streaming, "data set", "Oi");
    if (pDataSet == NULL) return NULL;
    return Py_BuildValue("i", pDataSet->SetStreaming(streaming)); // int to bool and bool to int conversions are implicit in C++
}

/**
 * Get the streaming flag from the encapsulated data set.
 * @param pDataSet - the encapsulated source NNDataSetBase*
 * @return a PyObject* that references the streaming integer
 * @throws exception upon failure to parse arguments or to build the return value
 */
PyObject* NNDataSetAccessors::GetStreaming(PyObject* self, PyObject* args) {
    NNDataSetBase* pDataSet = parsePtr<NNDataSetBase*>(args, "data set");
    if (pDataSet == NULL) return NULL;
    return Py_BuildValue("i", pDataSet->GetStreaming()); // bool to int conversion is implicit in C++
}

#endif
