/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __DSSTNEMODULE_H__
#define __DSSTNEMODULE_H__

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string>

#include "GpuTypes.h"
#include "NNTypes.h"
#include "cdl.h"

using std::map;
using std::string;

/**
 * Map from a string to a Mode enumerator.
 */
static map<string, Mode> stringToIntModeMap = {
    {"Prediction",  Mode::Prediction},
    {"Training",    Mode::Training},
    {"Validation",  Mode::Validation},
    {"Unspecified", Mode::Unspecified}
};

/**
 * Map from a Mode enumerator to a string.
 */
static map<Mode, string> intToStringModeMap = {
    {Mode::Prediction,  "Prediction"},
    {Mode::Training,    "Training"},
    {Mode::Validation,  "Validation"},
    {Mode::Unspecified, "Unspecified"}
};

/**
 * Map from a string to a TrainingMode enumerator.
 */
static map<string, TrainingMode> stringToIntTrainingModeMap = {
    {"SGD",      TrainingMode::SGD},
    {"Momentum", TrainingMode::Momentum},
    {"AdaGrad",  TrainingMode::AdaGrad},
    {"Nesterov", TrainingMode::Nesterov},
    {"RMSProp",  TrainingMode::RMSProp},
    {"AdaDelta", TrainingMode::AdaDelta},
    {"Adam",     TrainingMode::Adam}
};

/**
 * Map from a TrainingMode enumerator to a string.
 */
static map<TrainingMode, string> intToStringTrainingModeMap = {
    {TrainingMode::SGD,      "SGD"},
    {TrainingMode::Momentum, "Momentum"},
    {TrainingMode::AdaGrad,  "AdaGrad"},
    {TrainingMode::Nesterov, "Nesterov"},
    {TrainingMode::RMSProp,  "RMSProp"},
    {TrainingMode::AdaDelta, "AdaDelta"},
    {TrainingMode::Adam,     "Adam"}
};

/**
 * Parse an encapsulated pointer of type T*.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T>
static T const parsePtr(PyObject* args, char const* key) {
    PyObject* capsule = NULL;
    // Failure of PyArg_ParseTuple() requires neither raising an exception nor calling Py_XDECREF()
    // as per the example at https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        // Failure of PyCapsule_GetPointer() raises an exception so no need to raise another one.
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtr invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Parse an encapsulated pointer of type T* and one associated value of type V.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T, typename V>
static T const parsePtrAndOneValue(PyObject* args, V& value, char const* key, char const* format) {
    PyObject* capsule = NULL;
    if (!PyArg_ParseTuple(args, format, &capsule, &value)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtrAndOneValue invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Parse an encapsulated pointer of type T* and two associated values of types V and W.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T, typename V, typename W>
    static T const parsePtrAndTwoValues(PyObject* args, V& value1, W& value2, char const* key, char const* format) {
    PyObject* capsule = NULL;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtrAndTwoValues invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Parse an encapsulated pointer of type T* and three associated values of types V, W, and X.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T, typename V, typename W, typename X>
    static T const parsePtrAndThreeValues(PyObject* args, V& value1, W& value2, X& value3, char const* key, char const* format) {
    PyObject* capsule = NULL;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtrAndThreeValues invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Parse an encapsulated pointer of type T* and four associated values of types V, W, X, and Y.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T, typename V, typename W, typename X, typename Y>
    static T const parsePtrAndFourValues(PyObject* args, V& value1, W& value2, X& value3, Y& value4, char const* key, char const* format) {
    PyObject* capsule = NULL;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3, &value4)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtrAndFourValues invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Parse an encapsulated pointer of type T* and five associated values of types V, W, X, Y, and Z.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T, typename V, typename W, typename X, typename Y, typename Z>
    static T const parsePtrAndFiveValues(PyObject* args, V& value1, W& value2, X& value3, Y& value4, Z& value5, char const* key, char const* format) {
    PyObject* capsule = NULL;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3, &value4, &value5)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtrAndFiveValues invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Parse an encapsulated pointer of type T* and six associated values of types U, V, W, X, Y, and Z.
 * @param args - a PyObject* that references the encapsulated pointer
 * @return a pointer of type T*
 * @throws exception upon failure to parse or the encapsulated pointer is the wrong type
 */
template <typename T, typename U, typename V, typename W, typename X, typename Y, typename Z>
    static T const parsePtrAndSixValues(PyObject* args, U& value1, V& value2, W& value3, X& value4, Y& value5, Z& value6,
                                        char const* key, char const* format) {
    PyObject* capsule = NULL;
    if (!PyArg_ParseTuple(args, format, &capsule, &value1, &value2, &value3, &value4, &value5, &value6)) return NULL;
    if (PyCapsule_IsValid(capsule, key) != 0) {
        T const ptr = reinterpret_cast<T const>(PyCapsule_GetPointer(capsule, key));
        if (ptr == NULL) return NULL;
        return ptr;
    } else {
        std::string message = "parsePtrAndFiveValues invalid capsule: name = " + string(PyCapsule_GetName(capsule)) + "  key = " + string(key);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
}

/**
 * Check that a NumPy array is one-dimensional and contains 32-bit floating-point data.
 * @param numpyArray - a PyArrayObect* that references a NumPy array
 * @return the input NumPy array as a dummy normal return
 * @throws exception if the NumPy array does not contain 32-bit floating-point data
 */
static PyArrayObject* CheckNumPyArray(PyArrayObject* numpyArray) {
    if (!PyArray_ISFLOAT(numpyArray) || PyArray_ITEMSIZE(numpyArray) != 4) {
        PyErr_SetString(PyExc_RuntimeError, "CheckNumPyArray received incorrect NumPy array type; expected float32");
        return NULL;
    }
    return numpyArray; // Dummy return that isn't used by the caller
}

/**
 * Convert a NumPy array that contains 32-bit floating-point data to a vector.
 * @param numpyArray - a PyArrayObject* that references a NumPy array
 * @return a vector<float> into which the NumPy array data have been copied; upon error, an empty vector<float>
 */
static vector<float> NumPyArrayToVector(PyArrayObject* numpyArray) {
    npy_intp n = PyArray_SIZE(numpyArray);
    std::vector<float> v(n);
    void* data = PyArray_DATA(numpyArray);
    if (data == NULL) return v;
    memcpy(v.data(), data, n * sizeof(float));
    return v;
}

/**
 * Convert a Python array (i.e. a list) to a vector<NNDataSetBase*>
 * @param list - a PyObject* that references a Python list wherein each element is an encapsulated NNDataSetBase*
 * @return upon success, a vector<NNDataSetBase*>; upon error, an empty vector
 */
static vector<NNDataSetBase*> PythonListToDataSetBaseVector(PyObject* list) {
    Py_ssize_t size = PyList_Size(list);
    std::vector<NNDataSetBase*> vect(size);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* capsule = PyList_GetItem(list, i);
        if (capsule == NULL) goto fail;
        NNDataSetBase* pDataSetBase = reinterpret_cast<NNDataSetBase*>(PyCapsule_GetPointer(capsule, "data set"));
        if (pDataSetBase == NULL) goto fail;
        vect.at(i) = pDataSetBase;
    }
    return vect;
 fail:
    vect.clear();
    return vect;
}

/**
 * Convert a vector<NNDataSetBase*> to a Python array (i.e. a list).
 * @param vDataSetBase - a vector<NNDataSetBase*> passed by reference
 * @return a PyObject* that references a list wherein each element is an encapsulated NNDataSetBase*
 * @throws exception upon failure to create a new Python list, to encapsulate an NNDataSetBase*, or to add the capsule to the list
 */
static PyObject* DataSetBaseVectorToPythonList(vector<NNDataSetBase*>& vDataSetBase) {
    Py_ssize_t size = vDataSetBase.size();
    PyObject* list = PyList_New(size); // Success creates a new reference that must be DECREF'ed in case of subsequent failure.
    if (list == NULL) {
        Py_XDECREF(list); // list is NULL, so Py_XDECREF is appropriate.
        // Failure of PyList_New appears not to raise an exception, so raise one.
        std::string message = "DataSetVectorToPythonArray failed in PyList_New(" + std::to_string(size) + ")";
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* pDataSetBase = PyCapsule_New(reinterpret_cast<void*>(vDataSetBase.at(i)), "data set", NULL);
        if (PyList_SetItem(list, i, pDataSetBase) < 0) {
            // Failure of PyList_SetItem appears not to raise an exception, so raise one.
            std::string message = "DataSetVectorToPythonArray failed in PyList_SetItem for index = " + std::to_string(i);
            PyErr_SetString(PyExc_RuntimeError, message.c_str());
            goto fail;
        }
    }
    return list;
 fail:
    Py_DECREF(list); // list is not NULL, so Py_DECREF is appropriate.
    return NULL;
}

#endif
