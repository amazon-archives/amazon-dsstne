/*
   Copyright 2018  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef __UTILITIES_H__
#define __UTILITIES_H__

class Utilities {
    public:
        static PyObject* Startup(PyObject* self, PyObject* args);
        static PyObject* Shutdown(PyObject* self, PyObject* args);
        static PyObject* CreateCDLFromJSON(PyObject* self, PyObject* args);
        static PyObject* CreateCDLFromDefaults(PyObject* self, PyObject* args);
        static PyObject* DeleteCDL(PyObject* self, PyObject* args);
        static PyObject* LoadDataSetFromNetCDF(PyObject* self, PyObject* args);
        static PyObject* DeleteDataSet(PyObject* self, PyObject* args);
        static PyObject* LoadNeuralNetworkFromNetCDF(PyObject* self, PyObject* args);
        static PyObject* LoadNeuralNetworkFromJSON(PyObject* self, PyObject* args);
        static PyObject* DeleteNNNetwork(PyObject* self, PyObject* args);
        static PyObject* OpenFile(PyObject* self, PyObject* args);
        static PyObject* CloseFile(PyObject* self, PyObject* args);
        static PyObject* SetRandomSeed(PyObject* self, PyObject* args);
        static PyObject* GetMemoryUsage(PyObject* self, PyObject* args);
        static PyObject* Normalize(PyObject* self, PyObject* args);
        static PyObject* Transpose(PyObject* self, PyObject* args);
        static PyObject* CreateFloatGpuBuffer(PyObject* self, PyObject* args);
        static PyObject* DeleteFloatGpuBuffer(PyObject* self, PyObject* args);
        static PyObject* CreateUnsignedGpuBuffer(PyObject* self, PyObject* args);
        static PyObject* DeleteUnsignedGpuBuffer(PyObject* self, PyObject* args);
};

/**
 * Initialize the GPUs and MPI.
 * @param arglist - a command-line argument list
 * @throws exception upon failure to parse the 'args' arguments or failure of malloc
 */
PyObject* Utilities::Startup(PyObject* self, PyObject* args) {
    PyObject* arglist = NULL;
    // Failure of PyArg_ParseTuple() requires neither raising an exception or calling Py_XDECREF()
    // See the example at https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
    if (!PyArg_ParseTuple(args, "O", &arglist)) return NULL;
    // Convert the list of Python bytes objects to an array of C character arrays.
    int argc = PyList_Size(arglist);
    char** argv = reinterpret_cast<char**>(malloc(argc * sizeof(char*)));
    if (argv == NULL) {
      PyErr_NoMemory(); // See section 1.4 of https://docs.python.org/3/extending/extending.html
      return NULL;
    }
    for (int i = 0; i < argc; i++) {
      argv[i] = PyBytes_AsString(PyList_GetItem(arglist, i));
      if (argv[i] == NULL) return NULL; // Failure of PyBytes_AsString() sets an exception, so no need to raise another one.
    }
    // Pass the command-line arguments to Startup(). However, Startup() does not in fact call MPI_Init() because
    // MPI_Init() has been called when Python imported mpi4py and hopefully passed command-line arguments to MPI_Init().
    // In fact, since MPI v1.1 it is not necessary to supply command-line arguments to MPI_Init() other than
    // to override those arguments.
    getGpu().Startup(argc, argv);
    free(argv);
    Py_RETURN_NONE; // This macro is for a void C function and is equivalent to 'Py_INCREF(Py_None); return Py_None;'
}

/**
 * Shutdown the GPUs.
 */
PyObject* Utilities::Shutdown(PyObject* self, PyObject* args) {
    getGpu().Shutdown();
    Py_RETURN_NONE; // This macro is for a void C function and is equivalent to 'Py_INCREF(Py_None); return Py_None;'
}

/**
 * Create a CDL instance and initialize it from a JSON file.
 * @param jsonFilename - the JSON filename string
 * @return a PyObject* that references an encapsulated CDL*
 * @throws exception upon failure to parse arguments, to create a CDL instance, to parse the JSON file, or to create a capsule
 */
PyObject* Utilities::CreateCDLFromJSON(PyObject* self, PyObject* args) {
    char const* jsonFilename = NULL;
    if (!PyArg_ParseTuple(args, "s", &jsonFilename)) return NULL; // "s" format can parse a Python string object to a C character array
    CDL* pCDL = new CDL();
    if (pCDL == NULL) {
        PyErr_NoMemory(); // See section 1.4 of https://docs.python.org/3/extending/extending.html
        return NULL;
    }
    if (pCDL->Load_JSON(string(jsonFilename)) != 0) {
        std::string message = "Load_JSON could not parse JSON file: " + string(jsonFilename);
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return NULL;
    }
    // Failure of PyCapsule_New() sets an exception and returns NULL so no need for further error handling.
    return PyCapsule_New(reinterpret_cast<void*>(pCDL), "cdl", NULL);
}

/**
 * Create a CDL instance and initialize it with default values.
 * @return a PyObject* that references an encapsulated CDL*
 * @throws exception upon failure to create a CDL instance or to create a capsule
 */
PyObject* Utilities::CreateCDLFromDefaults(PyObject* self, PyObject* args) {
    CDL* pCDL = new CDL();
    if (pCDL == NULL) {
        PyErr_NoMemory(); // See section 1.4 of https://docs.python.org/3/extending/extending.html
        return NULL;
    }
    return PyCapsule_New(reinterpret_cast<void*>(pCDL), "cdl", NULL);
}

/**
 * Delete a CDL instance.
 * @param pCDL - an encapsulated CDL*
 * @throws exception upon failure to parse arguments
 */
PyObject* Utilities::DeleteCDL(PyObject* self, PyObject* args) {
    CDL* pCDL = parsePtr<CDL*>(args, "cdl");
    if (pCDL == NULL) return NULL;
    delete pCDL;
    Py_RETURN_NONE;
}
  
/**
 * Load a Python list of data sets from a CDF file.
 * @param dataFilename - the CDF filename
 * @return a PyObject* that references a list of encapsulated NNDataSetBase*
 * @throws exception upon failure to parse arguments, to parse the CDF file, to create the list, or to build the return value
 */
PyObject* Utilities::LoadDataSetFromNetCDF(PyObject* self, PyObject* args) {
    char const* dataFilename = NULL;
    if (!PyArg_ParseTuple(args, "s", &dataFilename)) return NULL;
    vector<NNDataSetBase*> vDataSetBase = LoadNetCDF(string(dataFilename));
    // Failure of LoadNetCDF() causes exit(-1) but check for an error anyway.
    if (vDataSetBase.empty()) {
        PyErr_NoMemory(); // See section 1.4 of https://docs.python.org/3/extending/extending.html
        return NULL;
    }
    return DataSetBaseVectorToPythonList(vDataSetBase);
}

/**
 * Delete a data set.
 * @param pDataSet - an encapsulated NNDataSetBase*
 * @throws exception upon failure to parse arguments
 */
PyObject* Utilities::DeleteDataSet(PyObject* self, PyObject* args) {
    NNDataSetBase* pDataSet = parsePtr<NNDataSetBase*>(args, "data set");
    if (pDataSet == NULL) return NULL;
    delete pDataSet;
    Py_RETURN_NONE;
}

/**
 * Load a neural network from a CDF file and a batch number.
 * @param networkFilename - the CDF filename string
 * @param batch - the batch unsigned integer
 * @return a PyObject* that references an encapsulated NNNetwork*
 * @throws exception upon failure to parse arguments, to parse the CDF file, or to create a capsule
 */
PyObject* Utilities::LoadNeuralNetworkFromNetCDF(PyObject* self, PyObject* args) {
    char const *networkFilename = NULL;
    uint32_t batch = 0;
    if (!PyArg_ParseTuple(args, "sI", &networkFilename, &batch)) return NULL; // "I" format converts a Python object to a C unsigned int
    NNNetwork* pNetwork = LoadNeuralNetworkNetCDF(string(networkFilename), batch);
    if (pNetwork == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Utilities::LoadNeuralNetworkFromNetCDF received NULL result from LoadNeuralNetworkNetCDF");
        return NULL;
    }
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork), "neural network", NULL);
}

/**
 * Load a neural network from a JSON config file, a batch number, and Python list of data sets.
 * @param jsonFilename - the JSON config filename string
 * @param batch - the batch unsigned integer
 * @param pDataSetBaseList - a list of encapsulated NNDataSetBase*
 * @return a PyObject* that references an encapsulated NNNetwork*
 * @throws exception upon failure to parse arguments, to parse the JSON config file, or to create a capsule
 */
PyObject* Utilities::LoadNeuralNetworkFromJSON(PyObject* self, PyObject* args) {
    char const* jsonFilename = NULL;
    uint32_t batch;
    PyObject* pDataSetBaseList = NULL;
    if (!PyArg_ParseTuple(args, "sIO", &jsonFilename, &batch, &pDataSetBaseList)) return NULL;
    vector<NNDataSetBase*> vDataSetBase = PythonListToDataSetBaseVector(pDataSetBaseList);
    if (vDataSetBase.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "Utilities::LoadNeuralNetworkFromJSON received empty vDataSetBase");
        return NULL;
    }
    NNNetwork* pNetwork = LoadNeuralNetworkJSON(string(jsonFilename), batch, vDataSetBase);
    if (pNetwork == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Utilities::LoadNeuralNetworkFromJSON received NULL result from LoadNeuralNetworkNetCDF");
        return NULL;
    }
    return PyCapsule_New(reinterpret_cast<void*>(pNetwork), "neural network", NULL);
}

/**
 * Delete a neural network.
 * @param pNetwork - an encapsulated NNNetwork*
 * @throws exception upon failure to parse arguments
 */
PyObject* Utilities::DeleteNNNetwork(PyObject* self, PyObject* args) {
    NNNetwork* pNetwork = parsePtr<NNNetwork*>(args, "neural network");
    if (pNetwork == NULL) return NULL;
    delete pNetwork;
    Py_RETURN_NONE;
}
  
/**
 * Open a FILE* stream.
 * @param filename - a filename string
 * @param mode - a file open mode string
 * @return an PyObject* that references an encapsulated FILE*
 * @throws exception upon failure to parse arguments, to open the file, or to create a capsule
 */
PyObject* Utilities::OpenFile(PyObject* self, PyObject* args) {
    char const *filename = NULL, *mode = NULL;
    if (!PyArg_ParseTuple(args, "ss", &filename, &mode)) return NULL;
    FILE* pFILE = fopen(filename, mode);
    if (pFILE == NULL) {
        PyErr_SetString(PyExc_IOError, "File open error");
        return NULL;
    }
    return PyCapsule_New(reinterpret_cast<void*>(pFILE), "file", NULL);
}

/**
 * Close a FILE* stream.
 * @param pFile - an encapsulated FILE*
 * @throws exception upon failure to parse arguments or to close the file
 */
PyObject* Utilities::CloseFile(PyObject* self, PyObject* args) {
    FILE* pFILE = parsePtr<FILE*>(args, "file");
    if (pFILE == NULL) return NULL;
    if (fclose(pFILE) != 0) {
        PyErr_SetString(PyExc_IOError, "File close error");
        return NULL;
    }
    Py_RETURN_NONE;
}
  
/**
 * Set random seed in the GPU.
 * @param randomSeed - the randomSeed unsigned integer
 * @throws exception upon failure to parse arguments
 */
PyObject* Utilities::SetRandomSeed(PyObject* self, PyObject* args) {
    unsigned long randomSeed = 0;
    if (!PyArg_ParseTuple(args, "I", &randomSeed)) return NULL; // "I" format converts a Python 3.0 object to a C unsigned int
    getGpu().SetRandomSeed(randomSeed);
    Py_RETURN_NONE;
}

/**
 * Get the GPU and CPU memory usage.
 * @return a PyObject* that references list of two integers; the first integer is the GPU memory usage and the second integer is the CPU memory usage
 */
PyObject* Utilities::GetMemoryUsage(PyObject* self, PyObject* args) {
    int gpuMemoryUsage, cpuMemoryUsage;
    getGpu().GetMemoryUsage(&gpuMemoryUsage, &cpuMemoryUsage);
    return Py_BuildValue("[ii]", gpuMemoryUsage, cpuMemoryUsage); // Building a Python list via [...] isn't strictly necessary.
}

/**
 * Transpose a NumPy 2D matrix to create a contiguous matrix that is returned through the output NumPy matrix.
 * @param ASINWeightArray - the output NumPy 2D matrix that is modified by this Transpose function
 * @param pEmbeddingArray - the input NumPy 2D matrix
 * @throws exception upon failure to parse arguments or if the matrices are incompatible as determined by dsstnecalculate_Transpose
 */
PyObject* Utilities::Transpose(PyObject* self, PyObject* args) {
    PyArrayObject *ASINWeightArray = NULL, *pEmbeddingArray = NULL;
    if (!PyArg_ParseTuple(args, "OO", &ASINWeightArray, &pEmbeddingArray)) return NULL;
    return dsstnecalculate_Transpose(ASINWeightArray, pEmbeddingArray);
}

/**
 * Create a GPU Buffer of type NNFloat and of the specified size.
 * @param size - the size unsigned integer
 * @return a PyObject* that references an encapsulated GpuBuffer<NNFloat>*
 * @throws exception upon failure to parse arguments or to create the GPU buffer
 */
PyObject* Utilities::CreateFloatGpuBuffer(PyObject* self, PyObject* args) {
    uint32_t size = 0;
    if (!PyArg_ParseTuple(args, "I", &size)) return NULL;
    GpuBuffer<NNFloat>* pGpuBuffer = new GpuBuffer<NNFloat>(size, true);
    if (pGpuBuffer == NULL) {
        PyErr_NoMemory(); // See section 1.4 of https://docs.python.org/3/extending/extending.html
        return NULL;
    }
    return PyCapsule_New(reinterpret_cast<void*>(pGpuBuffer), "float gpu buffer", NULL);
}
  
/**
 * Delete a GPU Buffer of type NNFloat.
 * @param pGpuBuffer - an encapsulated GpuBuffer<NNFloat>*
 * @throws exception upon failure to parse arguments
 */
PyObject* Utilities::DeleteFloatGpuBuffer(PyObject* self, PyObject* args) {
    GpuBuffer<NNFloat>* pGpuBuffer = parsePtr<GpuBuffer<NNFloat>*>(args, "float gpu buffer");
    if (pGpuBuffer == NULL) return NULL;
    delete pGpuBuffer;
    Py_RETURN_NONE;
}
  
/**
 * Create a GPU Buffer of type uint32_t and of the specified size.
 * @param size - the size unsigned integer
 * @return a PyObject* that references an encapsulated GpuBuffer<uint32_t>*
 * @throws exception upon failure to parse arguments or to create the GPU buffer
 */
PyObject* Utilities::CreateUnsignedGpuBuffer(PyObject* self, PyObject* args) {
    uint32_t size = 0;
    if (!PyArg_ParseTuple(args, "I", &size)) return NULL;
    GpuBuffer<uint32_t>* pGpuBuffer = new GpuBuffer<uint32_t>(size, true);
    if (pGpuBuffer == NULL) {
        PyErr_NoMemory(); // See section 1.4 of https://docs.python.org/3/extending/extending.html
        return NULL;
    }
    return PyCapsule_New(reinterpret_cast<void*>(pGpuBuffer), "unsigned gpu buffer", NULL);
}
  
/**
 * Delete a GPU Buffer of type uint32_t.
 * @param pGpuBuffer - a PyObject* that references an encapsulated GpuBuffer<uint32_t>*
 * @throws exception upon failure to parse arguments
 */
PyObject* Utilities::DeleteUnsignedGpuBuffer(PyObject* self, PyObject* args) {
    GpuBuffer<uint32_t>* pGpuBuffer = parsePtr<GpuBuffer<uint32_t>*>(args, "unsigned gpu buffer");
    if (pGpuBuffer == NULL) return NULL;
    delete pGpuBuffer;
    Py_RETURN_NONE;
}

#endif
