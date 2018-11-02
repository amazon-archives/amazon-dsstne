TO BUILD the 'dsstne' module for import into Python:

sudo rm -rf build
sudo python setup.py install


TO TEST the 'dsstne' module via Python:

1. Download and install the CIFAR-10 image recognition test data into the "amazon-dsstne/samples/cifar-10/cifar-10-batches-bin"
   directory as explained in the comments to the "amazon-dsstne/samples/cifar-10/dparse.cpp" file.

2. Execute training followed by prediction for image recognition data via
   'cd images' followed by 'python images.py train.cdl' followed by 'python images.py predict.cdl'

Training creates the "images/checkpoint/check*.nc" files and the "images/results/network.nc" file that is subsequently used
for prediction. Training and prediction use the "images/config.json", "images/train.cdl", and "images/predict.cdl" files
that are copied from the "amazon-dsstne/samples/cifar-10" directory and modified to specify the following data files:
   amazon-dsstne/samples/cifar-10/cifar-10-batches-bin/cifar10_training.nc
   amazon-dsstne/samples/cifar-10/cifar-10-batches-bin/cifar10_test.nc

It is also possible to execute via MPI Python scripts that import the 'dsstne' module; for example:
   'mpiexec -np 1 python images.py predict.cdl' where a parallel dataset is necessary for np > 1.


TO IMPORT the 'dsstne' module into Python:

'import dsstne as dn' will import and initialize the 'dsstne' module BUT 'from mpi4py import MPI' must occur BEFORE
'import dsstne as dn' so that MPI will have a communicator.


CODE ARCHITECTURE:

The 'dsstne' module is a Python-C++ extension module that defines Python-C++ extension functions and is written according to
instructions at the following URL:

https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html

However, the above URL fails to explain that the 'import_array()' required subroutine MUST occur in the same compilation unit
as all other code in the extension module; otherwise, a segmentation fault will occur. For this reason, the source code of the
'dsstne' module comprises only one .cc file ("dsstnemodule.cc") that includes all other source code as .h files.  Also, it
appears that the '#include <Python.h>' must be the FIRST of all #include directives, so this directive occurs first in the
"dsstnemodule.h" file, for which the #include directive occurs first in the "dsstnemodule.cc" file.


PYTHON-C++ INTERFACE:

The Python-C++ interface allows bi-directional transfer of these primitive data types: string, bool, int, and float. Following
transfer from Python, these data types are parsed by C++ via the PyArg_ParseTuple function. They are encoded by C++ for transfer
to Python via the Py_BuildValue function.

Classes may not be transferred acrosss the Python-C++ interface unless a separate Python module in built via C++ for the
transfer, which is a somewhat cumbersome limitation. However, a void* pointer may be transferred from C++ to Python when
enclosed in a "capsule" that is built by C++ via the PyCapsule_New function. Python can't open the resulting capsule but
it can return the capsule to C++ as the argument to a Python-C++ extension function. Upon receipt of the capsule, C++ can
open it via the PyCapsule_GetPointer function. Dynamic type checking is possible because PyCapsule_New allows specification
of a string key that is included in the capsule and that is checked by PyCapsule_GetPointer.

The C++ PyArg_ParseTuple function also permits parsing of Python non-primitive data types such as NumPy arrays and Python
arrays. The ability to parse Python arrays permits the transfer of a vector of data sets (a vector<NNDataSetBase*>)
between Python and C++. Because a Python array is in fact a list, C++ builds a Python list from the vector<NNDataSetBase*>
as follows. The list is created via the PyList_New function and then each NNDataSetBase* in the vector is enclosed in a
capsule by the PyCapsule_New function and added to the list via the PyList_SetItem function. The list is returned to Python
and treated as an array. When the list is sent back to C++ as the argument to a Python-C++ extension function, C++ obtains
each capsule on the list via the PyList_GetItem function, opens the capsule to obtain the NNDataSetBase*, and appends the
NNDataSetBase* to a vector. The C++ code that implements this bi-directional transfer is found in the dsstnemodule.h file
as the DataSetBaseVectorToPythonList and PythonListToDataSetBaseVector functions.


NUMPY ARRAYS:

There are two types of NumPy arrays: dense and sparse. Both types of array may be transferred from Python to C++. C++ is
able to modify the contents of a dense NumPy array such that the modified contents are available to Python. Also, C++ is
able to construct a new dense NumPy array and return it to Python.

Accessing a dense NumPy array from C++ is straightforward. The PyArray_DIMS, PyArray_NDIM, and PyArray_SIZE macros return
the dimensions, the number of dimensions, and the number of elements for the array, respectively. The PyArray_DATA macro
returns a void* pointer to a contiguous array of the NumPy array's data and permits C++ to directly access those data. The
NdArrayToVector function of dsstnemodule.h is an example of access to the data of a dense NumPy array.

Accessing a sparse NumPy array from C++ is more complicated. The following URL discusses the format of a compressed
sparse row (CSR) two-dimensional matrix:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

Four attributes describe a CRS matrix: the shape, which is a 2-tuple that specifies the number of rows and columns, and
three one-dimensional arrays: data, indices, and indptr. Via these four attributes, C++ is able to access a CSR matrix.
The NNDataSetAccessors::CreateSparseDataSet function of NNDataSetAccessors.h is an example of CSR matrix access.


ENUMS:

It is not possible to transfer an element of an enum from C++ to Python. For this reason, C++ uses a static map to
convert an enumerator to a string that is returned to Python. Similarly, Python specifies an enumerator via a string
and then C++ uses a static map to convert the string to an enumerator. Examples of these static maps are found in
the dsstnemodule.h and NNLayerAccessors.h files.


PYTHON-C++ API DOCUMENTATION:

Each extension function provided by the Python-C++ API is described in the dsstnemodule.cc file. The arguments for
each extension function are passed via the 'args' parameter that is parsed by the PyArg_ParseTuple function to obtain
one or more arguments. For each extension function, the documentation included with the source code describes each
argument.
