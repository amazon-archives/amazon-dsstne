from distutils.core import setup, Extension
import os
import numpy as np

os.environ["CC"] = "mpiCC"
os.environ["LDSHARED"] = "mpiCC -shared -Wl,--no-undefined"
os.environ["LDSHARED"] = "mpiCC -shared"    # comment this line out to see what isn't linking correctly

module1 = Extension('dsstne',
                    include_dirs = [np.get_include(),
                                    '/usr/local/cuda/include',
                                    'B40C',
                                    'B40C/KernelCommon',
                                    '/usr/local/include',
                                    '/usr/lib/openmpi/include',
                                    '/usr/local/include',
                                    '/usr/include/jsoncpp',
                                    '../src/amazon/dsstne/utils',
                                    '../src/amazon/dsstne/engine',
                                    '/usr/include/cppunit'],
                    libraries = ['dsstne',
                                 'cudnn',
                                 'curand',
                                 'cublas',
                                 'cudart',
                                 'jsoncpp',
                                 'netcdf',
                                 'blas',
                                 'dl',
                                 'stdc++',
                                 'netcdf_c++4'],
                    runtime_library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                    library_dirs = [os.path.dirname(os.path.realpath(__file__)),
                                    '/usr/lib/atlas-base',
                                    '/usr/local/cuda/lib64',
                                    '/usr/local/lib',
                                    '../amazon-dsstne/lib'],
                    language='c++11',
                    extra_compile_args=['-std=c++11',
                                        '-DOMPI_SKIP_MPICXX'],
                    sources = ['dsstnemodule.cc',
                               '../src/amazon/dsstne/utils/cdl.cpp'])

setup (name = 'DSSTNE',
       version = '1.0',
       description = 'This is a package that links dsstne functions to Python.',
       ext_modules = [module1])
