#
# For debugging add '-DMEMTRACKING -gdwarf-3' to CPPFLAGS and CFLAGS
#
CPP = /lib/cpp
#CPPFLAGS = -traditional -P -std=c++0x
CC = mpiCC

# switch to 1 for debugging
DEBUG ?= 0
ifeq ($(DEBUG), 1)
  $(info ************  DEBUG mode ************)
  CFLAGS = -DOMPI_SKIP_MPICXX -std=c++0x -g -O0
else
  $(info ************  RELEASE mode ************)
  CFLAGS = -DOMPI_SKIP_MPICXX -std=c++0x -O3
endif

NVCC = nvcc
CU_FLAGS = -use_fast_math --ptxas-options="-v" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_30,code=sm_30 -DOMPI_SKIP_MPICXX -std=c++11
CU_INCLUDES = -I/usr/local/cuda/include -IB40C -IB40C/KernelCommon -I/usr/local/include -I/usr/local/openmpi/include -I/usr/include/jsoncpp -I../utils -I../engine
CU_LIBS = -L/usr/lib/atlas-base -L/usr/local/cuda/lib64 -L. -L/usr/local/lib/
CU_LOADLIBS = -lcurand -lcublas -lcudart -lmpi -lmpi_cxx -ljsoncpp -lnetcdf_c++4 -lnetcdf -l:libcblas.a -l:libatlas.a -ldl -lstdc++
LOAD = mpiCC

