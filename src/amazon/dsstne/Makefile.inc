#
# For debugging add '-DMEMTRACKING -gdwarf-3' to CPPFLAGS and CFLAGS
#
CPP ?= /lib/cpp
#CPPFLAGS = -traditional -P -std=c++0x
CC = mpiCC
NVCC = nvcc
LOAD = mpiCC

# Build directory
BUILD_DIR ?= $(dir $(shell pwd))../../../build

#
# Used to exclude cuda headers from auto dependency generation
# For some odd reason nvcc includes the directory relative to the bin directory
# rather than the usual /usr/local/cuda/include
#
CUDA_SYSTEM_INCLUDE_DIR := $(dir $(shell which $(NVCC)))../target/x86_64/include

# switch to 1 for debugging
DEBUG ?= 0
ifeq ($(DEBUG), 1)
  $(info ************  DEBUG mode ************)
  CFLAGS = \
      -g \
      -O0 \
      -Wall \
      -std=c++11 \
      -fPIC \
      -DOMPI_SKIP_MPICXX \
      -MMD \
      -MP

  CU_FLAGS = \
      -g \
      -O0 \
      --device-debug \
      --generate-line-info \
      -std=c++11 \
      --compiler-options=-fPIC \
      --compiler-options=-Wall \
      -use_fast_math \
      --ptxas-options="-v" \
      -gencode arch=compute_70,code=sm_70 \
      -gencode arch=compute_61,code=sm_61 \
      -gencode arch=compute_52,code=sm_52 \
      -gencode arch=compute_30,code=sm_30 \
      -DOMPI_SKIP_MPICXX

else
  $(info ************  RELEASE mode ************)
  CFLAGS = \
      -O3 \
      -std=c++11 \
      -fPIC \
      -DOMPI_SKIP_MPICXX \
      -MMD \
      -MP

  CU_FLAGS = \
      -O3 \
      -std=c++11 \
      --compiler-options=-fPIC \
      -use_fast_math \
      --ptxas-options="-v" \
      -gencode arch=compute_70,code=sm_70 \
      -gencode arch=compute_61,code=sm_61 \
      -gencode arch=compute_52,code=sm_52 \
      -gencode arch=compute_30,code=sm_30 \
      -DOMPI_SKIP_MPICXX

endif

# add dependency include dirs as -isystem so that they are
# not included during auto dependency generation (see -MMD option for gcc)
CU_INCLUDES ?= \
    -I/usr/local/include \
    -isystem /usr/local/cuda/include \
    -isystem /usr/lib/openmpi/include \
    -isystem /usr/include/jsoncpp \
    -IB40C \
    -IB40C/KernelCommon

CU_LIBS ?= \
    -L/usr/lib/atlas-base \
    -L/usr/local/cuda/lib64 \
    -L/usr/local/lib/

CU_LOADLIBS = \
    -lcudnn \
    -lcurand \
    -lcublas \
    -lcudart \
    -ljsoncpp \
    -lnetcdf_c++4 \
    -lnetcdf \
    -lblas \
    -ldl \
    -lstdc++
