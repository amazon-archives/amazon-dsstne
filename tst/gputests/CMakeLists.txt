cmake_minimum_required (VERSION 3.2)

project (amazon-dsstne)

################################################################################
#
# Check for C++11 support
#
################################################################################

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(!COMPILER_SUPPORTS_CXX11)
    message(FATAL_ERROR "Your compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

################################################################################
#
# Dependencies
#
################################################################################

find_package(CUDA)
find_package(MPI)
find_package(PkgConfig)

PKG_CHECK_MODULES(CPPUNIT REQUIRED cppunit)
PKG_CHECK_MODULES(JSONCPP REQUIRED jsoncpp)
PKG_CHECK_MODULES(NETCDF REQUIRED netcdf)
PKG_CHECK_MODULES(NETCDF_CXX4 REQUIRED netcdf-cxx4)

################################################################################
#
# CuDNN
#
# Based on CMake scripts from Caffe:
# https://github.com/BVLC/caffe/blob/master/cmake/Cuda.cmake
#
################################################################################

function(detect_cuDNN)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

    find_path(CUDNN_INCLUDE cudnn.h
             PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
             DOC "Path to cuDNN include directory." )

    get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
    find_library(CUDNN_LIBRARY NAMES libcudnn.so # libcudnn_static.a
                              PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist}
                              DOC "Path to cuDNN library.")

    if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
        set(HAVE_CUDNN  TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)

        mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)
        message(STATUS "Found cuDNN (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")
    endif()
endfunction()

detect_cuDNN()
if(HAVE_CUDNN)
    add_definitions(-DUSE_CUDNN)
    include_directories(SYSTEM ${CUDNN_INCLUDE})
    list(APPEND Caffe_LINKER_LIBS ${CUDNN_LIBRARY})
endif()

################################################################################
#
# Compiler flags
#
################################################################################

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_NVCC_FLAGS} -use_fast_math -gencode arch=compute_50,code=sm_50 -gencode arch=compute_30,code=sm_30 -DOMPI_SKIP_MPICXX -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################################################################################
#
# Test suite
#
################################################################################

set(ENGINE_DIR ../../src/amazon/dsstne/engine)
set(UTILS_DIR ../../src/amazon/dsstne/utils)

include_directories(
    ${ENGINE_DIR}
    ${UTILS_DIR}
    ${CPPUNIT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_PATH}
    ${NETCDF_INCLUDE_DIR}
    ${NETCDF_CXX4_INCLUDE_DIR}
)

set(ENGINE_SOURCES
    ${ENGINE_DIR}/GpuTypes.cpp
    ${ENGINE_DIR}/NNLayer.cpp
    ${ENGINE_DIR}/NNNetwork.cpp
    ${ENGINE_DIR}/NNTypes.cpp
    ${ENGINE_DIR}/NNWeight.cpp
    ${ENGINE_DIR}/kernels.cu
    ${ENGINE_DIR}/kActivation.cu
    ${ENGINE_DIR}/kDelta.cu
    ${ENGINE_DIR}/kLoss.cu
)

set(UTILS_SOURCES
    ${UTILS_DIR}/NetCDFhelper.cpp
    ${UTILS_DIR}/Utils.cpp
)

set(TEST_SOURCES
    TestDune.cpp
)

cuda_add_executable(gputests
    ${ENGINE_SOURCES}
    ${TEST_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(gputests
    ${CPPUNIT_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
    ${JSONCPP_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NETCDF_LIBRARIES}
    ${NETCDF_CXX4_LIBRARIES}
)
