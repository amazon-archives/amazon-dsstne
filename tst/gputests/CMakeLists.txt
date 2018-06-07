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
PKG_CHECK_MODULES(NETCDF REQUIRED netcdf)
PKG_CHECK_MODULES(NETCDF_CXX4 REQUIRED netcdf-cxx4)

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
    ${MPI_CXX_INCLUDE_PATH}
    ${NETCDF_INCLUDE_DIR}
    ${NETCDF_CXX4_INCLUDE_DIR}
)

set(ENGINE_SOURCES
    ${ENGINE_DIR}/GpuTypes.cpp
    ${ENGINE_DIR}/kernels.cu
    ${ENGINE_DIR}/kActivation.cu
    ${ENGINE_DIR}/kDelta.cu
    ${ENGINE_DIR}/kLoss.cu
)

set(UTILS_SOURCES
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
    ${MPI_CXX_LIBRARIES}
    ${NETCDF_LIBRARIES}
    ${NETCDF_CXX4_LIBRARIES}
)
