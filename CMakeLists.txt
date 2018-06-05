cmake_minimum_required(VERSION 3.10)
project(amazon-dsstne LANGUAGES CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "The CMake build type was not specified -- using '${CMAKE_BUILD_TYPE}'")
else()
  message(STATUS "The CMake build type is '${CMAKE_BUILD_TYPE}'")
endif()


################################################################################
# Dependencies
################################################################################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CppUnit REQUIRED)
find_package(JsonCpp REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(NetCDF4 REQUIRED)
find_package(TensorflowCC REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)


################################################################################
# Setup a CUDA interface library that contains all of what DSSTNE needs
################################################################################
find_package(CUDA REQUIRED)
add_library(dsstne_cuda INTERFACE)
target_include_directories(dsstne_cuda INTERFACE ${CUDA_INCLUDE_DIRS})
find_library(CUDNN_LIBRARY cudnn HINTS "${CUDA_INCLUDE_DIRS}/../lib"
                                       "${CUDA_INCLUDE_DIRS}/../lib64")
target_link_libraries(dsstne_cuda INTERFACE ${CUDA_LIBRARIES}
                                            ${CUDA_CUBLAS_LIBRARIES}
                                            ${CUDA_curand_LIBRARY}
                                            ${CUDNN_LIBRARY})


################################################################################
# Setup the dsstne library
################################################################################
file(GLOB DSSTNE_SOURCES "src/amazon/dsstne/engine/*.cpp"
                         "src/amazon/dsstne/engine/*.cu"
                         "src/amazon/dsstne/utils/Config.cpp"
                         "src/amazon/dsstne/utils/FilterHelper.cpp"
                         "src/amazon/dsstne/utils/Filters.cpp"
                         "src/amazon/dsstne/utils/NetCDFhelper.cpp"
                         "src/amazon/dsstne/utils/NNRecsGenerator.cpp"
                         "src/amazon/dsstne/utils/TensorboardMetricsLogger.cpp"
                         "src/amazon/dsstne/utils/Utils.cpp")
add_library(dsstne ${DSSTNE_SOURCES})
target_include_directories(dsstne PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
                                         "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/amazon/dsstne/engine>"
                                         "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/amazon/dsstne/utils>")
# The following is needed because of https://gitlab.kitware.com/cmake/cmake/issues/17538
target_include_directories(dsstne PUBLIC /usr/lib/openmpi/include)
target_compile_features(dsstne PUBLIC cxx_std_11)
target_compile_options(dsstne PRIVATE -use_fast_math)
target_link_libraries(dsstne PUBLIC JsonCpp::jsoncpp NetCDF4::netcdf MPI::MPI_CXX dsstne_cuda TensorflowCC::Shared Boost::filesystem)

set(TARGET_ARCHITECTURES -gencode=arch=compute_70,code=sm_70
                         -gencode=arch=compute_61,code=sm_61
                         -gencode=arch=compute_52,code=sm_52
                         -gencode=arch=compute_30,code=sm_30)
target_compile_options(dsstne PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${TARGET_ARCHITECTURES}>)


################################################################################
# Setup tools
################################################################################
add_executable(train "src/amazon/dsstne/utils/Train.cpp")
target_link_libraries(train PRIVATE dsstne Boost::filesystem)

add_executable(predict "src/amazon/dsstne/utils/Predict.cpp")
target_link_libraries(predict PRIVATE dsstne)

add_executable(encoder "src/amazon/dsstne/utils/main.cpp")
target_link_libraries(encoder PRIVATE dsstne)

add_executable(generateNetCDF "src/amazon/dsstne/utils/NetCDFGenerator.cpp"
                              "src/amazon/dsstne/utils/NetCDFhelper.cpp")
target_link_libraries(generateNetCDF PRIVATE dsstne)


################################################################################
# Setup tests and examples
################################################################################
enable_testing()
add_subdirectory(tst/gputests)
add_subdirectory(tst/unittests)
add_subdirectory(samples/cifar-10)


##############################################################################
# Setup the installation target for dsstne and its tools
##############################################################################
install(TARGETS dsstne dsstne_cuda train predict encoder generateNetCDF
  EXPORT ${PROJECT_NAME}-targets
  INCLUDES DESTINATION include
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake/${PROJECT_NAME}
)
install(FILES cmake/${PROJECT_NAME}-config.cmake
              cmake/FindCppUnit.cmake
              cmake/FindJsonCpp.cmake
              cmake/FindNetCDF.cmake
              cmake/FindNetCDF4.cmake
        DESTINATION lib/cmake/${PROJECT_NAME})
install(DIRECTORY src/amazon
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
