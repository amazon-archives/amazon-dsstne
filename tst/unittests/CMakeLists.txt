cmake_minimum_required (VERSION 3.2)

project (amazon-dsstne)

################################################################################
#
# Compiler configuration
#
################################################################################

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(FATAL_ERROR "Your compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

################################################################################
#
# Dependencies
#
################################################################################

find_package(PkgConfig)

PKG_CHECK_MODULES(CPPUNIT REQUIRED cppunit)
PKG_CHECK_MODULES(NETCDF REQUIRED netcdf)
PKG_CHECK_MODULES(NETCDF_CXX4 REQUIRED netcdf-cxx4)

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
    ${NETCDF_INCLUDE_DIR}
    ${NETCDF_CXX4_INCLUDE_DIR}
)

set(UTILS_SOURCES
    ${UTILS_DIR}/NetCDFhelper.cpp
    ${UTILS_DIR}/Utils.cpp
)

set(TEST_SOURCES
    main.cpp
)

add_executable(unittests
    ${TEST_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(unittests
    ${CPPUNIT_LIBRARIES}
    ${NETCDF_LIBRARIES}
    ${NETCDF_CXX4_LIBRARIES}
)
