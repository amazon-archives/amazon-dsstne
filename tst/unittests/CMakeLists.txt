cmake_minimum_required (VERSION 3.2)

project (amazon-dsstne)

################################################################################
#
# Dependencies
#
################################################################################

find_package(PkgConfig)

PKG_CHECK_MODULES(CPPUNIT REQUIRED cppunit)

################################################################################
#
# Test suite
#
################################################################################

set(UTILS_DIR ../../src/amazon/dsstne/utils)

include_directories(
    ${UTILS_DIR}
    ${CPPUNIT_INCLUDE_DIR}
)

set(UTILS_SOURCES
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
)
