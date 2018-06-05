# Find the CppUnit library
#
# This module finds the CppUnit library and defines a target named
# CppUnit::cppunit which can be linked against. If the library can't
# be found on the system and `REQUIRED` was not passed to the `find_package`
# call, no target is created. Otherwise, if `REQUIRED` was used, an error
# is emitted.

if (TARGET CppUnit::cppunit)
  return()
endif()

find_path(CPPUNIT_INCLUDE_DIR "cppunit/TestCase.h")
find_library(CPPUNIT_LIBRARY cppunit)

if (NOT CPPUNIT_INCLUDE_DIR OR NOT CPPUNIT_LIBRARY)
  if (CppUnit_FIND_REQUIRED)
    message(FATAL_ERROR "The CppUnit library could not be found but it is REQUIRED.")
  else()
    return()
  endif()
endif()

if (NOT CppUnit_FIND_QUIETLY)
  message(STATUS "Found CppUnit library at ${CPPUNIT_LIBRARY}")
endif()

if ("${CPPUNIT_LIBRARY}" MATCHES ".*${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_library(cppunit_lib STATIC IMPORTED GLOBAL)
elseif("${CPPUNIT_LIBRARY}" MATCHES ".*${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_library(cppunit_lib SHARED IMPORTED GLOBAL)
else()
  message(FATAL_ERROR "The CppUnit library was found at ${CPPUNIT_LIBRARY}, but we don't know what type of library it is (shared vs static).")
endif()

target_include_directories(cppunit_lib INTERFACE "${CPPUNIT_INCLUDE_DIR}")
set_target_properties(cppunit_lib PROPERTIES IMPORTED_LOCATION "${CPPUNIT_LIBRARY}")
add_library(CppUnit::cppunit ALIAS cppunit_lib)
