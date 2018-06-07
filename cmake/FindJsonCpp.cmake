# Find the JsonCpp library
#
# This module finds the JsonCpp library and defines a target named
# JsonCpp::jsoncpp which can be linked against. If the library can't
# be found on the system and `REQUIRED` was not passed to the `find_package`
# call, no target is created. Otherwise, if `REQUIRED` was used, an error
# is emitted.

if (TARGET JsonCpp::jsoncpp)
  return()
endif()

find_path(JSONCPP_INCLUDE_DIR "json/features.h" PATH_SUFFIXES jsoncpp)
find_library(JSONCPP_LIBRARY jsoncpp)

if (NOT JSONCPP_INCLUDE_DIR OR NOT JSONCPP_LIBRARY)
  if (JsonCpp_FIND_REQUIRED)
    message(FATAL_ERROR "The JsonCpp library could not be found but it is REQUIRED.")
  else()
    return()
  endif()
endif()

if (NOT JsonCpp_FIND_QUIETLY)
  message(STATUS "Found JsonCpp library at ${JSONCPP_LIBRARY}")
endif()

if ("${JSONCPP_LIBRARY}" MATCHES ".*${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_library(jsoncpp_lib STATIC IMPORTED GLOBAL)
elseif("${JSONCPP_LIBRARY}" MATCHES ".*${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_library(jsoncpp_lib SHARED IMPORTED GLOBAL)
else()
  message(FATAL_ERROR "The JsonCpp library was found at ${JSONCPP_LIBRARY}, but we don't know what type of library it is (shared vs static).")
endif()

target_include_directories(jsoncpp_lib INTERFACE "${JSONCPP_INCLUDE_DIR}")
set_target_properties(jsoncpp_lib PROPERTIES IMPORTED_LOCATION "${JSONCPP_LIBRARY}")
add_library(JsonCpp::jsoncpp ALIAS jsoncpp_lib)
