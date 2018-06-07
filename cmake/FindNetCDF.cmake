# Find the NetCDF library
#
# This module finds the NetCDF library and defines a target named NetCDF::netcdf
# which can be linked against. If the library can't be found on the system and
# `REQUIRED` was not passed to the `find_package` call, no target is created.
# Otherwise, if `REQUIRED` was used, an error is emitted.

if (TARGET NetCDF::netcdf)
  return()
endif()

find_path(NETCDF_INCLUDE_DIR "netcdf.h")
find_library(NETCDF_LIBRARY netcdf)

if (NOT NETCDF_INCLUDE_DIR OR NOT NETCDF_LIBRARY)
  if (NetCDF_FIND_REQUIRED)
    message(FATAL_ERROR "The NetCDF library could not be found but it is REQUIRED.")
  else()
    return()
  endif()
endif()

if (NOT NetCDF_FIND_QUIETLY)
  message(STATUS "Found NetCDF library at ${NETCDF_LIBRARY}")
endif()

if ("${NETCDF_LIBRARY}" MATCHES ".*${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_library(netcdf_lib STATIC IMPORTED GLOBAL)
elseif("${NETCDF_LIBRARY}" MATCHES ".*${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_library(netcdf_lib SHARED IMPORTED GLOBAL)
else()
  message(FATAL_ERROR "The NetCDF library was found at ${NETCDF_LIBRARY}, but we don't know what type of library it is (shared vs static).")
endif()

target_include_directories(netcdf_lib INTERFACE "${NETCDF_INCLUDE_DIR}")
set_target_properties(netcdf_lib PROPERTIES IMPORTED_LOCATION "${NETCDF_LIBRARY}")
add_library(NetCDF::netcdf ALIAS netcdf_lib)
