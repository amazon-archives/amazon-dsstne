# Find the NetCDF-4 library
#
# This module finds the NetCDF-4 C++ library [1] and defines a target named
# NetCDF4::netcdf which can be linked against. If the library can't be found
# on the system and `REQUIRED` was not passed to the `find_package` call, no
# target is created. Otherwise, if `REQUIRED` was used, an error is emitted.
#
# This library depends on the NetCDF library.
#
# [1]: https://github.com/Unidata/netcdf-cxx4

if (TARGET NetCDF4::netcdf)
  return()
endif()

find_path(NETCDF4_INCLUDE_DIR "ncInt.h")
find_library(NETCDF4_LIBRARY netcdf_c++4)

# Try to find the NetCDF C library, which the C++ library depends on.
set(NetCDF_FIND_QUIETLY "${NetCDF4_FIND_QUIETLY}")
if (NetCDF4_FIND_REQUIRED)
  find_package(NetCDF REQUIRED)
else()
  find_package(NetCDF)
endif()

if (NOT NETCDF4_INCLUDE_DIR OR NOT NETCDF4_LIBRARY OR NOT TARGET NetCDF::netcdf)
  if (NetCDF4_FIND_REQUIRED)
    message(FATAL_ERROR "The NetCDF-4 library or one of its dependencies could not be found but it is REQUIRED.")
  else()
    return()
  endif()
endif()

if (NOT NetCDF4_FIND_QUIETLY)
  message(STATUS "Found NetCDF-4 library at ${NETCDF4_LIBRARY}")
endif()

if ("${NETCDF4_LIBRARY}" MATCHES ".*${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_library(netcdf4_lib STATIC IMPORTED GLOBAL)
elseif("${NETCDF4_LIBRARY}" MATCHES ".*${CMAKE_SHARED_LIBRARY_SUFFIX}")
  add_library(netcdf4_lib SHARED IMPORTED GLOBAL)
else()
  message(FATAL_ERROR "The NetCDF-4 library was found at ${NETCDF4_LIBRARY}, but we don't know what type of library it is (shared vs static).")
endif()

target_include_directories(netcdf4_lib INTERFACE "${NETCDF4_INCLUDE_DIR}")
set_target_properties(netcdf4_lib PROPERTIES IMPORTED_LOCATION "${NETCDF4_LIBRARY}")
target_link_libraries(netcdf4_lib INTERFACE NetCDF::netcdf) # Link to the C library
add_library(NetCDF4::netcdf ALIAS netcdf4_lib)
