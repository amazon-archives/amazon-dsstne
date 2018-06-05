include(CMakeFindDependencyMacro)
find_dependency(JsonCpp REQUIRED)
find_dependency(MPI REQUIRED)
find_dependency(NetCDF4 REQUIRED)

if(NOT TARGET amazon-dsstne::dsstne)
  include("${CMAKE_CURRENT_LIST_DIR}/amazon-dsstne-targets.cmake")
endif()
