
add_executable(dparse dparse.cpp)
target_link_libraries(dparse PRIVATE NetCDF4::netcdf)
target_compile_features(dparse PRIVATE cxx_std_11)
install(TARGETS dparse RUNTIME DESTINATION bin)
