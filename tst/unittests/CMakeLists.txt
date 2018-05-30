add_executable(unittests "main.cpp")
target_link_libraries(unittests PRIVATE dsstne cppunit)
add_test(unittests unittests)
