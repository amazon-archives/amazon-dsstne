add_executable(gputests "TestDune.cpp")
target_link_libraries(gputests PRIVATE dsstne cppunit)
add_test(gputests gputests)
