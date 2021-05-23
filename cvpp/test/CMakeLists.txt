
include(GoogleTest)
file(GLOB TEST_SRC *.cpp)

add_executable(cvppTest ${TEST_SRC})
target_link_libraries(cvppTest gtest gmock gtest_main cvpp)

gtest_discover_tests(cvppTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data/cpu)
