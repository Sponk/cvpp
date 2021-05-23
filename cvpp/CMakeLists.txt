file(GLOB SRC src/*.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(OPENMP_FLAGS "OpenMP::OpenMP_CXX")
endif()

add_library(cvpp ${SRC})
target_include_directories(cvpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cvpp PUBLIC Eigen3::Eigen ${OPENMP_FLAGS})

if(NOT NO_TEST)
	file(GLOB TEST_SRC test/*.cpp)

	add_executable(cvppTest ${TEST_SRC})
	target_link_libraries(cvppTest PUBLIC gtest gmock gtest_main cvpp)

	gtest_discover_tests(cvppTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data/cpu)
endif()
