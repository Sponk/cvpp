
file(GLOB SRC src/*.cpp)
file(GLOB_RECURSE INCLUDE include/*.h)

find_package(OpenCL)

if(${OpenCL_FOUND})
	add_library(cvcl ${SRC} ${INCLUDE})
	target_link_libraries(cvcl PUBLIC cvpp OpenCL::OpenCL)
	target_include_directories(cvcl PUBLIC include)

	if(NOT NO_TEST)
		file(GLOB TEST_SRC test/*.cpp)

		add_executable(cvclTest ${TEST_SRC})
		target_link_libraries(cvclTest PUBLIC gtest gmock gtest_main cvcl)

		#gtest_discover_tests(cvclTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data/opencl)
	endif()

	if(NOT NO_BENCHMARK)
		find_package(benchmark REQUIRED)
		add_executable(cvclBench benchmark/main.cpp)
		target_link_libraries(cvclBench PUBLIC benchmark cvcl)
	endif()
else()
	message("-- Disabling OpenCL backend")
endif()
