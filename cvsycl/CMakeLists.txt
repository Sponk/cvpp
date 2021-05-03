
file(GLOB SRC src/*.cpp)
file(GLOB_RECURSE INCLUDE include/*.h)

set(CMAKE_PREFIX_PATH /opt/hipSYCL/lib/cmake ${CMAKE_PREFIX_PATH})
#set(HIPSYCL_PLATFORM "tbb" CACHE STRING "" FORCE)
#set(HIPSYCL_PLATFORM "rocm" CACHE STRING "" FORCE)
#set(HIPSYCL_GPU_ARCH "gfx900" CACHE STRING "" FORCE)
#set(HIPSYCL_TARGETS "hip" CACHE STRING "" FORCE)

# set(HIPSYCL_TARGETS "omp;hip:gfx900,gfx906" CACHE STRING "" FORCE)
set(HIPSYCL_TARGETS "omp;hip:gfx900" CACHE STRING "" FORCE)

find_package(hipSYCL CONFIG)
set(SYCL_FOUND ${hipSYCL_FOUND})

#find_package(triSYCL REQUIRED)
#set(SYCL_FOUND ${triSYCL_FOUND})

if(${SYCL_FOUND})
	add_library(cvsycl ${SRC} ${INCLUDE})
	target_link_libraries(cvsycl PUBLIC cvpp)
	target_include_directories(cvsycl PUBLIC include /opt/hipSYCL/include/ /opt/rocm/hip/include/ /opt/hipSYCL/include/hipSYCL/contrib/)
	add_compile_definitions(PUBLIC EIGEN_DONT_VECTORIZE=1) ## So Eigen can be used on the GPU without issues!

	add_sycl_to_target(TARGET cvsycl SOURCES ${SRC})

	if(NOT NO_TEST)
		include(GoogleTest)
		file(GLOB TEST_SRC test/*.cpp)

		add_executable(cvsyclTest ${TEST_SRC})
		target_link_libraries(cvsyclTest PUBLIC gtest gmock gtest_main cvsycl)

		add_sycl_to_target(TARGET cvsyclTest SOURCES ${TEST_SRC})
		gtest_discover_tests(cvsyclTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data)
	endif()

	if(NOT NO_BENCHMARK)
		find_package(benchmark REQUIRED)
		add_executable(cvsyclBench benchmark/main.cpp)
		target_link_libraries(cvsyclBench PUBLIC benchmark cvsycl)

		add_sycl_to_target(TARGET cvsyclBench SOURCES benchmark/main.cpp)
	endif()

else()
	message("-- Disabling SYCL backend")
endif()
