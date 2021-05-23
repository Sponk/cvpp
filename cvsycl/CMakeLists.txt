
file(GLOB SRC src/*.cpp)
file(GLOB_RECURSE INCLUDE include/*.h)

option(USE_TRISYCLE "Enforce the triSYCL fallback over hipSYCL or ComputeCpp if they are found." OFF)
option(USE_COMPUTECPP "Enforce the ComputeCpp hipSYCL or triSYCL." OFF)

set(SYCL_FOUND FALSE)

if(NOT USE_TRISYCL AND NOT USE_COMPUTECPP)
	set(CMAKE_PREFIX_PATH /opt/hipSYCL/lib/cmake ${CMAKE_PREFIX_PATH})
	set(HIPSYCL_TARGETS "omp;hip:gfx900" CACHE STRING "" FORCE)

	find_package(hipSYCL CONFIG)
	set(SYCL_FOUND ${hipSYCL_FOUND})
elseif(NOT USE_TRISYCL AND USE_COMPUTECPP)
	set(ComputeCpp_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ComputeCpp)
	set(SYCL_LANGUAGE_VERSION "2020")

	find_package(ComputeCpp)
	set(SYCL_FOUND ${ComputeCpp_FOUND})

	set(CXX_STANDARD_REQUIRED 20)
	set(CXX_STANDARD 20)
endif()

if(NOT ${SYCL_FOUND} AND ${USE_TRISYCLE} AND NOT ${USE_COMPUTECPP})
	include(FetchContent)
	
	message("-- Fetching triSYCL")
	FetchContent_Declare(triSYCL
		GIT_REPOSITORY https://github.com/triSYCL/triSYCL.git
		GIT_TAG        origin/master
	)
	
	FetchContent_GetProperties(triSYCL)
	if(NOT triSYCL_POPULATED)
		FetchContent_Populate(triSYCL)
	endif()

	function(add_sycl_to_target T targetName)
		target_include_directories(${targetName} PUBLIC ${triSYCL_SOURCE_DIR}/include)
	endfunction()

	set(SYCL_FOUND TRUE)
endif()

if(${SYCL_FOUND})
	add_library(cvsycl ${SRC} ${INCLUDE})
	target_link_libraries(cvsycl PUBLIC cvpp)
	target_include_directories(cvsycl PUBLIC include)

	# Doesn't seem to be necessary with current hipSYCL anymore
	#	add_compile_definitions(PUBLIC EIGEN_DONT_VECTORIZE=1) ## So Eigen can be used on the GPU without issues!

	add_sycl_to_target(TARGET cvsycl SOURCES ${SRC})

	if(NOT NO_TEST)
		file(GLOB TEST_SRC test/*.cpp)

		add_executable(cvsyclTest ${TEST_SRC})
		target_link_libraries(cvsyclTest PUBLIC gtest gmock gtest_main cvsycl)

		add_sycl_to_target(TARGET cvsyclTest SOURCES ${TEST_SRC})
		gtest_discover_tests(cvsyclTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data/sycl)
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
