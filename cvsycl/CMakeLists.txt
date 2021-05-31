
file(GLOB SRC src/*.cpp)
file(GLOB_RECURSE INCLUDE include/*.h)

option(USE_TRISYCL "Enforce the triSYCL fallback over hipSYCL or ComputeCpp if they are found." OFF)
option(USE_COMPUTECPP "Enforce the ComputeCpp hipSYCL or triSYCL." OFF)

set(SYCL_FOUND FALSE)

if(NOT USE_TRISYCL AND NOT USE_COMPUTECPP)
	set(CMAKE_PREFIX_PATH /opt/hipSYCL/lib/cmake ${CMAKE_PREFIX_PATH})
	set(HIPSYCL_TARGETS "omp;hip:gfx900" CACHE STRING "" FORCE)

	find_package(hipSYCL CONFIG)
	set(SYCL_FOUND ${hipSYCL_FOUND})

	message("-- Using hipSYCL")

elseif(NOT USE_TRISYCL AND USE_COMPUTECPP)
	set(ComputeCpp_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ComputeCpp)
	set(SYCL_LANGUAGE_VERSION "2020")

	find_package(ComputeCpp)
	set(SYCL_FOUND ${ComputeCpp_FOUND})

	set(CXX_STANDARD_REQUIRED 20)
	set(CXX_STANDARD 20)
endif()

if((NOT SYCL_FOUND AND NOT USE_COMPUTECPP) OR USE_TRISYCL)
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

	## Fetch content seems to be broken on Apple...
#	if(APPLE)
		set(triSYCL_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/trisycl-src)
		include(${triSYCL_SOURCE_DIR}/cmake/FindtriSYCL.cmake)
#	else()
#		include(${triSYCL_SOURCE_DIR}/cmake/FindtriSYCL.cmake)
#	endif()

	message("-- Fetched triSYCL: ${triSYCL_SOURCE_DIR}")

	add_compile_definitions(TRISYCL=1)

	#function(add_sycl_to_target T targetName)
	#	message(${triSYCL_SOURCE_DIR}/include)
	#	target_include_directories(${targetName} PUBLIC ${triSYCL_SOURCE_DIR}/include)
	#endfunction()

	set(SYCL_FOUND TRUE)
	set(USE_TRISYCL TRUE)
endif()

if(${SYCL_FOUND})
	add_library(cvsycl ${SRC} ${INCLUDE})
	target_link_libraries(cvsycl PUBLIC cvpp)
	target_include_directories(cvsycl PUBLIC include)

	# Doesn't seem to be necessary with current hipSYCL anymore
	#	add_compile_definitions(PUBLIC EIGEN_DONT_VECTORIZE=1) ## So Eigen can be used on the GPU without issues!

	if(NOT USE_TRISYCL)
		add_sycl_to_target(TARGET cvsycl SOURCES ${SRC})
	else()
		add_sycl_to_target(cvsycl)
	endif()

	if(NOT NO_TEST)
		file(GLOB TEST_SRC test/*.cpp)

		add_executable(cvsyclTest ${TEST_SRC})
		target_link_libraries(cvsyclTest PUBLIC gtest gmock gtest_main cvsycl)


		if(NOT USE_TRISYCL)
			add_sycl_to_target(TARGET cvsyclTest SOURCES ${TEST_SRC})
		else()
			add_sycl_to_target(cvsyclTest)
		endif()

		gtest_discover_tests(cvsyclTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test/data/sycl)
	endif()

	if(NOT NO_BENCHMARK)
		find_package(benchmark REQUIRED)
		add_executable(cvsyclBench benchmark/main.cpp)
		target_link_libraries(cvsyclBench PUBLIC benchmark cvsycl)

		if(NOT USE_TRISYCL)
			add_sycl_to_target(TARGET cvsyclBench SOURCES benchmark/main.cpp)
		else()
			add_sycl_to_target(cvsyclBench)
		endif()
	endif()

	if(SWIG_FOUND)

		## SYCL
		set_property(SOURCE src/swig/cvsyclLua.i PROPERTY CPLUSPLUS ON)
		set_property(SOURCE src/swig/cvsyclLua.i PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

		swig_add_library(cvsyclLua
			TYPE MODULE
			LANGUAGE lua
			SOURCES src/swig/cvsyclLua.i
		)

		target_link_libraries(cvsyclLua PUBLIC cvsycl)
		set_property(TARGET cvsyclLua PROPERTY POSITION_INDEPENDENT_CODE ON)

		if(NOT USE_TRISYCL)
			add_sycl_to_target(TARGET cvsyclLua)
		else()
			add_sycl_to_target(cvsyclLua)
		endif()

		## Python
		find_package(Python3 COMPONENTS Development)
		if(Python3_Development_FOUND)
			set_property(SOURCE src/swig/cvsyclPython.i PROPERTY CPLUSPLUS ON)
			set_property(SOURCE src/swig/cvsyclPython.i PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

			swig_add_library(cvsyclPy
				LANGUAGE python
				SOURCES src/swig/cvsyclPython.i
			)

			set_property(TARGET cvsyclPy PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

			target_link_libraries(cvsyclPy PUBLIC cvsycl Python3::Module)
			set_property(TARGET cvsyclPy PROPERTY POSITION_INDEPENDENT_CODE ON)

			if(NOT USE_TRISYCL)
				add_sycl_to_target(TARGET cvsyclPy)
			else()
				add_sycl_to_target(cvsyclPy)
			endif()
		endif()

		set_property(TARGET cvsycl PROPERTY POSITION_INDEPENDENT_CODE ON)
	endif()
else()
	message("-- Disabling SYCL backend")
endif()
