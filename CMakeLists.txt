project(cvpp)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND NO_SYCL)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT IPO_SUPPORT OUTPUT error)

	if(IPO_SUPPORT)
		message(STATUS "IPO / LTO enabled")
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
	else()
		message(STATUS "IPO / LTO not supported: <${error}>")
	endif()
endif()

add_subdirectory(cvpp)

if(NOT NO_TEST)
	enable_testing()
	add_subdirectory(test)
endif()
