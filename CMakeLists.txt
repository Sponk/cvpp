project(cvpp)
cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND (NO_SYCL OR USE_TRISYCL))
	include(CheckIPOSupported)
	check_ipo_supported(RESULT IPO_SUPPORT OUTPUT error)

	if(IPO_SUPPORT)
		message(STATUS "IPO / LTO enabled")
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
	else()
		message(STATUS "IPO / LTO not supported: <${error}>")
	endif()
endif()

option(ENABLE_SANITIZER "Enables address sanitizer" OFF)

if(${ENABLE_SANITIZER})
	set(SANITIZE_FLAGS "-fsanitize=address,undefined,float-divide-by-zero")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --save-temps ${SANITIZE_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${SANITIZE_FLAGS}")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --save-temps ${SANITIZE_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${SANITIZE_FLAGS}")
endif()

if(NOT NO_TEST)
	include(GoogleTest)
	enable_testing()
endif()

if(NOT NO_SWIG)
	find_package(SWIG)
	include(UseSWIG)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_SWIG_OUTDIR ${CMAKE_BINARY_DIR}/bin)

set(UseSWIG_MODULE_VERSION 2)

add_subdirectory(cvpp)
add_subdirectory(cvsycl)
add_subdirectory(cvcl)
