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


if(SWIG_FOUND)

	target_include_directories(cvpp PUBLIC src/swig)

	set_property(SOURCE src/swig/cvpplua.i PROPERTY CPLUSPLUS ON)
	set_property(SOURCE src/swig/cvpplua.i PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

	swig_add_library(cvppLua
		TYPE MODULE
		LANGUAGE lua
		SOURCES src/swig/cvpplua.i
	)

	target_link_libraries(cvppLua cvpp)
	set_property(TARGET cvppLua PROPERTY POSITION_INDEPENDENT_CODE ON)

	## Python
	find_package(Python3 COMPONENTS Development)
	if(Python3_Development_FOUND)
		set_property(SOURCE src/swig/cvpppython.i PROPERTY CPLUSPLUS ON)
		set_property(SOURCE src/swig/cvpppython.i PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

		swig_add_library(cvppPy
			LANGUAGE python
			SOURCES src/swig/cvpppython.i
		)

		set_property(TARGET cvppPy PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

		target_link_libraries(cvppPy cvpp Python3::Module)
		set_property(TARGET cvppPy PROPERTY POSITION_INDEPENDENT_CODE ON)
	endif()

	set_property(TARGET cvpp PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
