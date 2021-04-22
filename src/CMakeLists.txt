file(GLOB SRC *.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(OPENMP_FLAGS "OpenMP::OpenMP_CXX")
endif()

add_library(cvpp ${SRC})
target_include_directories(cvpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cvpp PUBLIC Eigen3::Eigen ${OPENMP_FLAGS})
