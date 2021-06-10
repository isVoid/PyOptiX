cmake_minimum_required(VERSION 2.8.12)
project(optix)

set(CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/CMake"
    ${CMAKE_MODULE_PATH}
    )

find_package( OptiX REQUIRED     )
find_package( CUDA 10.0 REQUIRED )

include_directories(
    "${OptiX_INCLUDE}"
    "${CUDA_INCLUDE_DIRS}"
    )

add_subdirectory(pybind11)
add_subdirectory(optix)


pybind11_add_module(optix src/main.cpp)
target_link_libraries( optix PRIVATE
	"${CUDA_LIBRARIES}"
	)
target_compile_features( optix PRIVATE 
    cxx_std_17
    )
