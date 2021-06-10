pybind11_add_module(_optix ${CMAKE_SOURCE_DIR}/src/main.cpp)

target_link_libraries(_optix PRIVATE "${CUDA_LIBRARIES}")
target_compile_features(_optix PRIVATE cxx_std_17)

add_custom_command(
  TARGET _optix POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${OptiX_INCLUDE}
  $<TARGET_FILE_DIR:_optix>/include
)
