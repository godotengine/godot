# Given a cu_file (e.g. foo/bar.cu) relative to CMAKE_CURRENT_SOURCE_DIR
# and a thrust_target, create a cpp file that includes the .cu file, and set
# ${cpp_file_var} in the parent scope to the full path of the new file. The new
# file will be generated in:
# ${CMAKE_CURRENT_BINARY_DIR}/<thrust_target_prefix>/${cu_file}.cpp
function(thrust_wrap_cu_in_cpp cpp_file_var cu_file thrust_target)
  thrust_get_target_property(prefix ${thrust_target} PREFIX)
  set(wrapped_source_file "${CMAKE_CURRENT_SOURCE_DIR}/${cu_file}")
  set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${prefix}/${cu_file}.cpp")
  configure_file("${Thrust_SOURCE_DIR}/cmake/wrap_source_file.cpp.in" "${cpp_file}")
  set(${cpp_file_var} "${cpp_file}" PARENT_SCOPE)
endfunction()

# Enable RDC for a CUDA target. Encapsulates compiler hacks:
function(thrust_enable_rdc_for_cuda_target target_name)
  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    set_target_properties(${target_name} PROPERTIES
      COMPILE_FLAGS "-gpu=rdc"
    )
  else()
    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
    )
  endif()
endfunction()
