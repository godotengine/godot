# This test should always use per-thread streams on NVCC.
set_target_properties(${test_target} PROPERTIES
  COMPILE_OPTIONS
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--default-stream=per-thread>
)

# NVC++ does not have an equivalent option, and will always
# use the global stream by default.
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Feta")
  set_tests_properties(${test_target} PROPERTIES WILL_FAIL ON)
endif()
