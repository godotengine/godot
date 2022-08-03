# Disable unreachable code warnings.
# This test unconditionally throws in some places, the compiler will detect that
# control flow will never reach some instructions. This is intentional.
target_link_libraries(${test_target} PRIVATE thrust.silence_unreachable_code_warnings)

# The machinery behind this test is not compatible with NVC++.
# See https://github.com/NVIDIA/thrust/issues/1397
if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set_tests_properties(${test_target} PROPERTIES DISABLED True)
endif()
