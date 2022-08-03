file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

# These tests always build with RDC, so make sure that the sm_XX flags are
# compatible. See note in ThrustCudaConfig.cmake.
set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_RDC}")

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  if (NOT config_device STREQUAL "CUDA")
    continue()
  endif()

  foreach(test_src IN LISTS test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    string(PREPEND test_name "cuda.")

    thrust_add_test(test_target ${test_name} "${test_src}" ${thrust_target})

    # All in testing/cuda will test device-side launch (aka calling parallel
    # algorithms from device code), which requires the CUDA device-side runtime,
    # which requires RDC, so these always need to be built with RDC.
    thrust_enable_rdc_for_cuda_target(${test_target})
  endforeach()
endforeach()
