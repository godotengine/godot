foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  set(framework_target ${config_prefix}.test.framework)

  if ("CUDA" STREQUAL "${config_device}")
    set(framework_srcs
      testframework.cu
      cuda/testframework.cu
    )
  else()
    # Wrap the cu file inside a .cpp file for non-CUDA builds
    thrust_wrap_cu_in_cpp(framework_srcs testframework.cu ${thrust_target})
  endif()

  add_library(${framework_target} STATIC ${framework_srcs})
  target_link_libraries(${framework_target} PUBLIC ${thrust_target})
  target_include_directories(${framework_target} PRIVATE "${Thrust_SOURCE_DIR}/testing")
  thrust_clone_target_properties(${framework_target} ${thrust_target})
endforeach()
