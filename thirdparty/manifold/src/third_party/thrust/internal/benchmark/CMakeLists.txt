if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # MSVC builds fail at runtime. Benchmarks are linux-only for now.
  message(STATUS "Thrust benchmarking is not available on MSVC.")
  return()
endif()

add_custom_target(thrust.all.bench)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Skip non cpp.cuda targets:
  if (NOT config_host   STREQUAL "CPP" OR
      NOT config_device STREQUAL "CUDA")
    continue()
  endif()

  set(bench_target ${config_prefix}.bench)

  add_executable(${bench_target} bench.cu)
  target_link_libraries(${bench_target} PRIVATE ${thrust_target})
  target_include_directories(${bench_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  thrust_clone_target_properties(${bench_target} ${thrust_target})

  add_dependencies(thrust.all.bench ${bench_target})
  add_dependencies(${config_prefix}.all ${bench_target})
endforeach()
