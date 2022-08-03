# The async tests perform a large amount of codegen, making them expensive to
# build and test. To keep compilation and runtimes manageable, the tests are
# broken up into many files per algorithm to enable parallelism during
# compilation and testing. The structure of these test directories are:
#
# thrust/testing/async/<algorithm_name>/<unit_test>.cu
#
# These generate executables and CTest tests named
# ${config_prefix}.test.async.<algorithm_name>.<unit_test>.

# The async tests only support CUDA enabled configs. Create a list of valid
# thrust targets:
set(cuda_configs)
foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  if (config_device STREQUAL CUDA)
    list(APPEND cuda_configs ${thrust_target})
  endif()
endforeach()

list(LENGTH cuda_configs num_cuda_configs)
if (num_cuda_configs EQUAL 0)
  return() # No valid configs found, nothing to do.
endif()

# Process a single algorithm directory, adding all .cu/cpp files as tests for
# each valid backend. algo_name is the name of the subdir (<algorithm_name>
# above) and is used for naming the executable/targets.
function(thrust_add_async_test_dir algo_name)
  file(GLOB test_srcs
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
    CONFIGURE_DEPENDS
    "${algo_name}/*.cu"
    "${algo_name}/*.cpp"
  )

  # Per-algorithm, all-config metatarget: thrust.all.test.async.[algo].all
  set(algo_meta_target thrust.all.test.async.${algo_name}.all)
  add_custom_target(${algo_meta_target})

  foreach(thrust_target IN LISTS cuda_configs)
    thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

    # Per-algorithm, per-config metatarget: thrust.[config].test.async.[algo].all
    set(algo_config_meta_target ${config_prefix}.test.async.${algo_name}.all)
    add_custom_target(${algo_config_meta_target})
    add_dependencies(${algo_meta_target} ${algo_config_meta_target})

    foreach(test_src IN LISTS test_srcs)
      get_filename_component(test_name "${test_src}" NAME_WLE)
      string(PREPEND test_name async.${algo_name}.)

      thrust_add_test(test_target ${test_name} "${test_src}" ${thrust_target})
      if(THRUST_ENABLE_TESTS_WITH_RDC)
        thrust_enable_rdc_for_cuda_target(${test_target})
      endif()

      add_dependencies(${algo_config_meta_target} ${test_target})
    endforeach()
  endforeach()
endfunction()

# Grab all algorithm subdirectories:
set(test_dirs)
file(GLOB contents
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*"
)

foreach(test_dir IN LISTS contents)
  if(IS_DIRECTORY "${test_dir}")
    list(APPEND test_dirs "${test_dir}")
  endif()
endforeach()

# Process all test dirs:
foreach(test_dir IN LISTS test_dirs)
  get_filename_component(algo_name "${test_dir}" NAME_WLE)
  thrust_add_async_test_dir(${algo_name})
endforeach()
