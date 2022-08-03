# Setup FileCheck if requested and available:
option(THRUST_ENABLE_EXAMPLE_FILECHECK
  "Check example output with the LLVM FileCheck utility."
  OFF
)
set(filecheck_data_path "${Thrust_SOURCE_DIR}/internal/test")

if (THRUST_ENABLE_EXAMPLE_FILECHECK)
  # TODO this should go into a find module
  find_program(THRUST_FILECHECK_EXECUTABLE
    DOC "Path to the LLVM FileCheck utility."
    NAMES
      FileCheck
      FileCheck-3.9
      FileCheck-4.0
      FileCheck-5.0
      FileCheck-6.0
      FileCheck-7
      FileCheck-8
      FileCheck-9
  )

  if (NOT THRUST_FILECHECK_EXECUTABLE)
    message(FATAL_ERROR
      "Could not find the LLVM FileCheck utility. Set THRUST_FILECHECK_EXECUTABLE manually, "
      "or disable THRUST_ENABLE_EXAMPLE_FILECHECK."
    )
  endif()

  execute_process(
    COMMAND "${THRUST_FILECHECK_EXECUTABLE}" "${filecheck_data_path}/thrust.smoke.filecheck"
    INPUT_FILE "${Thrust_SOURCE_DIR}/cmake/filecheck_smoke_test"
    RESULT_VARIABLE exit_code
  )

  if (0 EQUAL exit_code)
    message(STATUS "FileCheck enabled: ${THRUST_FILECHECK_EXECUTABLE}")
  else()
    message(FATAL_ERROR
      "The current THRUST_FILECHECK_EXECUTABLE ('${THRUST_FILECHECK_EXECUTABLE}') "
      "does not seem to be a valid FileCheck executable."
    )
  endif()
endif()

# Create meta targets that build all examples for a single configuration:
foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  set(config_meta_target ${config_prefix}.examples)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

# Update flags to reflect RDC options. See note in ThrustCudaConfig.cmake --
# these flag variables behave unintuitively:
if (THRUST_ENABLE_EXAMPLES_WITH_RDC)
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_RDC}")
else()
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_NO_RDC}")
endif()

## thrust_add_example
#
# Add an example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for post-processing target information per-backend.
# example_name: The name of the example minus "<config_prefix>.example." For
#   instance, examples/vector.cu will be "vector", and examples/cuda/copy.cu
#   would be "cuda.copy".
# example_src: The source file that implements the example.
# thrust_target: The reference thrust target with configuration information.
#
function(thrust_add_example target_name_var example_name example_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Wrap the .cu file in .cpp for non-CUDA backends
  if ("CUDA" STREQUAL "${config_device}")
    set(real_example_src "${example_src}")
  else()
    thrust_wrap_cu_in_cpp(real_example_src "${example_src}" ${thrust_target})
  endif()

  # The actual name of the test's target:
  set(example_target ${config_prefix}.example.${example_name})
  set(${target_name_var} ${example_target} PARENT_SCOPE)

  # Related target names:
  set(config_meta_target ${config_prefix}.examples)
  set(example_meta_target thrust.all.example.${example_name})

  add_executable(${example_target} "${real_example_src}")
  target_link_libraries(${example_target} ${thrust_target})
  target_include_directories(${example_target} PRIVATE "${Thrust_SOURCE_DIR}/examples")
  thrust_clone_target_properties(${example_target} ${thrust_target})

  # Add to the active configuration's meta target
  add_dependencies(${config_meta_target} ${example_target})

  # Meta target that builds examples with this name for all configurations:
  if (NOT TARGET ${example_meta_target})
    add_custom_target(${example_meta_target})
  endif()
  add_dependencies(${example_meta_target} ${example_target})

  if ("CUDA" STREQUAL "${config_device}" AND
      THRUST_ENABLE_EXAMPLES_WITH_RDC)
    thrust_enable_rdc_for_cuda_target(${example_target})
  endif()

  # Get the name of FileCheck input by stripping out the config name.
  # (e.g. "thrust.cpp.cuda.cpp14.example.xxx" -> "thrust.example.xxx.filecheck")
  string(REPLACE "${config_prefix}" "thrust"
    filecheck_reference_file
    "${example_target}.filecheck"
  )

  add_test(NAME ${example_target}
    COMMAND "${CMAKE_COMMAND}"
    "-DEXAMPLE_EXECUTABLE=$<TARGET_FILE:${example_target}>"
    "-DFILECHECK_ENABLED=${THRUST_ENABLE_EXAMPLE_FILECHECK}"
    "-DFILECHECK_EXECUTABLE=${THRUST_FILECHECK_EXECUTABLE}"
    "-DREFERENCE_FILE=${filecheck_data_path}/${filecheck_reference_file}"
    -P "${Thrust_SOURCE_DIR}/cmake/ThrustRunExample.cmake"
  )

  # Run OMP/TBB tests in serial. Multiple OMP processes will massively
  # oversubscribe the machine with GCC's OMP, and we want to test these with
  # the full CPU available to each unit test.
  set(config_systems ${config_host} ${config_device})
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    set_tests_properties(${example_target} PROPERTIES RUN_SERIAL ON)
  endif()
endfunction()

file(GLOB example_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  foreach(example_src IN LISTS example_srcs)
    get_filename_component(example_name "${example_src}" NAME_WLE)
    thrust_add_example(example_target ${example_name} "${example_src}" ${thrust_target})
  endforeach()
endforeach()

add_subdirectory(cmake)
add_subdirectory(cuda)
