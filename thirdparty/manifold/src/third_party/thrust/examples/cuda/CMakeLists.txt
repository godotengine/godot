file(GLOB example_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  if (NOT config_device STREQUAL "CUDA")
    continue()
  endif()

  foreach(example_src IN LISTS example_srcs)
    get_filename_component(example_name "${example_src}" NAME_WLE)
    string(PREPEND example_name "cuda.")
    thrust_add_example(example_target ${example_name} "${example_src}" ${thrust_target})
  endforeach()
endforeach()
